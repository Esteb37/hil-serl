"""SpotEnv's ROS 2 side: one rclpy node, in its own context and spun in a background thread.

Talks to the spot_ros2 driver (cmd_vel, arm_pose_commands, Trigger services, joint_states, TF, compressed cameras), to
the gamepad (spot_hil_serl's gamepad_teleop: /hil_serl/teleop and /hil_serl/event) and to the viser GUI (the same
events in, /hil_serl/status and /hil_serl/policy_images out).
"""

import threading
import time
from collections import Counter

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.context import Context
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, JointState, Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

TELEOP_BUTTONS = ("gripper", "success", "failure", "start", "stop", "pitch")  # gamepad_teleop's button order


class SpotRos:
    def __init__(self, cfg):
        self.cfg = cfg
        # Own context, no signal handlers: Ctrl-C reaches the caller while the node can still stop the robot.
        self.context = Context()
        rclpy.init(context=self.context, signal_handler_options=SignalHandlerOptions.NO)
        self.node = Node("spot_env", context=self.context)
        self.lock = threading.Lock()
        self.images: dict[str, CompressedImage] = {}
        self.image_time: dict[str, float] = {}
        self.gripper_angle: float | None = None
        self.teleop_axes = np.zeros(6)
        self.teleop_buttons = np.zeros(len(TELEOP_BUTTONS), dtype=bool)
        self.teleop_time = 0.0
        self.events: Counter[str] = Counter()

        t = cfg.TOPICS
        for key, cam in cfg.CAMERAS.items():
            self.node.create_subscription(
                CompressedImage, cam["topic"], lambda m, k=key: self._on_image(k, m), qos_profile_sensor_data
            )
        self.node.create_subscription(JointState, t["joint_states"], self._on_joints, qos_profile_sensor_data)
        self.node.create_subscription(Joy, t["teleop"], self._on_teleop, 1)
        self.node.create_subscription(String, t["event"], self._on_event, QoSProfile(depth=20))
        self.cmd_vel = self.node.create_publisher(Twist, t["cmd_vel"], 1)
        self.arm_pose = self.node.create_publisher(PoseStamped, t["arm_pose"], 1)
        self.status = self.node.create_publisher(String, t["status"], 1)
        self.policy_images = self.node.create_publisher(CompressedImage, t["policy_images"], 1)
        self.clients = {name: self.node.create_client(Trigger, srv) for name, srv in cfg.SERVICES.items()}
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self.node, spin_thread=False)

        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    @property
    def log(self):
        return self.node.get_logger()

    def _on_image(self, key: str, msg: CompressedImage):
        with self.lock:
            self.images[key] = msg
            self.image_time[key] = time.monotonic()

    def _on_joints(self, msg: JointState):
        for name, q in zip(msg.name, msg.position):
            if name.endswith(self.cfg.GRIPPER_JOINT):
                self.gripper_angle = q

    def _on_teleop(self, msg: Joy):
        with self.lock:
            self.teleop_axes = np.clip(np.asarray(msg.axes[:6], dtype=np.float64), -1.0, 1.0)
            self.teleop_buttons = np.asarray(msg.buttons[: len(TELEOP_BUTTONS)], dtype=bool)
            self.teleop_time = time.monotonic()

    def _on_event(self, msg: String):
        with self.lock:
            self.events[msg.data] += 1

    def image(self, key: str) -> tuple[CompressedImage | None, float]:
        """Latest frame of camera `key` and its age (s)."""
        with self.lock:
            return self.images.get(key), time.monotonic() - self.image_time.get(key, -np.inf)

    def teleop(self) -> np.ndarray:
        """Gamepad axes [vx, vy, wz, arm_x, arm_z, roll], zero when gamepad_teleop is silent."""
        with self.lock:
            fresh = time.monotonic() - self.teleop_time < self.cfg.TELEOP_TIMEOUT
            return self.teleop_axes.copy() if fresh else np.zeros(6)

    def button(self, name: str) -> bool:
        """Whether gamepad button `name` (TELEOP_BUTTONS) is held."""
        with self.lock:
            fresh = time.monotonic() - self.teleop_time < self.cfg.TELEOP_TIMEOUT
            return fresh and bool(self.teleop_buttons[TELEOP_BUTTONS.index(name)])

    def pop_event(self, name: str) -> bool:
        """Whether event `name` arrived since the last pop; consumes it."""
        with self.lock:
            return self.events.pop(name, 0) > 0

    def clear_events(self, *names: str):
        with self.lock:
            for name in names or list(self.events):
                self.events.pop(name, None)

    def lookup(self, target: str, source: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Latest pose of `source` in `target`: position, quaternion (x, y, z, w); None without TF."""
        try:
            tf = self.tf.lookup_transform(target, source, Time()).transform
        except TransformException:
            return None
        t, r = tf.translation, tf.rotation
        return np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w])

    def pose2d(self, target: str, source: str) -> np.ndarray | None:
        """(x, y, yaw) of `source` in `target`, None without TF."""
        pose = self.lookup(target, source)
        if pose is None:
            return None
        return np.array([pose[0][0], pose[0][1], Rotation.from_quat(pose[1]).as_euler("xyz")[2]])

    def send_velocity(self, v: np.ndarray):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = (float(x) for x in v)
        self.cmd_vel.publish(msg)

    def send_hand(self, position: np.ndarray, quat_xyzw: np.ndarray):
        msg = PoseStamped()
        msg.header.frame_id = self.cfg.ARM_COMMAND_FRAME
        msg.header.stamp = self.node.get_clock().now().to_msg()
        p, q = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = (float(x) for x in position)
        q.x, q.y, q.z, q.w = (float(x) for x in quat_xyzw)
        self.arm_pose.publish(msg)

    def call(self, name: str, timeout: float) -> bool:
        """Call Trigger service `name` (cfg.SERVICES) and wait up to `timeout` s for its reply."""
        client = self.clients[name]
        if not client.wait_for_service(timeout_sec=min(timeout, 2.0)):
            self.log.error(f"{client.srv_name}: no such service (is the spot_ros2 driver running?)")
            return False
        done = threading.Event()
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            self.log.warn(f"{client.srv_name}: no reply within {timeout:.1f} s")
            return False
        res = future.result()
        if not res.success:
            self.log.warn(f"{client.srv_name} failed: {res.message}")
        return res.success

    def publish_status(self, payload: str):
        self.status.publish(String(data=payload))

    def publish_policy_images(self, jpeg: bytes):
        msg = CompressedImage(format="jpeg", data=jpeg)
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.policy_images.publish(msg)

    def close(self):
        if self.context.ok():
            self.send_velocity(np.zeros(3))
        self.executor.shutdown()
        self.thread.join(timeout=2.0)
        self.node.destroy_node()
        self.context.try_shutdown()
