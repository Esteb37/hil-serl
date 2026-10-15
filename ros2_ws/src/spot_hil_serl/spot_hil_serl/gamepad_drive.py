"""Drive Spot with the gamepad, no SpotEnv: gamepad_teleop's /hil_serl/teleop and /hil_serl/event -> the spot_ros2
driver's cmd_vel, arm_pose_commands and Trigger services (run it in the driver's namespace). Never run it next to a
SpotEnv script: both would command the robot.

Nothing moves until Start, which calls the `start` services (stand, arm_unstow). Then the left stick walks, the triggers
turn and the right stick moves the hand (body frame x, z) at `hand_speed`, within the `hand_low`..`hand_high` box, at
least `floor_clearance` above `floor_frame` and at most `hand_max_error` ahead of the measured hand. Y switches the
gripper between pointing forward (after Start) and down (`hand_pitch_down`), turning `pitch_step` per tick; the bumpers
roll it about its own axis, `roll_step` per tick within ±`roll_limit` (as SpotEnv). A toggles the gripper. Back stops
the base and calls the `stop` services (arm_stow, sit), and motion waits for the next Start.
"""

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


class GamepadDrive(Node):
    def __init__(self):
        super().__init__("gamepad_drive")
        self.rate = float(self.declare_parameter("rate", 10.0).value)
        self.base_scale = np.array(self.declare_parameter("base_scale", [0.4, 0.3, 0.6]).value)  # m/s, m/s, rad/s
        self.hand_step = float(self.declare_parameter("hand_speed", 0.3).value) / self.rate
        self.hand_max_error = float(self.declare_parameter("hand_max_error", 0.05).value)
        self.hand_low = np.array(self.declare_parameter("hand_low", [0.55, 0.0, -0.42]).value)
        self.hand_high = np.array(self.declare_parameter("hand_high", [1.0, 0.0, 0.2]).value)
        self.pitch_down_angle = float(self.declare_parameter("hand_pitch_down", 1.5708).value)
        self.pitch_step = float(self.declare_parameter("pitch_step", 0.25).value)
        self.roll_step = float(self.declare_parameter("roll_step", 0.15).value)
        self.roll_limit = float(self.declare_parameter("roll_limit", 1.5708).value)
        self.floor_frame = self.declare_parameter("floor_frame", "spot/feet_center").value
        self.floor_clearance = float(self.declare_parameter("floor_clearance", 0.03).value)
        self.body_frame = self.declare_parameter("body_frame", "spot/body").value
        self.hand_frame = self.declare_parameter("hand_frame", "spot/hand").value
        self.arm_command_frame = self.declare_parameter("arm_command_frame", "body").value
        self.gripper_joint = self.declare_parameter("gripper_joint", "arm_f1x").value
        self.gripper_open_angle = float(self.declare_parameter("gripper_open_angle", -1.5708).value)
        self.teleop_timeout = float(self.declare_parameter("teleop_timeout", 0.3).value)
        self.service_timeout = float(self.declare_parameter("service_timeout", 20.0).value)
        self.start_services = list(self.declare_parameter("start", ["stand", "arm_unstow"]).value)
        self.stop_services = list(self.declare_parameter("stop", ["arm_stow", "sit"]).value)

        self.active = False  # after a successful Start, until Back
        self.axes = np.zeros(6)  # vx, vy, wz, arm_x, arm_z, roll
        self.pitch_down = False  # Y; self.pitch turns toward it
        self.pitch = 0.0
        self.roll = 0.0
        self.teleop_time = 0.0
        self.walking = False
        self.hand_target: np.ndarray | None = None
        self.gripper_angle: float | None = None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.cmd_vel = self.create_publisher(Twist, "cmd_vel", 1)
        self.arm_pose = self.create_publisher(PoseStamped, "arm_pose_commands", 1)
        services = ReentrantCallbackGroup()
        self.triggers = {
            name: self.create_client(Trigger, name, callback_group=services)
            for name in {*self.start_services, *self.stop_services, "open_gripper", "close_gripper"}
        }
        self.create_subscription(JointState, "joint_states", self._on_joints, qos_profile_sensor_data)
        teleop_topic = self.declare_parameter("teleop_topic", "/hil_serl/teleop").value
        event_topic = self.declare_parameter("event_topic", "/hil_serl/event").value
        self.create_subscription(Joy, teleop_topic, self._on_teleop, 1)
        # Events run one at a time in their own group: a service sequence blocks them, not the motion timer.
        self.create_subscription(String, event_topic, self._on_event, QoSProfile(depth=20),
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info("Press Start to stand up and unstow the arm.")

    def _on_joints(self, msg: JointState):
        for name, q in zip(msg.name, msg.position):
            if name.endswith(self.gripper_joint):
                self.gripper_angle = q

    def _on_teleop(self, msg: Joy):
        self.axes = np.clip(np.asarray(msg.axes[:6], dtype=np.float64), -1.0, 1.0)
        self.teleop_time = time.monotonic()

    def _on_event(self, msg: String):
        if msg.data == "start":
            self.active = False
            self.hand_target = None  # re-anchor to the measured hand
            self.pitch_down, self.pitch, self.roll = False, 0.0, 0.0  # arm_unstow points the gripper forward
            self.active = self._call_all(self.start_services)
            self.get_logger().info("Driving." if self.active else "Start failed: not driving.")
        elif msg.data == "stop":
            self.active = False
            self._send_velocity(np.zeros(3))
            self._call_all(self.stop_services)
        elif msg.data == "pitch_toggle" and self.active:
            self.pitch_down = not self.pitch_down
            self.get_logger().info(f"Gripper pointing {'down' if self.pitch_down else 'forward'}")
        elif msg.data == "gripper_toggle" and self.active:
            if self.gripper_angle is None:
                self.get_logger().warn("No gripper joint state: not toggling the gripper.")
                return
            is_open = self.gripper_angle / self.gripper_open_angle > 0.5
            self._call("close_gripper" if is_open else "open_gripper")

    def _tick(self):
        fresh = time.monotonic() - self.teleop_time < self.teleop_timeout
        axes = self.axes if fresh and self.active else np.zeros(6)
        v = axes[:3] * self.base_scale
        if v.any() or self.walking:  # one zero on release, then quiet
            self._send_velocity(v)
        self.walking = bool(v.any())
        if not self.active:
            return
        pitch, roll = self.pitch, self.roll
        self.pitch += np.clip((self.pitch_down_angle if self.pitch_down else 0.0) - self.pitch, -self.pitch_step, self.pitch_step)
        self.roll = float(np.clip(self.roll + axes[5] * self.roll_step, -self.roll_limit, self.roll_limit))
        if axes[3:5].any() or (self.pitch, self.roll) != (pitch, roll):
            self._move_hand(axes[3:5])

    def _move_hand(self, d: np.ndarray):
        try:
            t = self.tf.lookup_transform(self.body_frame, self.hand_frame, Time()).transform.translation
        except TransformException as e:
            self.get_logger().warn(f"No hand pose: {e}", throttle_duration_sec=2.0)
            return
        hand = np.array([t.x, t.y, t.z])
        target = self.hand_target
        target = np.clip(hand, self.hand_low, self.hand_high) if target is None else target.copy()
        target[[0, 2]] += d * self.hand_step
        target = np.clip(target, hand - self.hand_max_error, hand + self.hand_max_error)
        target = np.clip(target, self.hand_low, self.hand_high)
        try:
            floor = self.tf.lookup_transform(self.body_frame, self.floor_frame, Time()).transform.translation.z
            target[2] = max(target[2], floor + self.floor_clearance)
        except TransformException as e:
            self.get_logger().warn(f"No floor frame, not moving the hand: {e}", throttle_duration_sec=2.0)
            return
        self.hand_target = target
        msg = PoseStamped()
        msg.header.frame_id = self.arm_command_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        pos, q = msg.pose.position, msg.pose.orientation
        pos.x, pos.y, pos.z = (float(x) for x in self.hand_target)
        q.x, q.y, q.z, q.w = (float(x) for x in Rotation.from_euler("xyz", [self.roll, self.pitch, 0.0]).as_quat())
        self.arm_pose.publish(msg)

    def _send_velocity(self, v: np.ndarray):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = (float(x) for x in v)
        self.cmd_vel.publish(msg)

    def _call_all(self, names: list[str]) -> bool:
        """Call `names` in order; stop at the first failure."""
        return all(self._call(name) for name in names)

    def _call(self, name: str) -> bool:
        client = self.triggers[name]
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"{client.srv_name}: no such service (is spot_ros2 running?)")
            return False
        done = threading.Event()
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda _: done.set())
        if not done.wait(self.service_timeout):
            self.get_logger().error(f"{client.srv_name}: no reply within {self.service_timeout:.0f} s")
            return False
        res = future.result()
        log = self.get_logger().info if res.success else self.get_logger().error
        log(f"{client.srv_name}: {res.message}")
        return res.success


def main():
    rclpy.init()
    node = GamepadDrive()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node._send_velocity(np.zeros(3))
        node.destroy_node()
        rclpy.try_shutdown()
