"""HIL-SERL on Spot in viser (http://localhost:<port>). Robot logic adapted from dso-demo's viser_viz: the URDF from
robot_description + joint_states placed by TF, camera images, Trigger buttons, battery.

Scene, in `world_frame`: Spot from /<robot>/robot_description (latched) and /<robot>/joint_states. Its meshes need the
spot_description package (e.g. dso-demo's install sourced); without it only the body and hand frames show. From SpotEnv's status
(`status_topic`, JSON): the hand target and the arm's workspace box in the body frame, home and the base's box (at the
height of `ground_frame`, like the grid).

GUI:
- Episode: phase, episode and step, who drives, the reward classifier's success probability, success and intervention
  rates over the last 20 episodes, and the live hand position (body frame) for calibrating the arm box.
- Controls: buttons publishing SpotEnv's events (`event_topic`): start, success, failure, stop, set_home, save. These are
  the GUI's only way to act on the robot, apart from the Spot folder.
- Spot: a button per `commands` entry (spot_ros2 Trigger services), its reply below. These move the robot.
- Battery; Cameras: each of `cameras` (CompressedImage topics, turned `camera_rotations` degrees clockwise), and the
  policy's 128x128 inputs side by side (`policy_images_topic`).
"""

import io
import json
import threading
import time

import cv2
import numpy as np
import rclpy
import viser
import yourdfpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, JointState
from spot_msgs.msg import BatteryState, BatteryStateArray
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from viser.extras import ViserUrdf

CONTROLS = (  # button label, event, color
    ("Start episode", "start", "green"),
    ("Success", "success", "teal"),
    ("Failure", "failure", "orange"),
    ("Stop (stow arm, sit)", "stop", "red"),
    ("Set home here", "set_home", None),
    ("Save data", "save", None),
)
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def resolve_mesh(fname: str) -> str:
    if fname.startswith("package://"):
        pkg, _, rel = fname[len("package://"):].partition("/")
        return f"{get_package_share_directory(pkg)}/{rel}"
    return fname.removeprefix("file://")


def yaw_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


class ViserGui(Node):
    def __init__(self):
        super().__init__("viser_gui")
        port = self.declare_parameter("port", 8080).value
        rate = float(self.declare_parameter("rate", 20.0).value)
        self.robot = self.declare_parameter("robot", "spot").value
        self.world = self.declare_parameter("world_frame", "spot/vision").value
        self.ground = self.declare_parameter("ground_frame", "spot/feet_center").value
        self.hand_frame = self.declare_parameter("hand_frame", "spot/hand").value
        cameras = self.declare_parameter("cameras", Parameter.Type.STRING_ARRAY).value or []
        rotations = self.declare_parameter("camera_rotations", Parameter.Type.INTEGER_ARRAY).value or [0] * len(cameras)
        self.camera_width = self.declare_parameter("camera_width", 480).value
        self.camera_period = 1.0 / float(self.declare_parameter("camera_rate", 10.0).value)
        commands = self.declare_parameter("commands", Parameter.Type.STRING_ARRAY).value or []
        battery = self.declare_parameter("battery_topic", "").value

        self.server = viser.ViserServer(port=port)
        self.grid = self.server.scene.add_grid("/grid", width=10.0, height=10.0)
        self.status: dict | None = None
        self.status_lock = threading.Lock()

        with self.server.gui.add_folder("Episode"):
            self.phase_text = self.server.gui.add_markdown("Waiting for SpotEnv's status")
            self.episode_text = self.server.gui.add_markdown("")
            self.server.gui.add_markdown("Reward classifier: success probability")
            self.prob_bar = self.server.gui.add_progress_bar(0.0, color="gray")
            self.rates_text = self.server.gui.add_markdown("")
            self.hand_text = self.server.gui.add_markdown("")
        events = self.create_publisher(String, self.declare_parameter("event_topic", "/hil_serl/event").value, QoSProfile(depth=20))
        with self.server.gui.add_folder("Controls"):
            for label, event, color in CONTROLS:
                button = self.server.gui.add_button(label, color=color)
                button.on_click(lambda _, e=event: events.publish(String(data=e)))
        if commands:
            self._add_command_gui(commands)
        self.battery = None
        if battery:
            with self.server.gui.add_folder("Battery"):
                self.battery = (self.server.gui.add_markdown("no data"), self.server.gui.add_progress_bar(0.0, color="gray"))
            self.create_subscription(BatteryStateArray, battery, self._on_battery, 1)

        self.cam_folder = self.server.gui.add_folder("Cameras")
        self.cam_images: dict[str, viser.GuiImageHandle] = {}
        self.cam_last: dict[str, float] = {}
        for topic, rot in zip(cameras, rotations):
            self.create_subscription(
                CompressedImage, topic, lambda m, t=topic, r=ROTATIONS[rot]: self._on_image(t, r, m), qos_profile_sensor_data
            )
        self.create_subscription(
            CompressedImage, self.declare_parameter("policy_images_topic", "/hil_serl/policy_images").value,
            self._on_policy_images, 1,
        )
        self.create_subscription(String, self.declare_parameter("status_topic", "/hil_serl/status").value, self._on_status, 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.vis: ViserUrdf | None = None
        self.root = self.server.scene.add_frame(f"/{self.robot}", show_axes=False, visible=False)
        self.root_frame = ""
        self.joints: dict[str, float] = {}
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, f"/{self.robot}/robot_description", self._on_description, latched)
        self.create_subscription(
            JointState, f"/{self.robot}/joint_states", lambda m: self.joints.update(zip(m.name, m.position)), 10
        )
        self.body = self.server.scene.add_frame("/body", axes_length=0.2, axes_radius=0.008, visible=False)
        self.arm_box = self.server.scene.add_box("/body/arm_box", color=(40, 160, 255), wireframe=True, visible=False)
        self.hand_target = self.server.scene.add_frame("/body/hand_target", axes_length=0.12, axes_radius=0.006, visible=False)
        self.hand = self.server.scene.add_frame("/hand", axes_length=0.1, axes_radius=0.005, visible=False)
        self.home = self.server.scene.add_frame("/home", axes_length=0.3, axes_radius=0.01, visible=False)
        self.base_box = self.server.scene.add_box("/home/base_box", color=(255, 160, 40), wireframe=True)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"viser at http://localhost:{port}, world frame '{self.world}'")

    def _add_command_gui(self, services: list[str]):
        with self.server.gui.add_folder(f"{self.robot} commands"):
            buttons = [self.server.gui.add_button(s.replace("_", " ").capitalize()) for s in services]
            status = self.server.gui.add_markdown("")
        for service, button in zip(services, buttons):
            client = self.create_client(Trigger, f"/{self.robot}/{service}")
            button.on_click(lambda _, c=client, s=service: self._trigger(s, c, buttons, status))

    def _trigger(self, service: str, client, buttons: list, status: viser.GuiMarkdownHandle):
        """Call a Trigger service, the buttons disabled until it answers (from viser's thread)."""
        if not client.service_is_ready():
            status.content = f"**{service}**: no service (is the driver running?)"
            return
        for b in buttons:
            b.disabled = True
        status.content = f"**{service}**: ..."
        done = threading.Event()
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda _: done.set())
        if done.wait(30.0):
            res = future.result()
            status.content = f"**{service}**: {'ok' if res.success else 'failed'}. {res.message}"
            self.get_logger().info(f"/{self.robot}/{service}: {res.success}, {res.message}")
        else:
            status.content = f"**{service}**: no answer"
        for b in buttons:
            b.disabled = False

    def _on_description(self, msg: String):
        robot = yourdfpy.URDF.load(io.StringIO(msg.data), build_scene_graph=False, load_meshes=False).robot
        try:
            urdf = yourdfpy.URDF(robot=robot, filename_handler=resolve_mesh)
        except Exception as e:  # spot_description not found: no meshes
            self.get_logger().warn(f"Spot's meshes unavailable ({e}): showing its body frame only")
            return
        if self.vis is not None:
            self.vis.remove()
        self.vis = ViserUrdf(self.server, urdf, root_node_name=f"/{self.robot}")
        root = urdf.base_link
        self.root_frame = root if root.startswith(f"{self.robot}/") else f"{self.robot}/{root}"
        self.get_logger().info(f"Loaded the URDF, root frame '{self.root_frame}'")

    def _on_battery(self, msg: BatteryStateArray):
        if not msg.battery_states:
            return
        b = msg.battery_states[0]  # Spot has one
        left = b.estimated_runtime.sec
        content = f"{b.charge_percentage:.0f}%, {left // 3600}:{left // 60 % 60:02d} h left" + (
            ", charging" if b.status == BatteryState.STATUS_CHARGING else ""
        )
        text, bar = self.battery
        if text.content == content:  # 16 Hz from spot_ros2: update only on change
            return
        text.content = content
        bar.value = b.charge_percentage
        bar.color = "red" if b.charge_percentage < 20 else "yellow" if b.charge_percentage < 50 else "green"

    def _decode(self, msg: CompressedImage) -> np.ndarray | None:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _show(self, label: str, img: np.ndarray):
        if label in self.cam_images:
            self.cam_images[label].image = img
        else:
            with self.cam_folder:
                self.cam_images[label] = self.server.gui.add_image(img, label=label, format="jpeg")

    def _on_image(self, topic: str, rotation: int | None, msg: CompressedImage):
        now = time.monotonic()
        if now - self.cam_last.get(topic, 0.0) < self.camera_period:
            return
        self.cam_last[topic] = now
        img = self._decode(msg)
        if img is None:
            self.get_logger().warn(f"{topic}: failed to decode image", throttle_duration_sec=5.0)
            return
        if rotation is not None:
            img = cv2.rotate(img, rotation)
        h, w = img.shape[:2]
        self._show(topic, cv2.resize(img, (self.camera_width, round(h * self.camera_width / w))))

    def _on_policy_images(self, msg: CompressedImage):
        img = self._decode(msg)
        if img is not None:
            self._show("policy inputs", cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))

    def _on_status(self, msg: String):
        with self.status_lock:
            self.status = json.loads(msg.data)

    def _lookup(self, frame: str):
        """Pose of `frame` in the world as viser (position, wxyz), None without TF."""
        try:
            tf = self.tf_buffer.lookup_transform(self.world, frame, Time()).transform
        except TransformException:
            return None
        # geometry_msgs quaternion is (x,y,z,w); viser wants (w,x,y,z).
        t, r = tf.translation, tf.rotation
        return np.array([t.x, t.y, t.z]), (r.w, r.x, r.y, r.z)

    def _tick(self):
        ground = self._lookup(self.ground)
        z = 0.0 if ground is None else ground[0][2]
        self.grid.position = (0.0, 0.0, z)
        if self.vis is not None and (pose := self._lookup(self.root_frame)) is not None:
            self.root.position, self.root.wxyz = pose
            self.root.visible = True
            self.vis.update_cfg(np.array([self.joints.get(j, 0.0) for j in self.vis.get_actuated_joint_names()]))
        if (pose := self._lookup(self.hand_frame)) is not None:
            self.hand.position, self.hand.wxyz = pose
            self.hand.visible = True
        with self.status_lock:
            s = self.status
        if s is None:
            return
        self._update_gui(s)
        if (pose := self._lookup(s["body_frame"])) is not None:
            self.body.position, self.body.wxyz = pose
            self.body.visible = True
            low, high = np.array(s["arm_low"]), np.array(s["arm_high"])
            self.arm_box.position = (low + high) / 2
            self.arm_box.dimensions = tuple(np.maximum(high - low, 0.01))
            self.arm_box.visible = True
            self.hand_target.position = s["hand_target"]
            self.hand_target.wxyz = np.roll(Rotation.from_euler("xyz", s["hand_rpy"]).as_quat(), 1)
            self.hand_target.visible = True
        if s["home"] is not None:
            x, y, yaw = s["home"]
            self.home.position, self.home.wxyz = (x, y, z), yaw_wxyz(yaw)
            self.home.visible = True
            self.base_box.dimensions = (2 * s["base_limit"][0], 2 * s["base_limit"][1], 0.01)

    def _update_gui(self, s: dict):
        phase = f"**{s['phase']}**" + (f": {s['message']}" if s["message"] else "")
        driver = "human (gamepad)" if s["intervening"] else "policy"
        episode = (
            f"Episode **{s['episode']}**, step {s['step']} / {s['max_steps']}, driven by **{driver}**, "
            f"intervened {100 * s['intervention_rate']:.0f}% of steps" + (", **success**" if s["succeed"] else "")
        )
        rates = "No finished episodes yet"
        if s["recent_episodes"]:
            rates = (
                f"Last {s['recent_episodes']} episodes: success {100 * s['recent_success_rate']:.0f}%, "
                f"interventions {100 * s['recent_intervention_rate']:.0f}% of steps"
            )
        hand = "Hand: no TF" if s["hand"] is None else "Hand (body frame): x {:.3f}, y {:.3f}, z {:.3f} m".format(*s["hand"])
        hand += f", gripper {100 * s['gripper']:.0f}% open"
        for handle, content in ((self.phase_text, phase), (self.episode_text, episode), (self.rates_text, rates), (self.hand_text, hand)):
            if handle.content != content:
                handle.content = content
        prob = 100 * s["classifier_prob"]
        if abs(self.prob_bar.value - prob) > 0.5:
            self.prob_bar.value = prob
            self.prob_bar.color = "green" if prob > 75 else "gray"


def main():
    rclpy.init()
    node = ViserGui()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
