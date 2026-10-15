"""Stand-in for the spot_ros2 driver: dry-runs SpotEnv, the gamepad and the GUI without the robot.

Run it on a ROS_DOMAIN_ID of its own. It refuses to start if anything else publishes <ns>/joint_states or subscribes
<ns>/cmd_vel (a real driver).

A kinematic Spot: cmd_vel moves the body (each command held `cmd_duration`, as in spot_ros2), arm_pose_commands (body
frame) move the hand toward the target at `hand_speed`, open_gripper/close_gripper set the claw, and the other Trigger
services just succeed. Publishes TF (<prefix>vision -> body -> hand, body -> feet_center), joint_states (arm_f1x) and
the hand and frontleft cameras: top-down renders of a checkered floor with a red plushy and a blue bin (frontleft
sideways, as mounted). Closing the gripper with the hand low over the plushy picks it up; opening drops it under the
hand.
"""

import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

BODY_HEIGHT = 0.5  # m, body above the floor
GRIPPER_OPEN = -1.5708
STOWED = np.array([0.55, 0.0, 0.25])
READY = np.array([0.7, 0.0, 0.3])


def render(center: np.ndarray, yaw: float, size_m: tuple[float, float], shape: tuple[int, int], plushy, bin_) -> np.ndarray:
    """Top-down BGR view of the floor, centred at `center` (x, y), image up along `yaw`, size_m = (width, height)."""
    w, h = shape
    right, fwd = np.meshgrid(np.linspace(-size_m[0] / 2, size_m[0] / 2, w), np.linspace(size_m[1] / 2, -size_m[1] / 2, h))
    c, s = np.cos(yaw), np.sin(yaw)
    x = center[0] + c * fwd + s * right  # image right is the view's -y
    y = center[1] + s * fwd - c * right
    img = np.where(((np.floor(x / 0.25) + np.floor(y / 0.25)) % 2 == 0)[..., None], 170, 120).astype(np.uint8)
    img = np.repeat(img, 3, axis=2)
    (bx, by), half = bin_
    in_bin = (np.abs(x - bx) < half) & (np.abs(y - by) < half)
    img[in_bin] = (160, 60, 30)
    img[in_bin & ((np.abs(x - bx) > half - 0.04) | (np.abs(y - by) > half - 0.04))] = (200, 90, 40)
    img[np.hypot(x - plushy[0], y - plushy[1]) < 0.08] = (40, 40, 220)
    return img


class SpotMock(Node):
    def __init__(self):
        super().__init__("spot_mock")
        self.prefix = self.declare_parameter("frame_prefix", "spot/").value
        self.cmd_duration = float(self.declare_parameter("cmd_duration", 0.125).value)
        self.hand_speed = float(self.declare_parameter("hand_speed", 0.5).value)
        self.plushy = np.array(self.declare_parameter("plushy", [1.4, 0.3]).value)
        self.bin = np.array(self.declare_parameter("bin", [0.6, -1.3]).value), 0.2
        self._refuse_if_real_driver()
        self.pose = np.zeros(3)  # x, y, yaw in vision
        self.velocity, self.velocity_until = np.zeros(3), 0.0
        self.hand, self.hand_target = STOWED.copy(), STOWED.copy()
        self.hand_q = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w in the body frame, set instantly by arm_pose_commands
        self.gripper = 0.0
        self.held = False
        self.last = time.monotonic()
        self.tf = TransformBroadcaster(self)
        self.joints = self.create_publisher(JointState, "joint_states", 10)
        self.cameras = {
            name: self.create_publisher(CompressedImage, f"camera/{name}/compressed", 1) for name in ("hand", "frontleft")
        }
        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 1)
        self.create_subscription(PoseStamped, "arm_pose_commands", self._on_arm_pose, 10)
        for name in ("open_gripper", "close_gripper", "arm_stow", "arm_unstow", "stand", "sit", "claim", "release",
                     "power_on", "power_off"):
            self.create_service(Trigger, name, lambda req, res, n=name: self._on_service(n, res))
        self.create_timer(0.02, self._tick)
        self.create_timer(0.1, self._publish_cameras)
        self.get_logger().info("Mock Spot up: plushy (red) at %s, bin (blue) at %s" % (self.plushy, self.bin[0]))

    def _refuse_if_real_driver(self):
        time.sleep(2.0)  # discovery
        if self.count_publishers("joint_states") or self.count_subscribers("cmd_vel"):
            raise SystemExit(f"Something else serves {self.get_namespace()} on ROS_DOMAIN_ID: not mocking next to a real Spot")

    def _on_cmd_vel(self, msg: Twist):
        self.velocity = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.velocity_until = time.monotonic() + self.cmd_duration

    def _on_arm_pose(self, msg: PoseStamped):
        if msg.header.frame_id != "body":
            self.get_logger().warn(f"arm_pose_commands in '{msg.header.frame_id}', only 'body' is mocked")
            return
        p, q = msg.pose.position, msg.pose.orientation
        self.hand_target = np.array([p.x, p.y, p.z])
        self.hand_q = (q.x, q.y, q.z, q.w)

    def _hand_world(self) -> np.ndarray:
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        return self.pose[:2] + np.array([c * self.hand[0] - s * self.hand[1], s * self.hand[0] + c * self.hand[1]])

    def _on_service(self, name: str, res: Trigger.Response) -> Trigger.Response:
        if name == "open_gripper":
            self.gripper, self.held = GRIPPER_OPEN, False
        elif name == "close_gripper":
            self.gripper = 0.0
            low = self.hand[2] + BODY_HEIGHT < 0.15
            self.held = low and np.linalg.norm(self._hand_world() - self.plushy) < 0.1
            if self.held:
                self.gripper = -0.4  # closed on the plushy
        elif name in ("arm_stow", "arm_unstow"):
            self.hand = self.hand_target = (STOWED if name == "arm_stow" else READY).copy()
        res.success, res.message = True, f"mock {name}"
        self.get_logger().info(f"{name}" + (" (picked up the plushy)" if name == "close_gripper" and self.held else ""))
        return res

    def _tick(self):
        now = time.monotonic()
        dt, self.last = now - self.last, now
        v = self.velocity if now < self.velocity_until else np.zeros(3)
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose += dt * np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
        step = self.hand_target - self.hand
        self.hand += step * min(1.0, self.hand_speed * dt / max(np.linalg.norm(step), 1e-9))
        if self.held:
            self.plushy = self._hand_world()
        stamp = self.get_clock().now().to_msg()
        q = (0.0, 0.0, np.sin(self.pose[2] / 2), np.cos(self.pose[2] / 2))
        self.tf.sendTransform([
            self._transform(stamp, "vision", "body", (*self.pose[:2], BODY_HEIGHT), q),
            self._transform(stamp, "body", "hand", self.hand, self.hand_q),
            self._transform(stamp, "body", "feet_center", (0.0, 0.0, -BODY_HEIGHT), (0.0, 0.0, 0.0, 1.0)),
        ])
        self.joints.publish(JointState(header=Header(stamp=stamp), name=[f"{self.prefix}arm_f1x"], position=[self.gripper]))

    def _transform(self, stamp, parent: str, child: str, xyz, q) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.prefix + parent, self.prefix + child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(a) for a in xyz)
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = (float(a) for a in q)
        return t

    def _publish_cameras(self):
        stamp = self.get_clock().now().to_msg()
        height = max(self.hand[2] + BODY_HEIGHT, 0.05)  # hand camera: looks down from the hand
        hand = render(self._hand_world(), self.pose[2], (1.4 * height, 1.05 * height), (320, 240), self.plushy, self.bin)
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        ahead = self.pose[:2] + 1.6 * np.array([c, s])  # frontleft: the floor 0-3.2 m ahead, grayscale, sideways
        front = render(ahead, self.pose[2], (2.4, 3.2), (240, 320), self.plushy, self.bin)
        front = cv2.rotate(cv2.cvtColor(cv2.cvtColor(front, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
                           cv2.ROTATE_90_COUNTERCLOCKWISE)
        for name, img in (("hand", hand), ("frontleft", front)):
            msg = CompressedImage(format="jpeg", data=cv2.imencode(".jpg", cv2.resize(img, (640, 480)))[1].tobytes())
            msg.header.stamp, msg.header.frame_id = stamp, f"{self.prefix}{name}"
            self.cameras[name].publish(msg)


def main():
    rclpy.init()
    try:
        rclpy.spin(SpotMock())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
