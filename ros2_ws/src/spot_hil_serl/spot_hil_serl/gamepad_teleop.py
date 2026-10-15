"""Gamepad (joy's game_controller_node on /joy, SDL's standard layout) -> SpotEnv's teleop and events.

/hil_serl/teleop (sensor_msgs/Joy), on every /joy message:
  axes [vx, vy, wz, arm_x, arm_z, roll] in [-1, 1]: the left stick walks (forward/back, strafe), the triggers turn
  (left trigger left, right trigger right), the right stick moves the hand in Spot's side view (up/down, and
  right/left = forward/back), the bumpers roll the gripper about its own axis (`roll_buttons`: negative, positive).
  buttons [gripper, success, failure, start, stop, pitch], held (A, B, X, Start, Back, Y by default).
/hil_serl/event (std_msgs/String) on each press: gripper_toggle, success, failure, start, stop, pitch_toggle (Y: the
gripper pointing forward <-> down).

Each output axis is `axis_scales[i] * /joy axes[axis_indices[i]]`, zero within `deadzone` and rescaled outside it so it
still reaches ±1. A trigger reads `trigger_rest` released and `trigger_pressed` fully pressed. It stays 0 until first seen
released, since some controllers report a half-pressed trigger until it is touched.
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy
from std_msgs.msg import String

EVENTS = ("gripper_toggle", "success", "failure", "start", "stop", "pitch_toggle")  # one per entry of `buttons`


class GamepadTeleop(Node):
    def __init__(self):
        super().__init__("gamepad_teleop")
        # game_controller_node: axes 0 left x (left +), 1 left y (up +), 2 right x, 3 right y, 4/5 left/right trigger.
        self.axis_indices = list(self.declare_parameter("axis_indices", [1, 0, 3, 2]).value)  # vx, vy, arm_z, arm_x
        self.axis_scales = np.array(self.declare_parameter("axis_scales", [1.0, 1.0, 1.0, -1.0]).value)
        self.triggers = list(self.declare_parameter("trigger_indices", [4, 5]).value)  # turn left, turn right
        self.trigger_rest = float(self.declare_parameter("trigger_rest", 0.0).value)
        self.trigger_pressed = float(self.declare_parameter("trigger_pressed", -1.0).value)
        self.buttons = list(self.declare_parameter("buttons", [0, 1, 2, 6, 4, 3]).value)  # A, B, X, Start, Back, Y
        self.roll_buttons = list(self.declare_parameter("roll_buttons", [9, 10]).value)  # LB negative, RB positive
        self.deadzone = float(self.declare_parameter("deadzone", 0.1).value)
        self.armed = [False, False]
        self.held = [False] * len(EVENTS)
        self.raw_held: list[int] = []
        self.teleop = self.create_publisher(Joy, self.declare_parameter("teleop_topic", "/hil_serl/teleop").value, 1)
        self.events = self.create_publisher(
            String, self.declare_parameter("event_topic", "/hil_serl/event").value, QoSProfile(depth=20)
        )
        self.create_subscription(Joy, "/joy", self._on_joy, 10)

    def _dead(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.clip((np.abs(x) - self.deadzone) / (1.0 - self.deadzone), 0.0, 1.0)

    def _trigger(self, i: int, axes: list[float]) -> float:
        value = axes[self.triggers[i]]
        self.armed[i] |= abs(value - self.trigger_rest) < 0.05
        pressed = (value - self.trigger_rest) / (self.trigger_pressed - self.trigger_rest)
        return float(np.clip(pressed, 0.0, 1.0)) if self.armed[i] else 0.0

    def _on_joy(self, msg: Joy):
        sticks = self._dead(self.axis_scales * np.array([msg.axes[i] for i in self.axis_indices]))
        turn = self._dead(np.array([self._trigger(0, msg.axes) - self._trigger(1, msg.axes)]))[0]
        for i, b in enumerate(msg.buttons):
            if b and (i >= len(self.raw_held) or not self.raw_held[i]):
                self.get_logger().info(f"/joy button {i} pressed")
        self.raw_held = list(msg.buttons)
        held = [bool(msg.buttons[i]) for i in self.buttons]
        out = Joy(header=msg.header)
        roll = float(msg.buttons[self.roll_buttons[1]]) - float(msg.buttons[self.roll_buttons[0]])
        out.axes = [float(x) for x in (sticks[0], sticks[1], turn, sticks[3], sticks[2], roll)]
        out.buttons = [int(b) for b in held]
        self.teleop.publish(out)
        for name, now, before in zip(EVENTS, held, self.held):
            if now and not before:
                self.events.publish(String(data=name))
                self.get_logger().info(f"Event: {name}")
        self.held = held


def main():
    rclpy.init()
    node = GamepadTeleop()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
