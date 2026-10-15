"""Gym env for Spot with arm, through ROS 2 and the spot_ros2 driver, for HIL-SERL.

Action (8,) in [-1, 1]: base velocity [vx, vy, wz] (times BASE_SCALE, body frame), hand motion [dx, dz] (ARM_STEP m per
step, body frame), hand pitch mode (>= 0.5 gripper pointing down, <= -0.5 forward, else stay; the wrist turns at
PITCH_STEP per step), hand roll about the gripper's axis (ROLL_STEP per step, within ±ROLL_LIMIT), gripper (<= -0.5
close, >= 0.5 open, else stay: the hybrid agent's discrete grasp action, kept last). The hand target stays in the
ARM_LIMIT box and within ARM_MAX_ERROR of the measured hand (the paper's reference limiting). The base stops moving away
from home once outside BASE_LIMIT. A gripper step stops the base and waits for the gripper.

Observation: images {key: IMAGE_SIZE² RGB crop per CAMERAS entry}, state {hand_pose: hand xyz in the body frame,
hand_rot: the hand's x and y axes in the body frame (rotation matrix columns, no Euler singularity pointing down),
gripper_pos: 1 open .. 0 closed, base_vel: [vx, vy, wz] in the body frame, base_pose: [x, y, yaw] from home}.

reset() opens the gripper, brings the hand to ARM_RESET pointing forward, unrolled, and walks Spot back to home (where it stood at the first reset,
or where the GUI's "Set home" was pressed), offset by RANDOM_RESET_*. It then waits for Start (gamepad or GUI) while
the operator re-places the object, and the gamepad drives Spot meanwhile. Episodes end at MAX_EPISODE_LENGTH, on
failure (X) or on stop (Back). After a stop the arm stows and Spot sits until the next Start.
"""

import json
import os
import time
from collections import deque
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation


class DefaultSpotEnvConfig:
    """SpotEnv's settings. An experiment's EnvConfig overrides the task-specific ones."""

    CAMERAS: dict[str, dict] = {  # image key -> CompressedImage topic and cv2 rotation (None: as is)
        "hand": {"topic": "/spot/camera/hand/compressed", "rotate": None},
        "front": {"topic": "/spot/camera/frontleft/compressed", "rotate": cv2.ROTATE_90_CLOCKWISE},  # mounted sideways
    }
    IMAGE_CROP: dict[str, callable] = {}  # image key -> crop of the rotated image
    IMAGE_SIZE = 128
    TOPICS: dict[str, str] = {
        "cmd_vel": "/spot/cmd_vel",
        "arm_pose": "/spot/arm_pose_commands",
        "joint_states": "/spot/joint_states",
        "teleop": "/hil_serl/teleop",
        "event": "/hil_serl/event",
        "status": "/hil_serl/status",
        "policy_images": "/hil_serl/policy_images",
    }
    SERVICES: dict[str, str] = {  # spot_ros2 Trigger services
        name: f"/spot/{name}" for name in ("open_gripper", "close_gripper", "stand", "sit", "arm_unstow", "arm_stow")
    }
    WORLD_FRAME = "spot/vision"  # home and base_pose (visual odometry: drifts less than odom)
    BODY_FRAME = "spot/body"
    HAND_FRAME = "spot/hand"
    FLOOR_FRAME = "spot/feet_center"  # on the floor under the feet
    ARM_COMMAND_FRAME = "body"  # Spot's own frame name: spot_ros2 hands it to the SDK as is
    GRIPPER_JOINT = "arm_f1x"
    GRIPPER_OPEN_ANGLE = -1.5708  # rad when fully open (0: closed)

    HZ = 10
    MAX_EPISODE_LENGTH = 200
    BASE_SCALE = np.array([0.4, 0.3, 0.6])  # m/s, m/s, rad/s at action ±1
    ARM_STEP = 0.03  # m per step at action ±1
    ARM_MAX_ERROR = 0.05  # m the hand target may lead the measured hand
    HAND_PITCH_DOWN = np.pi / 2  # rad: hand pitch in the body frame pointing down (forward: 0, the reset mode)
    PITCH_STEP = 0.25  # rad per step while switching between forward and down
    ROLL_STEP = 0.15  # rad per step at action ±1, about the gripper's axis
    ROLL_LIMIT = np.pi / 2  # rad
    ARM_RESET = np.array([0.75, 0.0, 0.0])  # hand position in the body frame at reset
    ARM_LIMIT_LOW = np.array([0.55, 0.0, -0.55])  # z: a backstop; ARM_FLOOR_CLEARANCE is the real floor limit
    ARM_LIMIT_HIGH = np.array([1.0, 0.0, 0.2])
    ARM_FLOOR_CLEARANCE = 0.03  # m: the hand frame's origin stays this high above FLOOR_FRAME
    BASE_LIMIT = np.array([2.0, 2.0])  # m: |x|, |y| from home, in home's frame
    RANDOM_RESET_XY = 0.2  # m: uniform start offset from home
    RANDOM_RESET_YAW = 0.3  # rad
    RESET_GAIN = 1.5  # 1/s: walk-back P gain
    RESET_TOLERANCE = np.array([0.05, 0.05])  # m, rad
    RESET_TIMEOUT = 20.0  # s
    GRIPPER_OPEN_THRESHOLD = 0.5  # gripper_pos above this: open
    GRIPPER_TIMEOUT = 2.0  # s for a gripper service to finish
    SERVICE_TIMEOUT = 20.0  # s for stand, sit, arm_stow, arm_unstow
    WAIT_FOR_START = True
    IMAGE_TIMEOUT = 0.5  # s: older frames are logged as stale
    TELEOP_TIMEOUT = 0.3  # s: older teleop reads as released


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose b (x, y, yaw) in the frame of pose a."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose b, given in the frame of pose a, in a's parent frame."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def rotate(v: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class SpotEnv(gym.Env):
    def __init__(self, fake_env: bool = False, save_video: bool = False, config: DefaultSpotEnvConfig = None):
        self.config = config
        self.hz = config.HZ
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        size = config.IMAGE_SIZE
        self.action_space = gym.spaces.Box(-np.ones(8, dtype=np.float32), np.ones(8, dtype=np.float32))
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "hand_pose": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "hand_rot": gym.spaces.Box(-1, 1, shape=(6,)),
                        "gripper_pos": gym.spaces.Box(0, 1, shape=(1,)),
                        "base_vel": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "base_pose": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(size, size, 3), dtype=np.uint8) for key in config.CAMERAS}
                ),
            }
        )
        self.save_video = save_video
        self.recording_frames: list[dict[str, np.ndarray]] = []
        self.episode = 0
        self.curr_path_length = 0
        self.history: deque[tuple[bool, float]] = deque(maxlen=20)  # recent episodes: (succeeded, intervention rate)
        self.intervening = False
        self.intervention_steps = 0
        self.classifier_prob = 0.0
        self.succeeded = False
        if fake_env:
            return

        from spot_env.spot_ros import SpotRos  # ROS 2 only on the robot side, not in the learner

        self.ros = SpotRos(config)
        self.home: np.ndarray | None = None
        self.robot_ready = False
        self.stopped = False
        self.arm_target = config.ARM_RESET.astype(np.float64)
        self.pitch_down = False  # the pitch mode; self.pitch turns toward it
        self.pitch = 0.0
        self.roll = 0.0
        self.last_pose: tuple[np.ndarray, float] | None = None

    # Robot state

    @property
    def gripper_pos(self) -> float:
        angle = self.ros.gripper_angle
        return 0.0 if angle is None else float(np.clip(angle / self.config.GRIPPER_OPEN_ANGLE, 0.0, 1.0))

    @property
    def gripper_open(self) -> bool:
        return self.gripper_pos > self.config.GRIPPER_OPEN_THRESHOLD

    def _hand(self) -> np.ndarray | None:
        pose = self.ros.lookup(self.config.BODY_FRAME, self.config.HAND_FRAME)
        return None if pose is None else pose[0]

    def _hand_rot(self) -> np.ndarray:
        """Hand x and y axes in the body frame, measured (commanded without TF)."""
        pose = self.ros.lookup(self.config.BODY_FRAME, self.config.HAND_FRAME)
        rot = self._hand_rotation() if pose is None else Rotation.from_quat(pose[1])
        return rot.as_matrix()[:, :2].T.flatten()

    def _hand_rotation(self) -> Rotation:
        """Commanded hand orientation in the body frame: pitch, then roll about the gripper's own axis."""
        return Rotation.from_euler("xyz", [self.roll, self.pitch, 0.0])

    def _pose(self) -> np.ndarray | None:
        return self.ros.pose2d(self.config.WORLD_FRAME, self.config.BODY_FRAME)

    def _base_pose(self) -> np.ndarray | None:
        """Body (x, y, yaw) in home's frame, None without home or TF."""
        pose = self._pose()
        return None if pose is None or self.home is None else relative(self.home, pose)

    # Commands

    def teleop_action(self) -> tuple[np.ndarray, bool]:
        """The gamepad's action and whether it is in use (a stick, trigger or bumper out of its deadzone, A or Y pressed)."""
        axes = self.ros.teleop()  # vx, vy, wz, arm_x, arm_z, roll
        pitch = gripper = 0.0
        if self.ros.pop_event("pitch_toggle"):
            pitch = -1.0 if self.pitch_down else 1.0
        if self.ros.pop_event("gripper_toggle"):
            gripper = -1.0 if self.gripper_open else 1.0
        action = np.r_[axes[:5], pitch, axes[5], gripper]
        return action, bool(np.any(action))

    def _apply(self, action: np.ndarray):
        """Send one action: a gripper change (stops the base and waits), else base velocity and hand target."""
        g = action[7]
        if (g <= -0.5 and self.gripper_open) or (g >= 0.5 and not self.gripper_open):
            self.ros.send_velocity(np.zeros(3))
            self.ros.call("close_gripper" if g < 0 else "open_gripper", self.config.GRIPPER_TIMEOUT)
            return
        self.ros.send_velocity(self._limit_base(action[:3] * self.config.BASE_SCALE))
        if abs(action[5]) >= 0.5:
            self.pitch_down = action[5] > 0
        self._turn_hand(self.config.HAND_PITCH_DOWN if self.pitch_down else 0.0, self.roll + action[6] * self.config.ROLL_STEP)
        target = self.arm_target.copy()
        target[[0, 2]] += action[3:5] * self.config.ARM_STEP
        self._send_hand(target)

    def _turn_hand(self, pitch: float, roll: float):
        """Move the commanded orientation toward (pitch, roll), at most PITCH_STEP and ROLL_STEP per call."""
        c = self.config
        self.pitch += np.clip(pitch - self.pitch, -c.PITCH_STEP, c.PITCH_STEP)
        self.roll += np.clip(np.clip(roll, -c.ROLL_LIMIT, c.ROLL_LIMIT) - self.roll, -c.ROLL_STEP, c.ROLL_STEP)

    def _send_hand(self, target: np.ndarray):
        hand = self._hand()
        if hand is not None:
            target = np.clip(target, hand - self.config.ARM_MAX_ERROR, hand + self.config.ARM_MAX_ERROR)
        target = np.clip(target, self.config.ARM_LIMIT_LOW, self.config.ARM_LIMIT_HIGH)
        floor = self.ros.lookup(self.config.BODY_FRAME, self.config.FLOOR_FRAME)
        if floor is not None:
            target[2] = max(target[2], floor[0][2] + self.config.ARM_FLOOR_CLEARANCE)
        self.arm_target = target
        self.ros.send_hand(self.arm_target, self._hand_rotation().as_quat())

    def _limit_base(self, v: np.ndarray) -> np.ndarray:
        """Drop the velocity components that take the base further out of BASE_LIMIT around home."""
        pose = self._base_pose()
        if pose is None:
            return v
        v_home = rotate(v[:2], pose[2])
        v_home[(np.abs(pose[:2]) > self.config.BASE_LIMIT) & (np.sign(v_home) == np.sign(pose[:2]))] = 0.0
        return np.r_[rotate(v_home, -pose[2]), v[2]]

    # Gym API

    def step(self, action: np.ndarray) -> tuple:
        start = time.monotonic()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply(action)
        self.curr_path_length += 1
        time.sleep(max(0.0, 1.0 / self.hz - (time.monotonic() - start)))
        obs = self._get_obs()
        self._check_set_home()
        failed = self.ros.pop_event("failure")
        self.stopped = self.ros.pop_event("stop")
        if self.stopped:
            self.ros.send_velocity(np.zeros(3))
        done = self.curr_path_length >= self.max_episode_length or failed or self.stopped
        info = {
            "succeed": False,
            "success_button": self.ros.button("success"),
            "save_requested": self.ros.pop_event("save"),
        }
        self._publish_status("running")
        return obs, 0, done, False, info

    def reset(self, **kwargs) -> tuple:
        self._end_episode()
        self.ros.send_velocity(np.zeros(3))
        self._wait_ready()
        while True:
            if self.stopped:
                self._publish_status("stopped", "Stowing the arm and sitting down")
                self.ros.call("arm_stow", self.config.SERVICE_TIMEOUT)
                self.ros.call("sit", self.config.SERVICE_TIMEOUT)
                self.robot_ready = False
                self._wait_for_start("stopped", "Stopped. Start stands Spot up.", teleop=False)
                self.stopped = False
            if not self.robot_ready:
                self._publish_status("resetting", "Standing up, unstowing the arm")
                self.robot_ready = self.ros.call("stand", self.config.SERVICE_TIMEOUT) and self.ros.call(
                    "arm_unstow", self.config.SERVICE_TIMEOUT
                )
                if not self.robot_ready:
                    raise RuntimeError("Spot did not stand up and unstow its arm: is the lease claimed, motors on?")
            if self._go_to_start() and (not self.config.WAIT_FOR_START or self._wait_for_start("waiting")):
                break
            self.stopped = True
        self.ros.clear_events("success", "failure", "gripper_toggle", "pitch_toggle")
        self.curr_path_length = 0
        self.episode += 1
        self.last_pose = None
        obs = self._get_obs()
        self._publish_status("running")
        return obs, {"succeed": False}

    def report(self, intervening: bool | None = None, classifier_prob: float | None = None, succeed: bool = False):
        """Per-step facts from the wrappers (gamepad, reward), for the status and episode statistics."""
        if intervening is not None:
            self.intervening = bool(intervening)
            self.intervention_steps += int(intervening)
        if classifier_prob is not None:
            self.classifier_prob = float(classifier_prob)
        self.succeeded = self.succeeded or bool(succeed)

    def close(self):
        if hasattr(self, "ros"):
            self._end_episode()
            self.ros.close()

    # Reset steps

    def _wait_ready(self):
        """Wait for TF, joint states and every camera."""
        last_log = 0.0
        while True:
            missing = [key for key in self.config.CAMERAS if self.ros.image(key)[0] is None]
            if self.ros.gripper_angle is None:
                missing.append("joint_states")
            if self._hand() is None or self._pose() is None:
                missing.append("TF")
            if not missing:
                return
            if time.monotonic() - last_log > 5.0:
                self.ros.log.info(f"Waiting for {', '.join(missing)} (is the spot_ros2 driver running?)")
                last_log = time.monotonic()
            self._publish_status("waiting", f"Waiting for {', '.join(missing)}")
            time.sleep(0.2)

    def _go_to_start(self) -> bool:
        """Open the gripper, hand to ARM_RESET, walk to home with a random offset; False if stopped on the way."""
        self._publish_status("resetting")
        if not self.gripper_open:
            self.ros.call("open_gripper", self.config.GRIPPER_TIMEOUT)
        if not self._move_hand(self.config.ARM_RESET):
            return False
        if self.home is None:
            self.home = self._pose()
            self.ros.log.info(f"Home set at x={self.home[0]:.2f} y={self.home[1]:.2f} yaw={self.home[2]:.2f}")
        xy, yaw = self.config.RANDOM_RESET_XY, self.config.RANDOM_RESET_YAW
        offset = np.random.uniform([-xy, -xy, -yaw], [xy, xy, yaw])
        return self._walk_to(compose(self.home, offset))

    def _move_hand(self, target: np.ndarray, timeout: float = 5.0) -> bool:
        """Bring the hand to `target` (body frame) pointing forward, unrolled, at ARM_STEP per tick; False if stopped."""
        self.pitch_down = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ros.pop_event("stop"):
                return False
            hand = self._hand()
            if hand is not None and np.linalg.norm(hand - target) < 0.02 and self.pitch == 0.0 and self.roll == 0.0:
                return True
            self._turn_hand(0.0, 0.0)
            step = target - self.arm_target
            self._send_hand(self.arm_target + step * min(1.0, self.config.ARM_STEP / max(np.linalg.norm(step), 1e-9)))
            self._publish_status("resetting")
            time.sleep(1.0 / self.hz)
        self.ros.log.warn(f"Hand did not reach {np.round(target, 3)} within {timeout:.0f} s")
        return True

    def _walk_to(self, goal: np.ndarray) -> bool:
        """Walk to `goal` (x, y, yaw in WORLD_FRAME), P control on cmd_vel; False if stopped."""
        deadline = time.monotonic() + self.config.RESET_TIMEOUT
        try:
            while time.monotonic() < deadline:
                if self.ros.pop_event("stop"):
                    return False
                pose = self._pose()
                if pose is None:  # TF dropped out: stand still until it is back
                    self.ros.send_velocity(np.zeros(3))
                    time.sleep(1.0 / self.hz)
                    continue
                err = relative(pose, goal)
                if np.hypot(err[0], err[1]) < self.config.RESET_TOLERANCE[0] and abs(err[2]) < self.config.RESET_TOLERANCE[1]:
                    return True
                scale = self.config.BASE_SCALE
                self.ros.send_velocity(np.clip(err * self.config.RESET_GAIN, -scale, scale))
                self._publish_status("resetting", "Walking back to the start pose")
                time.sleep(1.0 / self.hz)
            self.ros.log.warn(f"Start pose not reached within {self.config.RESET_TIMEOUT:.0f} s")
            return True
        finally:
            self.ros.send_velocity(np.zeros(3))

    def _wait_for_start(self, phase: str, message: str = "Place the object, then press Start", teleop: bool = True) -> bool:
        """Wait for Start, the gamepad driving Spot meanwhile (if `teleop`); False on stop."""
        self.ros.clear_events("start")
        while not self.ros.pop_event("start"):
            if self.ros.pop_event("stop"):
                return False
            self._check_set_home()
            if teleop:
                action, active = self.teleop_action()
                if active:
                    self._apply(action)
            self._publish_status(phase, message)
            time.sleep(1.0 / self.hz)
        return True

    def _check_set_home(self):
        if self.ros.pop_event("set_home") and (pose := self._pose()) is not None:
            self.home = pose
            self.ros.log.info(f"Home set at x={self.home[0]:.2f} y={self.home[1]:.2f} yaw={self.home[2]:.2f}")

    def _end_episode(self):
        if self.curr_path_length:
            self.history.append((self.succeeded, self.intervention_steps / self.curr_path_length))
        self.curr_path_length = 0
        self.intervention_steps = 0
        self.intervening = False
        self.succeeded = False
        self.classifier_prob = 0.0
        if self.save_video:
            self._save_video()

    # Observation and status

    def _get_obs(self) -> dict:
        size = self.config.IMAGE_SIZE
        images, crops = {}, {}
        for key, cam in self.config.CAMERAS.items():
            msg, age = self.ros.image(key)
            if age > self.config.IMAGE_TIMEOUT:
                self.ros.log.warn(f"{key} image is {age:.2f} s old", throttle_duration_sec=2.0)
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if cam["rotate"] is not None:
                img = cv2.rotate(img, cam["rotate"])
            crops[key] = self.config.IMAGE_CROP[key](img) if key in self.config.IMAGE_CROP else img
            images[key] = cv2.resize(crops[key], (size, size), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
        if self.save_video:
            self.recording_frames.append(crops)
        strip = np.concatenate(list(images.values()), axis=1)[..., ::-1]
        self.ros.publish_policy_images(cv2.imencode(".jpg", strip)[1].tobytes())

        now, pose = time.monotonic(), self._pose()
        vel = np.zeros(3)
        if pose is not None and self.last_pose is not None and now > self.last_pose[1]:
            vel = relative(self.last_pose[0], pose) / (now - self.last_pose[1])
        if pose is not None:
            self.last_pose = pose, now
        hand, base_pose = self._hand(), self._base_pose()
        state = {
            "hand_pose": self.arm_target.copy() if hand is None else hand,
            "hand_rot": self._hand_rot(),
            "gripper_pos": np.array([self.gripper_pos]),
            "base_vel": vel,
            "base_pose": np.zeros(3) if base_pose is None else base_pose,
        }
        return {"images": images, "state": state}

    def _publish_status(self, phase: str, message: str = ""):
        hand, base_pose = self._hand(), self._base_pose()
        recent = np.array(self.history) if self.history else None
        status = {
            "phase": phase,
            "message": message,
            "episode": self.episode,
            "step": self.curr_path_length,
            "max_steps": self.max_episode_length,
            "intervening": self.intervening,
            "intervention_rate": self.intervention_steps / max(self.curr_path_length, 1),
            "classifier_prob": float(self.classifier_prob),
            "succeed": self.succeeded,
            "recent_episodes": len(self.history),
            "recent_success_rate": None if recent is None else float(recent[:, 0].mean()),
            "recent_intervention_rate": None if recent is None else float(recent[:, 1].mean()),
            "gripper": self.gripper_pos,
            "hand": None if hand is None else hand.tolist(),
            "hand_target": self.arm_target.tolist(),
            "hand_rpy": [float(self.roll), float(self.pitch), 0.0],
            "hand_mode": "down" if self.pitch_down else "forward",
            "arm_low": self.config.ARM_LIMIT_LOW.tolist(),
            "arm_high": self.config.ARM_LIMIT_HIGH.tolist(),
            "home": None if self.home is None else self.home.tolist(),
            "base_pose": None if base_pose is None else base_pose.tolist(),
            "base_limit": self.config.BASE_LIMIT.tolist(),
            "world_frame": self.config.WORLD_FRAME,
            "body_frame": self.config.BODY_FRAME,
        }
        self.ros.publish_status(json.dumps(status))

    def _save_video(self):
        if not self.recording_frames:
            return
        os.makedirs("./videos", exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for key in self.recording_frames[0]:
            h, w = self.recording_frames[0][key].shape[:2]
            path = f"./videos/{key}_{stamp}.mp4"
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.hz, (w, h))
            for frames in self.recording_frames:
                writer.write(frames[key])
            writer.release()
            print(f"Saved video for camera {key} at {path}")
        self.recording_frames.clear()
