"""Top-down car on random curved roads (PyBullet + pygame) for HIL-SERL: drive from the start to the red goal box and
park in it.

World: WORLD_SIZE² px of grass at PX_PER_M px per meter, one road per episode from the `maps` pool (roads.py). Frame: x
right, y down on screen, so a positive yaw turns clockwise (right).

Action (2,) in [-1, 1]: steer (+1: full right), throttle (> 0 drive; < 0 brake while rolling forward, reverse once
slower than REVERSE_SPEED).

Observation, all the policy gets: images {cone: IMAGE_SIZE² RGB of the visibility cone, the car at the bottom centre
heading up, black outside the cone}, state {gps: [distance to the goal / WORLD_SIZE, sin and cos of its bearing (+:
right)], velocity: body [vx, vy, yaw rate] (m/s, rad/s), acceleration: body [ax, ay] (m/s²)}.

The car is a PyBullet rigid body without gravity, pushed by planar tire forces: rear-wheel drive, brakes, drag, rolling
resistance, and each axle's sideways force cancelling its slip up to the friction limit (beyond it the car slides).
The reward oracle gives 1 and ends the episode once the car is parked in the box (`parked`). Off the road (OFF_ROAD of
the car), X on the gamepad and MAX_EPISODE_LENGTH end it with 0. The window shows the whole map, the cone, the policy's
image (top right), who drives (green POLICY, red GAMEPAD) and the GPS; the gamepad (SDL GameController layout) drives
through `teleop_action`. With `checkpoint_dir`, the first episode after each new checkpoint there runs on GIF_MAP and is
saved as gifs/checkpoint_<step>.gif.
"""

import os
import re
import time
from collections import deque
from typing import Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")  # the gamepad keeps working without window focus
os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")  # Ctrl-C and SIGTERM act as without SDL, not as a window close

import cv2
import gymnasium as gym
import numpy as np
import pybullet as p
import pygame
from pybullet_utils.bullet_client import BulletClient
from PIL import Image
from pygame._sdl2 import controller

from car_env.roads import make_road, rectangle

GRAVITY = 9.81
AXIS_MAX = 32767  # SDL GameController axis full scale


class DefaultCarEnvConfig:
    """CarEnv's settings. Geometry in px, physics in SI units."""

    WORLD_SIZE = 1000  # px, square
    PX_PER_M = 10.0
    ROAD_WIDTH = 70  # px
    ROAD_LENGTH = (800, 1400)  # px, uniform per map
    MIN_RADIUS = 100  # px: the tightest curve
    STRAIGHT_ENDS = 100  # px of straight road after the start and before the end
    ROAD_GAP = 40  # px of grass at least between two parts of the same road
    CAR_SIZE = (50, 20)  # px, length x width
    GOAL_SIZE = (50, 20)
    TRAIN_MAPS = range(25)  # map seeds for demos and training
    EVAL_MAPS = range(25, 30)  # held out for evaluation
    COLORS = {"grass": (70, 130, 60), "road": (90, 90, 90), "goal": (220, 30, 30), "car": (40, 90, 220)}
    DRIVER_COLORS = {"policy": (40, 200, 60), "gamepad": (255, 40, 40)}
    GIF_SCALE = 0.5
    GIF_MAP = 1  # map seed the checkpoint GIFs drive

    HZ = 10
    SUBSTEPS = 24  # PyBullet steps per control step (240 Hz)
    MAX_EPISODE_LENGTH = 400
    MASS = 1200.0  # kg
    WHEELBASE = 3.0  # m, centre of mass halfway
    MAX_STEER = 0.5  # rad at steer ±1
    MAX_SPEED = 20.0  # m/s at full throttle, where drag balances the drive force
    MAX_REVERSE_SPEED = 4.0  # m/s
    DRIVE_ACCEL = 10.0  # m/s² at full throttle from standstill
    BRAKE_ACCEL = 8.0  # m/s² at full brake
    ROLLING_ACCEL = 0.2  # m/s²
    REVERSE_SPEED = 0.3  # m/s: braking slower than this reverses
    FRICTION = 0.9  # tire-road
    SLIP_TIME = 0.05  # s: the tires cancel sideways slip this fast below the friction limit

    CONE_RANGE = 250  # px
    CONE_FOV = np.pi / 2  # rad, at most pi
    IMAGE_SIZE = 128

    SUCCESS_OVERLAP = 0.5  # fraction of the car inside the goal box
    STOP_SPEED = 0.3  # m/s
    ALIGN_TOLERANCE = np.deg2rad(40)  # rad between the car's and the box's heading
    OFF_ROAD = 0.5  # fraction of the car off the road that ends the episode
    RANDOM_RESET_LATERAL = 5  # px, uniform start offset across the road
    RANDOM_RESET_YAW = 0.1  # rad
    DEADZONE = 0.1  # gamepad sticks and triggers


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class CarEnv(gym.Env):
    def __init__(
        self,
        config: DefaultCarEnvConfig,
        maps: Sequence[int] = (),
        fake_env: bool = False,
        wait_for_start: bool = False,
        checkpoint_dir: str | None = None,
        headless: bool = False,
    ):
        """`maps`: the map seeds episodes sample from. `wait_for_start`: each episode waits for Start on the gamepad.
        `fake_env`: the spaces only (the learner). `checkpoint_dir`: the learner's checkpoints, to record a GIF after
        each new one. `headless`: no window, gamepad or real-time pacing; only GIF episodes are drawn."""
        self.config = config
        n = config.IMAGE_SIZE
        self.action_space = gym.spaces.Box(-np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "gps": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "velocity": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "acceleration": gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
                    }
                ),
                "images": gym.spaces.Dict({"cone": gym.spaces.Box(0, 255, shape=(n, n, 3), dtype=np.uint8)}),
            }
        )
        self.screen = None
        self.headless = headless
        if fake_env:
            return

        self.maps = list(maps)
        self.wait_for_start = wait_for_start
        self.rng = np.random.default_rng()
        self.roads = {}  # map seed -> (Road, its pygame surface)
        self.episode = 0
        self.history: deque[bool] = deque(maxlen=20)  # recent episodes' success
        self.outcome = ""
        self.pressed: set[int] = set()
        self.intervening = False
        self.intervention_steps = 0
        self.checkpoint_dir = checkpoint_dir
        self.recorded = self._latest_checkpoint()  # resuming: only checkpoints saved from now on
        self.frames = None  # this episode's GIF frames while recording

        length, width = config.CAR_SIZE
        gx, gy = np.meshgrid((np.arange(20) + 0.5) / 20 - 0.5, (np.arange(8) + 0.5) / 8 - 0.5)
        self.footprint = np.c_[gx.ravel() * length, gy.ravel() * width]  # px, car frame: area samples
        self.cone_side = max(config.CONE_RANGE, 2 * config.CONE_RANGE * np.sin(config.CONE_FOV / 2))  # px, square
        u, v = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
        ahead, right = (n - v) * self.cone_side / n, (u - n / 2) * self.cone_side / n
        self.cone_mask = (np.hypot(ahead, right) <= config.CONE_RANGE) & (
            np.abs(np.arctan2(right, ahead)) <= config.CONE_FOV / 2
        )

        self.sim = BulletClient(p.DIRECT)
        self.sim.setTimeStep(1.0 / (config.HZ * config.SUBSTEPS))
        half = [length / 2 / config.PX_PER_M, width / 2 / config.PX_PER_M, 0.75]
        self.car = self.sim.createMultiBody(config.MASS, self.sim.createCollisionShape(p.GEOM_BOX, halfExtents=half))
        self.sim.changeDynamics(self.car, -1, linearDamping=0.0, angularDamping=0.0)

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        if not headless:
            controller.init()
            if controller.get_count() == 0:
                raise RuntimeError("No gamepad found: connect one and restart.")
            self.pad = controller.Controller(0)
        # The map, then a side panel with the policy's view and the readouts: nothing covers the map
        self.screen = pygame.display.set_mode((config.WORLD_SIZE + 2 * n + 20, config.WORLD_SIZE))
        pygame.display.set_caption("HIL-SERL car")
        self.overlay = pygame.Surface((config.WORLD_SIZE, config.WORLD_SIZE), pygame.SRCALPHA)
        self.font = pygame.font.Font(None, 26)
        self.big_font = pygame.font.Font(None, 48)

    # Car state and physics

    def _state(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Pose (x, y px, yaw), world velocity (m/s) and yaw rate (rad/s)."""
        position, orientation = self.sim.getBasePositionAndOrientation(self.car)
        linear, angular = self.sim.getBaseVelocity(self.car)
        yaw = self.sim.getEulerFromQuaternion(orientation)[2]
        return np.r_[np.array(position[:2]) * self.config.PX_PER_M, yaw], np.array(linear[:2]), angular[2]

    def _drive(self, action: np.ndarray):
        """SUBSTEPS physics steps under the action's tire forces (car frame: x forward, y right)."""
        c = self.config
        steer = action[0] * c.MAX_STEER
        wheel_side = np.array([-np.sin(steer), np.cos(steer)])
        half, axle_mass = c.WHEELBASE / 2, c.MASS / 2
        grip = c.FRICTION * axle_mass * GRAVITY
        for _ in range(c.SUBSTEPS):
            pose, velocity, yaw_rate = self._state()
            vx, vy = velocity @ rotation(pose[2])
            front_slip = wheel_side @ [vx, vy + yaw_rate * half]
            front = -np.clip(axle_mass * front_slip / c.SLIP_TIME, -grip, grip) * wheel_side
            rear = -np.clip(axle_mass * (vy - yaw_rate * half) / c.SLIP_TIME, -grip, grip)
            force = rotation(pose[2]) @ (front + [self._longitudinal(action[1], vx), rear])
            position = [pose[0] / c.PX_PER_M, pose[1] / c.PX_PER_M, 0.0]
            self.sim.applyExternalForce(self.car, -1, [*force, 0.0], position, p.WORLD_FRAME)
            self.sim.applyExternalTorque(self.car, -1, [0.0, 0.0, half * (front[1] - rear)], p.WORLD_FRAME)
            self.sim.stepSimulation()

    def _longitudinal(self, throttle: float, vx: float) -> float:
        """Forward force (N): drive, brake or reverse, minus drag and rolling resistance."""
        c = self.config
        rolling = c.MASS * c.ROLLING_ACCEL
        drag = (c.MASS * c.DRIVE_ACCEL - rolling) / c.MAX_SPEED  # N per m/s: full throttle tops out at MAX_SPEED
        if throttle >= 0:
            force = throttle * c.MASS * c.DRIVE_ACCEL
        elif vx > c.REVERSE_SPEED:
            force = throttle * c.MASS * c.BRAKE_ACCEL
        else:
            force = throttle * (drag * c.MAX_REVERSE_SPEED + rolling)
        return force - drag * vx - np.sign(vx) * rolling

    # Perception and the reward oracle

    def _obs(self) -> dict:
        c = self.config
        pose, velocity, yaw_rate = self._state()
        to_goal = self.road.goal[:2] - pose[:2]
        bearing = np.arctan2(to_goal[1], to_goal[0]) - pose[2]
        acceleration = ((velocity - self.last_velocity) * c.HZ) @ rotation(pose[2])
        self.last_velocity = velocity
        self.cone_image = self._cone(pose)
        return {
            "state": {
                "gps": np.array([np.linalg.norm(to_goal) / c.WORLD_SIZE, np.sin(bearing), np.cos(bearing)], np.float32),
                "velocity": np.r_[velocity @ rotation(pose[2]), yaw_rate].astype(np.float32),
                "acceleration": acceleration.astype(np.float32),
            },
            "images": {"cone": self.cone_image},
        }

    def _cone(self, pose: np.ndarray) -> np.ndarray:
        """The map inside the visibility cone, the car at the bottom centre heading up, black elsewhere."""
        n = self.config.IMAGE_SIZE
        c, s = np.cos(pose[2]) * self.cone_side / n, np.sin(pose[2]) * self.cone_side / n
        # Image pixel (u, v) -> map: the car + (n - v) scaled steps ahead + (u - n/2) to the right
        warp = np.array([[-s, -c, pose[0] + n * c + n / 2 * s], [c, -s, pose[1] + n * s - n / 2 * c]])
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        image = cv2.warpAffine(self.road.image, warp, (n, n), flags=flags, borderValue=self.config.COLORS["grass"])
        image[~self.cone_mask] = 0
        return image

    def _car_points(self, pose: np.ndarray) -> np.ndarray:
        return pose[:2] + self.footprint @ rotation(pose[2]).T

    def _on_road(self, points: np.ndarray) -> float:
        """Fraction of the points on the road."""
        xy = np.round(points).astype(int)
        inside = np.all((xy >= 0) & (xy < self.config.WORLD_SIZE), axis=1)
        xy = np.clip(xy, 0, self.config.WORLD_SIZE - 1)
        return float(np.mean(inside & self.road.mask[xy[:, 1], xy[:, 0]]))

    def _in_goal(self, points: np.ndarray) -> float:
        """Fraction of the points inside the goal box."""
        local = (points - self.road.goal[:2]) @ rotation(self.road.goal[2])
        return float(np.mean(np.all(np.abs(local) <= np.array(self.config.GOAL_SIZE) / 2, axis=1)))

    def parked(self) -> bool:
        """The reward oracle: SUCCESS_OVERLAP of the car in the goal box, slower than STOP_SPEED, heading within
        ALIGN_TOLERANCE of the box's."""
        c = self.config
        pose, velocity, _ = self._state()
        return (
            self._in_goal(self._car_points(pose)) >= c.SUCCESS_OVERLAP
            and np.linalg.norm(velocity) < c.STOP_SPEED
            and abs(wrap(pose[2] - self.road.goal[2])) <= c.ALIGN_TOLERANCE
        )

    # Gamepad and window

    def _pump(self):
        """Window events: closing the window stops the script like Ctrl-C; gamepad presses wait for `_pop`."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.CONTROLLERBUTTONDOWN:
                self.pressed.add(event.button)

    def _pop(self, button: int) -> bool:
        pressed = button in self.pressed
        self.pressed.discard(button)
        return pressed

    def _axis(self, axis: int) -> float:
        """Stick in [-1, 1], trigger in [0, 1]: zero within DEADZONE and rescaled outside it to still reach ±1."""
        value = np.clip(self.pad.get_axis(axis) / AXIS_MAX, -1.0, 1.0)
        return float(np.sign(value) * max(0.0, abs(value) - self.config.DEADZONE) / (1 - self.config.DEADZONE))

    def teleop_action(self) -> tuple[np.ndarray, bool]:
        """The gamepad's action (left stick x steers, right trigger drives, left trigger brakes and reverses) and
        whether it is in use."""
        self._pump()
        steer = self._axis(pygame.CONTROLLER_AXIS_LEFTX)
        drive = self._axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
        brake = self._axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)
        return np.array([steer, drive - brake], np.float32), bool(steer or drive or brake)

    def report(self, intervening: bool):
        """Whether the gamepad drove this step (GamepadIntervention)."""
        self.intervening = bool(intervening)
        self.intervention_steps += int(intervening)

    def _render(self, message: str = ""):
        c = self.config
        pose, velocity, _ = self._state()
        self.screen.blit(self.map_surface, (0, 0))
        angles = pose[2] + np.linspace(-c.CONE_FOV / 2, c.CONE_FOV / 2, 16)
        cone = np.vstack([pose[:2], pose[:2] + c.CONE_RANGE * np.c_[np.cos(angles), np.sin(angles)]])
        self.overlay.fill((0, 0, 0, 0))
        pygame.draw.polygon(self.overlay, (255, 230, 0, 60), cone.tolist())
        self.screen.blit(self.overlay, (0, 0))
        driver = c.DRIVER_COLORS["gamepad" if self.intervening else "policy"]
        car = rectangle(pose, c.CAR_SIZE).tolist()
        pygame.draw.polygon(self.screen, c.COLORS["car"], car)
        pygame.draw.polygon(self.screen, driver, car, 4)
        front = pose[:2] + 0.35 * c.CAR_SIZE[0] * np.array([np.cos(pose[2]), np.sin(pose[2])])
        pygame.draw.circle(self.screen, (255, 255, 255), front.tolist(), 4)

        n, left = c.IMAGE_SIZE, c.WORLD_SIZE + 10
        self.screen.fill((0, 0, 0), (c.WORLD_SIZE, 0, 2 * n + 20, c.WORLD_SIZE))
        inset = pygame.transform.scale(pygame.image.frombuffer(self.cone_image.tobytes(), (n, n), "RGB"), (2 * n, 2 * n))
        self.screen.blit(inset, (left, 10))
        pygame.draw.rect(self.screen, (255, 255, 255), (left, 10, 2 * n, 2 * n), 1)
        banner = pygame.Rect(left, 2 * n + 20, 2 * n, 50)
        pygame.draw.rect(self.screen, driver, banner)
        label = self.big_font.render("GAMEPAD" if self.intervening else "POLICY", True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=banner.center))

        to_goal = self.road.goal[:2] - pose[:2]
        bearing = np.degrees(wrap(np.arctan2(to_goal[1], to_goal[0]) - pose[2]))
        lines = [
            f"map {self.map}  step {self.curr_path_length}/{c.MAX_EPISODE_LENGTH}",
            f"speed {np.linalg.norm(velocity):.1f} m/s",
            f"GPS {np.linalg.norm(to_goal) / c.PX_PER_M:.0f} m {bearing:+.0f} deg",
            f"success {sum(self.history)}/{len(self.history)} recent",
            self.outcome,
            message,
        ]
        for i, line in enumerate(filter(None, lines)):
            self.screen.blit(self.font.render(line, True, (255, 255, 255)), (left, 2 * n + 85 + 26 * i))
        pygame.display.flip()

    # Episodes

    def step(self, action: np.ndarray) -> tuple:
        c = self.config
        start = time.monotonic()
        self._pump()
        self._drive(np.clip(action, -1.0, 1.0))
        self.curr_path_length += 1
        obs = self._obs()
        pose, _, _ = self._state()
        succeed = self.parked()
        if succeed:
            outcome = "success"
        elif 1 - self._on_road(self._car_points(pose)) >= c.OFF_ROAD:
            outcome = "off road"
        elif self._pop(pygame.CONTROLLER_BUTTON_X):
            outcome = "aborted (X)"
        elif self.curr_path_length >= c.MAX_EPISODE_LENGTH:
            outcome = "timeout"
        else:
            outcome = None
        if outcome:
            self.history.append(succeed)
            self.outcome = f"last: {outcome}"
            print(
                f"Episode {self.episode} (map {self.map}): {outcome} after {self.curr_path_length} steps, "
                f"{self.intervention_steps} from the gamepad"
            )
        if self.frames is not None:
            self._render()
            self._record(done=outcome is not None)
        elif not self.headless:
            self._render()
            time.sleep(max(0.0, 1.0 / c.HZ - (time.monotonic() - start)))
        return obs, int(succeed), outcome is not None, False, {"succeed": succeed}

    def reset(self, **kwargs) -> tuple:
        c = self.config
        latest = self._latest_checkpoint()
        self.frames = [] if latest > self.recorded else None
        self.recorded = max(self.recorded, latest)
        self.map = c.GIF_MAP if self.frames is not None else int(self.rng.choice(self.maps))
        if self.map not in self.roads:
            road = make_road(self.map, c)
            surface = pygame.image.frombuffer(road.image.tobytes(), road.image.shape[1::-1], "RGB").convert()
            self.roads[self.map] = (road, surface)
        self.road, self.map_surface = self.roads[self.map]
        x, y, yaw = self.road.start
        offset = self.rng.uniform(-1, 1) * c.RANDOM_RESET_LATERAL
        position = [(x - np.sin(yaw) * offset) / c.PX_PER_M, (y + np.cos(yaw) * offset) / c.PX_PER_M, 0.0]
        yaw += self.rng.uniform(-1, 1) * c.RANDOM_RESET_YAW
        self.sim.resetBasePositionAndOrientation(self.car, position, self.sim.getQuaternionFromEuler([0, 0, yaw]))
        self.sim.resetBaseVelocity(self.car, [0, 0, 0], [0, 0, 0])
        self.last_velocity = np.zeros(2)
        self.curr_path_length = self.intervention_steps = 0
        self.intervening = False
        self.episode += 1
        obs = self._obs()
        self._pump()
        self.pressed.clear()
        while self.wait_for_start and not self._pop(pygame.CONTROLLER_BUTTON_START):
            self._render("Press Start to drive")
            time.sleep(1.0 / c.HZ)
            self._pump()
        if not self.headless:
            self._render()
        return obs, {"succeed": False}

    def _latest_checkpoint(self) -> int:
        """The step of the newest checkpoint in `checkpoint_dir`, -1 without one."""
        if not self.checkpoint_dir or not os.path.isdir(self.checkpoint_dir):
            return -1
        steps = [re.fullmatch(r"checkpoint_(\d+)", name) for name in os.listdir(self.checkpoint_dir)]
        return max((int(m[1]) for m in steps if m), default=-1)

    def _record(self, done: bool):
        """Grab the window into the GIF; at the episode's end, save it next to the checkpoints."""
        frame = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        size = (round(frame.shape[1] * self.config.GIF_SCALE), round(frame.shape[0] * self.config.GIF_SCALE))
        self.frames.append(Image.fromarray(cv2.resize(frame, size, interpolation=cv2.INTER_AREA)))
        if not done:
            return
        path = os.path.join(self.checkpoint_dir, "gifs", f"checkpoint_{self.recorded}.gif")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.frames[0].save(
            path, save_all=True, append_images=self.frames[1:], duration=round(1000 / self.config.HZ), loop=0
        )
        print(f"Saved {path}")
        self.frames = None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.sim.disconnect()
