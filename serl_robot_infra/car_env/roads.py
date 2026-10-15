"""Random curved roads for CarEnv: map `seed` is one road from a start to a red goal box on a square of grass."""

from dataclasses import dataclass

import cv2
import numpy as np

STEP = 5.0  # px between centerline points


@dataclass
class Road:
    start: np.ndarray  # (x, y px, yaw rad): the car's start pose, one car length into the road
    goal: np.ndarray  # (x, y, yaw): the goal box's centre and heading, one car length before the road's end
    mask: np.ndarray  # (WORLD_SIZE, WORLD_SIZE) bool, indexed [y, x]: True on the road
    image: np.ndarray  # (WORLD_SIZE, WORLD_SIZE, 3) uint8 RGB: grass, road, goal box
    center: np.ndarray  # (N, 2) px: the centerline from start to end, every STEP px


def rectangle(pose: np.ndarray, size: tuple[float, float]) -> np.ndarray:
    """Corners (4, 2) of a length x width rectangle centred on pose (x, y, yaw)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * np.array(size) / 2
    return pose[:2] + half @ np.array([[c, s], [-s, c]])


def make_road(seed: int, config) -> Road:
    """A road whose curvature is a smooth random profile (radius >= MIN_RADIUS), straight for STRAIGHT_ENDS at both
    ends, redrawn until it fits in the world and keeps ROAD_GAP of grass between its own parts."""
    rng = np.random.default_rng(seed)
    size, width = config.WORLD_SIZE, config.ROAD_WIDTH
    while True:
        length = rng.uniform(*config.ROAD_LENGTH)
        s = np.arange(0.0, length, STEP)
        waves = zip(rng.uniform(0.4, 1.0, 3), rng.uniform(250, 800, 3), rng.uniform(0, 2 * np.pi, 3))
        profile = np.clip(sum(a * np.sin(2 * np.pi * s / wavelength + phase) for a, wavelength, phase in waves), -1, 1)
        ends = np.clip(np.minimum(s, length - s) / config.STRAIGHT_ENDS - 1, 0, 1)  # 0 on the straight ends
        yaw = rng.uniform(-np.pi, np.pi) + np.cumsum(profile * ends / config.MIN_RADIUS) * STEP
        steps = np.c_[np.cos(yaw), np.sin(yaw)] * STEP
        shape = np.cumsum(steps, axis=0) - steps[0]
        slack = size - 2 * width - (shape.max(0) - shape.min(0))
        if np.all(slack >= 0) and _clear(shape, width + config.ROAD_GAP, 3 * width):
            break
    center = shape - shape.min(0) + width + rng.uniform(0, slack)
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    i, g = round(config.CAR_SIZE[0] / STEP), len(center) - 1 - round(config.CAR_SIZE[0] / STEP)
    goal = np.r_[center[g], yaw[g]]

    mask = np.zeros((size, size), np.uint8)
    cv2.polylines(mask, [np.round(center).astype(np.int32)], False, 1, thickness=width)
    image = np.empty((size, size, 3), np.uint8)
    image[:] = config.COLORS["grass"]
    image[mask > 0] = config.COLORS["road"]
    cv2.fillPoly(image, [np.round(rectangle(goal, config.GOAL_SIZE)).astype(np.int32)], config.COLORS["goal"])
    return Road(np.r_[center[i], yaw[i]], goal, mask > 0, image, center)


def _clear(center: np.ndarray, gap: float, arc: float) -> bool:
    """No two centerline points more than `arc` px apart along the road come closer than `gap` px."""
    i, j = np.triu_indices(len(center), k=int(arc / STEP))
    return bool(np.min(np.linalg.norm(center[i] - center[j], axis=1)) >= gap)
