"""SpotEnv wrappers: gamepad interventions, the success reward and the grasp penalty."""

import gymnasium as gym
import numpy as np


class GamepadIntervention(gym.ActionWrapper):
    """The gamepad's action replaces the policy's while it is in use (a stick, trigger or bumper out of its deadzone, A or Y).

    As SpacemouseIntervention: the executed human action is reported in info["intervene_action"].
    """

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        expert, active = self.env.unwrapped.teleop_action()
        self.env.unwrapped.report(intervening=active)
        return (expert, True) if active else (action, False)

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info


class SpotRewardWrapper(gym.Wrapper):
    """Sparse success reward, ending the episode: the operator's B press, or the reward classifier's sigmoid above
    `threshold` (with the gripper open if `require_gripper_open`: the object was let go). Without a classifier
    (`classifier_func` None, e.g. before one is trained) only B gives reward."""

    def __init__(self, env, classifier_func=None, threshold: float = 0.75, require_gripper_open: bool = True):
        super().__init__(env)
        self.classifier_func = classifier_func
        self.threshold = threshold
        self.require_gripper_open = require_gripper_open

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        spot = self.env.unwrapped
        prob = 0.0 if self.classifier_func is None else float(1 / (1 + np.exp(-np.asarray(self.classifier_func(obs)).item())))
        classified = prob > self.threshold and (spot.gripper_open or not self.require_gripper_open)
        success = spot.ros.pop_event("success") or classified
        spot.report(classifier_prob=prob, succeed=success)
        info["succeed"] = success
        info["classifier_prob"] = prob
        return obs, int(success), done or success, truncated, info


class SpotGripperPenaltyWrapper(gym.Wrapper):
    """info["grasp_penalty"] = `penalty` for gripper actions that change the gripper's state (open <-> closed), as the
    Franka experiments' GripperPenaltyWrapper; the hybrid agent's grasp critic adds it to the reward."""

    def __init__(self, env, penalty: float = -0.02):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        was_open = self.env.unwrapped.gripper_open
        obs, rew, done, truncated, info = self.env.step(action)
        g = info.get("intervene_action", action)[-1]
        info["grasp_penalty"] = self.penalty if (g <= -0.5 and was_open) or (g >= 0.5 and not was_open) else 0.0
        return obs, rew, done, truncated, info
