import os

import jax
import numpy as np

from experiments.config import DefaultTrainingConfig
from serl_launcher.networks.reward_classifier import load_classifier_func
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from spot_env.spot_env import DefaultSpotEnvConfig, SpotEnv
from spot_env.wrappers import GamepadIntervention, SpotGripperPenaltyWrapper, SpotRewardWrapper


class EnvConfig(DefaultSpotEnvConfig):
    """Pick a plushy off the floor and drop it in a bin 1-2 m away. Calibrate the crops and the arm box on the robot
    (viser shows the policy's crops and the live hand position)."""

    CAMERAS = {
        **DefaultSpotEnvConfig.CAMERAS,
        "eagle": {"topic": "/hil_serl/camera/eagle/compressed", "rotate": None},  # bird's-eye webcam (spot_hil_serl webcam)
    }
    IMAGE_CROP = {
        "hand": lambda img: img[:, 80:-80],  # 640x480 -> centre 480x480
        "front": lambda img: img[160:, :],  # rotated frontleft (480x640): the lower 480x480, the floor ahead
        "eagle": lambda img: img[:, 80:-80],  # 640x480 -> centre 480x480: frame the plushy's area and the bin
    }
    MAX_EPISODE_LENGTH = 200  # 20 s at 10 Hz
    ARM_RESET = np.array([0.75, 0.0, 0.0])
    ARM_LIMIT_LOW = np.array([0.55, 0.0, -0.55])  # z: a backstop below the floor; the floor limit below applies
    ARM_FLOOR_CLEARANCE = 0.03  # m: hand frame above the floor (spot/feet_center); check the jaw tips clear it
    ARM_LIMIT_HIGH = np.array([1.0, 0.0, 0.2])
    BASE_LIMIT = np.array([2.0, 2.0])
    RANDOM_RESET_XY = 0.2
    RANDOM_RESET_YAW = 0.3


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["hand", "front", "eagle"]
    classifier_keys = ["hand", "front", "eagle"]
    proprio_keys = ["hand_pose", "hand_rot", "gripper_pos", "base_vel"]
    classifier_threshold = 0.75
    checkpoint_period = 2000
    buffer_period = 1000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    max_traj_length = 200
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = SpotEnv(fake_env=fake_env, save_video=save_video, config=EnvConfig())
        if not fake_env:
            env = GamepadIntervention(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            checkpoint = os.path.abspath("classifier_ckpt/")
            classifier_func = None
            if os.path.isdir(checkpoint):
                classifier_func = load_classifier_func(
                    key=jax.random.PRNGKey(0),
                    sample=env.observation_space.sample(),
                    image_keys=self.classifier_keys,
                    checkpoint_path=checkpoint,
                )
            else:
                print(f"No reward classifier at {checkpoint}: success only from the gamepad's B button.")
            env = SpotRewardWrapper(env, classifier_func, threshold=self.classifier_threshold)
        env = SpotGripperPenaltyWrapper(env, penalty=-0.02)
        return env
