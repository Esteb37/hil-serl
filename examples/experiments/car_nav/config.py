import os

from absl import flags

from car_env.car_env import CarEnv, DefaultCarEnvConfig
from experiments.config import DefaultTrainingConfig
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from spot_env.wrappers import GamepadIntervention


class TrainConfig(DefaultTrainingConfig):
    """Drive a random curved road to the red box and park in it, seeing only the visibility cone, the GPS and the car's
    own velocity and acceleration. The env's oracle gives the reward: no classifier."""

    image_keys = ["cone"]
    proprio_keys = ["gps", "velocity", "acceleration"]
    checkpoint_period = 2000
    buffer_period = 1000
    cta_ratio = 2
    random_steps = 0
    discount = 0.97  # steeper decay -> faster parking pays off more
    max_traj_length = DefaultCarEnvConfig.MAX_EPISODE_LENGTH
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"  # plain SAC, no gripper action

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # Policy scripts (train_rlpd.py, train_bc.py) define eval_n_trajs: they start episodes right away, on the
        # held-out maps when evaluating. record_demos.py waits for Start before each episode.
        policy = "eval_n_trajs" in flags.FLAGS
        config = DefaultCarEnvConfig()
        headless = os.environ.get("HEADLESS") == "1"  # pure RL: no window, gamepad or real-time pacing
        evaluating = policy and flags.FLAGS.eval_n_trajs > 0
        maps = config.EVAL_MAPS if evaluating else config.TRAIN_MAPS
        # The train_rlpd.py actor records a GIF after each checkpoint the learner saves
        checkpoints = flags.FLAGS.checkpoint_path if policy and not evaluating and "checkpoint_path" in flags.FLAGS else None
        env = CarEnv(
            config,
            maps=maps,
            fake_env=fake_env,
            wait_for_start=not policy,
            checkpoint_dir=checkpoints and os.path.abspath(checkpoints),
            headless=headless,
        )
        if not fake_env and not headless:
            env = GamepadIntervention(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env
