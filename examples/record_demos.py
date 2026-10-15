import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def save():
        os.makedirs("./demo_data", exist_ok=True)
        file_name = f"./demo_data/{FLAGS.exp_name}_demos_{uuid}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_count} demos to {file_name}")

    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)

            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                    save()
                trajectory = []
                returns = 0
                obs, info = env.reset()
            elif info.get("save_requested", False):
                save()
    finally:
        save()
        env.close()


if __name__ == "__main__":
    app.run(main)