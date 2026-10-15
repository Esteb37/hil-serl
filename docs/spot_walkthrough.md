# HIL-SERL on Spot: pick-and-place walkthrough

Spot with arm learns to pick a plushy off the floor and drop it in a bin 1-2 m away, with HIL-SERL
([paper notes](hil_serl_paper_notes.md)). Inputs: the hand RGB camera, the frontleft body camera and a bird's-eye
("eagle") USB webcam on the workstation watching the scene (RGB only, no depth; see the notes). Outputs: base velocity and hand motion (continuous) and the gripper (discrete). An Xbox gamepad
replaces the SpaceMouse for demonstrations and takeovers.

## What runs where

| Where | What |
| --- | --- |
| Spot's NUC | `spot_ros2` driver (namespace `spot`, `ROS_DOMAIN_ID=22`) |
| Workstation, `ros2 launch spot_hil_serl spot_hil_serl.launch.py` | `game_controller_node` (joy) → `gamepad_teleop`, `webcam` (→ `/hil_serl/camera/eagle/compressed`), `viser_gui` (http://localhost:8080), `spot_startup` (claim, power on, stand; stow, sit, release on Ctrl-C) |
| Workstation, scripts in `examples/experiments/spot_pick_place/` | `SpotEnv` (gym env, `serl_robot_infra/spot_env`) inside the recording/training scripts, plus the JAX learner |

`SpotEnv` talks to the driver through plain topics and services: `cmd_vel`, `arm_pose_commands` (hand pose in the body
frame), `open_gripper`/`close_gripper`/`stand`/`sit`/`arm_unstow`/`arm_stow`, `joint_states`, TF and the compressed
cameras. With the gamepad and the GUI it uses `/hil_serl/teleop`, `/hil_serl/event`, `/hil_serl/status` and
`/hil_serl/policy_images`.

| | |
| --- | --- |
| Action (8, in [-1, 1]) | `vx, vy, wz` (× 0.4 m/s, 0.3 m/s, 0.6 rad/s), hand `dx, dz` in the body frame (× 3 cm per step), pitch mode (≥ 0.5 gripper down, ≤ -0.5 forward, else stay; turns 0.25 rad per step), roll about the gripper's axis (× 0.15 rad per step, ±90°), gripper (≤ -0.5 close, ≥ 0.5 open, else stay; last, for the grasp critic) |
| Observation | `hand`, `front`, `eagle`: 128x128 crops. State: `hand_pose` (body frame), `hand_rot` (hand x/y axes in the body frame), `gripper_pos`, `base_vel` (`base_pose` from home also available) |
| Rate, length | 10 Hz, 200 steps (20 s) |
| Reward | Reward classifier on `hand` + `front` + `eagle` (sigmoid > 0.75, gripper open), or B on the gamepad |
| Safety | Hand target inside `ARM_LIMIT_*`, ≥ 3 cm above the floor (`ARM_FLOOR_CLEARANCE`, from `spot/feet_center`, at the hand frame's origin) and ≤ 5 cm from the measured hand; base can't leave ±2 m around home; Back = stop |

Settings live in `examples/experiments/spot_pick_place/config.py` (task) and `serl_robot_infra/spot_env/spot_env.py`
(`DefaultSpotEnvConfig`: topics, frames, scales); node settings in `ros2_ws/src/spot_hil_serl/config/params.yaml`.

## Gamepad (Xbox, joy's `game_controller_node`)

| Input | Action |
| --- | --- |
| Left stick | walk forward/back, strafe left/right |
| Left / right trigger | turn left / right |
| Right stick | hand up/down; right/left = forward/back (Spot seen from its side) |
| Y | gripper pointing forward ↔ down (forward after every reset) |
| LB / RB | roll the gripper about its own axis (±90°) |
| A | toggle the gripper |
| B | held: label frames as success (classifier data); press: success, ends the episode |
| X | failure, ends the episode |
| Start | start the episode (after the reset) / stand up after a stop |
| Back | stop: ends the episode, stows the arm, Spot sits until Start |

Moving any stick, trigger or bumper out of its deadzone, or pressing A or Y, is a takeover: the gamepad's action replaces the
policy's and is stored as an intervention. While SpotEnv waits for Start, the gamepad drives Spot freely. The GUI's
**Controls** buttons send the same events (plus **Set home here** and **Save data**).

Check the mapping once with the pad plugged in: `ros2 topic echo /hil_serl/teleop` (axes `vx, vy, wz, arm_x, arm_z,
roll`) and `ros2 topic echo /hil_serl/event`. If an axis is mirrored, flip its sign in `axis_scales`, and swap
`roll_buttons` to reverse the roll (`params.yaml`). The buttons act on the robot only while a script runs SpotEnv.

## Setup (once)

```bash
conda activate hilserl                  # Python 3.10; ROS Humble sourced by ~/.bashrc
pip install -e serl_launcher && pip install -r serl_launcher/requirements.txt
pip install -e serl_robot_infra --no-deps   # spot_env (skips the Franka/RealSense/SpaceMouse deps)
pip install viser==1.1.1 yourdfpy
# JAX on the GPU: the code needs jax 0.4.35 (as the README pins); in this env that also needs these two pins
pip install "jax==0.4.35" "jaxlib==0.4.35" "orbax-checkpoint==0.11.5" "nvidia-cuda-nvcc-cu12==12.6.85"
python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)]
mkdir -p ~/.serl && cp examples/experiments/resnet10_params.pkl ~/.serl/   # else downloaded on first use

cd ros2_ws
# spot_msgs (battery and lease messages, copied from dso-demo): rosidl needs the system Python, not conda's
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c 'source /opt/ros/humble/setup.bash && colcon build --packages-select spot_msgs'
colcon build --packages-select spot_hil_serl
```

The webcam is `/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_...-video-index0` (`params.yaml`, `webcam.device`). Only
one program can hold it: close other viewers before launching. Mount it high so it sees the plushy's area and the bin
all the time; its crop is `IMAGE_CROP["eagle"]`. To train without it, drop `eagle` from `CAMERAS`, `image_keys` and
`classifier_keys` in the experiment config.

Spot's 3D model in viser needs `spot_description`, e.g. `source ~/esteban_ws/dso-demo/install/setup.bash` before
`ros2_ws/install/setup.bash`. Without it the GUI shows Spot's body and hand frames only.

Every terminal below:
```bash
conda activate hilserl && source ~/esteban_ws/hil-serl/ros2_ws/install/setup.bash   # ROS_DOMAIN_ID=22 from ~/.bashrc
cd ~/esteban_ws/hil-serl/examples/experiments/spot_pick_place
```

## Workflow

Keep the tablet's E-stop at hand for every step that moves the robot.

1. **Bring up** (driver running on the NUC, tablet released): `ros2 launch spot_hil_serl spot_hil_serl.launch.py`.
   Spot stands. Ctrl-C stows the arm, sits and releases.
2. **Calibrate** (first time): run `bash record_classifier_data.sh` and press Start. Drive the hand to the floor with the
   gamepad and read its position in the GUI's **Episode** folder. Set `ARM_LIMIT_LOW[2]` so the fingertips stay just
   above the floor (`ARM_FLOOR_CLEARANCE`, 3 cm at the hand frame: check the jaw tips with the gripper down),
   `ARM_LIMIT_HIGH`, and `ARM_RESET`. Adjust `IMAGE_CROP` (and the webcam's mount) until the GUI's
   **policy inputs** show the plushy and the bin. Home is where Spot stands at the first reset (or **Set home here**). Place the bin so the front
   camera sees it from there.
3. **Reward classifier data**: `bash record_classifier_data.sh`. Each episode starts with Start. Do the task with the
   gamepad and **hold B** while the plushy lies in the bin with the gripper open. Everything else is a negative, so also
   cover near-misses (plushy on the bin's rim, in the gripper above the bin, next to it). It stops at 200 positives; aim
   for 3-5x more negatives. Ctrl-C or **Save data** keeps what is there. Output: `classifier_data/`.
4. **Train the classifier**: `bash train_classifier.sh` → `classifier_ckpt/`. Check it during the next step: false
   positives need more negative data.
5. **Demos**: `bash record_demos.sh`. Do the task 20 times. The classifier (or B) ends a successful episode; X or the
   timeout drops a failed one. Output: `demo_data/`.
6. **(Optional) BC baseline**: `bash train_bc.sh`, evaluate with `bash train_bc.sh --eval_n_trajs=10`.
7. **HIL training**: `bash run_learner.sh` (add `--debug` to skip wandb), then `bash run_actor.sh` in another terminal.
   After each episode Spot walks back home (±20 cm, ±0.3 rad), opens the gripper and raises the hand. Put the plushy
   somewhere new and press Start. Take over briefly when the policy heads somewhere useless, and let it explore
   otherwise. Avoid long takeovers all the way to success (see the paper notes). The GUI shows the intervention and
   success rates over the last 20 episodes.
8. **Evaluate**: `bash run_actor.sh --eval_checkpoint_step=N --eval_n_trajs=20`.

Checkpoints and buffers go to `run/` (every 2000 learner steps / 1000 actor steps, or **Save data**). Restarting with
an existing `run/` resumes after an Enter prompt. Remove `run/` for a fresh start.

## Dry run without the robot

`spot_mock` stands in for the driver: a kinematic Spot, top-down camera renders with a red plushy and a blue bin, and
a gripper that picks the plushy when closed low over it. Run it on a ROS domain of its own (it refuses to start next to
a real driver):

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
ros2 launch spot_hil_serl spot_hil_serl.launch.py mock:=true
```

Then use the steps above in other terminals with the same two exports. The whole pipeline (classifier data → classifier
→ demos → learner + actor) was dry-run this way with a scripted gamepad, without viser.

## Things to check on the robot first

- **Arm and base together**: SpotEnv streams `arm_pose_commands` (arm-only commands) alongside `cmd_vel`
  (mobility-only). Spot keeps each subsystem's last command, so the hand should hold its body-frame target while
  walking. Verify at low speed before training.
- **Pitch and roll** turn at most 0.25 / 0.15 rad per step (about 2.5 / 1.5 rad/s); try Y and the bumpers with the
  hand high first.
- **frontleft** is mounted sideways. SpotEnv and the GUI rotate it 90° clockwise; check that the image is upright.
- **Obstacle avoidance** stays on with `cmd_vel`, so Spot may keep its body back from the bin. The arm reaches over it.
- **Wi-Fi latency**: a `cmd_vel` holds 0.125 s on the driver. The env logs a warning when camera frames are over 0.5 s
  old.
- **viser** was not run in the dry run (the workstation froze twice while it was up, cause unknown). Watch the
  machine the first time you open the GUI.
