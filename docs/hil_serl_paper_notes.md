# HIL-SERL: paper notes

Luo, Xu, Wu, Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*, 2024
([pdf](https://hil-serl.github.io/static/hil-serl-paper.pdf), arXiv 2410.21845). These notes collect the numbers and
design choices needed to reproduce it, the implementation details of this repo that the paper leaves out, and how they
map onto our Spot setup ([spot_walkthrough.md](spot_walkthrough.md)).

## In one paragraph

Off-policy RL (RLPD, SAC-based) on real robots from pixels, bootstrapped with 20-30 teleoperated demonstrations and
sped up by **online human corrections**: whenever the policy misbehaves, the operator takes over with a SpaceMouse,
and those corrections go into both the demo and the RL buffer. Reward is a **binary success classifier** trained on
~200 positive / ~1000 negative images. Images go through a **frozen, ImageNet-pretrained ResNet-10**. Grippers get a
separate **DQN grasp critic** over {open, close, stay}. With this, 11 tasks (precise assembly, dual-arm, deformable,
dynamic) reach **100% success in 1-2.5 h** of real-world training (timing belt: 6 h), on average 2x the success rate
of HG-DAgger and 1.8x faster cycle times.

## Does it use depth?

**No: RGB only.** Every task's observation is 1-4 RGB crops (wrist and/or side cameras) resized to **128x128**, plus
proprioception (end-effector pose, twist, force/torque, gripper position). The encoder is an ImageNet-pretrained
ResNet-10, which takes 3-channel images. No depth, point clouds or monocular depth anywhere. The paper attributes spatial
generalization to wrist cameras (ego-centric views) and the relative proprioceptive frame, not to 3D input.

For Spot we therefore use the hand RGB camera, the frontleft body camera (grayscale, replicated to 3 channels) and a
bird's-eye USB webcam as the side camera.
UniDepthV2 is not needed. Depth could be added as an extra image key, but that is outside what the paper validates.

## System

| Component | What it does |
| --- | --- |
| Actor process | Runs the policy on the robot at **10 Hz**, applies human interventions, sends transitions to the learner |
| Learner process | Holds both replay buffers, samples **50/50 demo/RL** (RLPD symmetric sampling), updates, publishes params to the actor |
| Demo buffer | 20-30 offline demos + every intervention transition |
| RL buffer | Every transition (policy and intervention) |
| Reward | Binary classifier on images; episode ends at the first success |
| Low-level control | 1 kHz impedance controller fed 10 Hz setpoints, with reference limiting |

Actor and learner talk over the network ([agentlace](https://github.com/youliangtan/agentlace)): transitions go
from actor to learner, and params go back every `steps_per_update` learner steps.

### RL algorithm (RLPD)

- Critic: `L_Q = E[(Q(s,a) - (r + γ E_{a'~π}[Q_target(s',a')]))²]`; target network by Polyak averaging.
- Actor: `L_π = -E_s[E_{a~π}[Q(s,a)] + α H(π(·|s))]`, temperature α tuned automatically (SAC).
- Each batch is half demo data, half online data.

### Gripper: hybrid continuous/discrete

- Two MDPs share observations: continuous arm motion (SAC above) and a discrete gripper action
  {close, stay, open} (with two grippers, 3² = 9 combinations).
- Grasp critic trained with double-DQN: `L = E[(r + γ Q_θ'(s', argmax_a' Q_θ(s',a')) - Q_θ(s,a))²]`.
- At inference: continuous action from the actor, gripper action = argmax of the grasp critic, concatenated.
- A small **grasp penalty** (code: -0.02) on gripper actions that change the gripper state discourages chattering.

### Observation and action representation

- Proprioception is expressed relative to the end-effector frame at the episode start, `T_{b0,bt} = T_{b0}⁻¹ T_{bt}`.
  The policy then cares only about the relative pose to the target, so it survives the target moving mid-episode.
- Actions are 6D twists in the **current** end-effector frame, mapped to the base frame for the controller.
- The episode's start pose is randomized uniformly within a box (e.g. 2-8 cm in x/y, 1-10° in rz).

### Controller safety (the "reference limiting" trick)

Impedance law `F = k_p·e + k_d·ė + F_ff + F_cor` with `e = p - p_ref`. Instead of lowering gains, clip `|e| ≤ Δ`, so the
spring force is bounded by `k_p·Δ + 2 k_d·Δ·f`. The arm stays stiff and accurate but cannot push hard when the policy
asks for a far-away target in contact. Plus an absolute workspace box on the pose. Dynamic tasks (Jenga whipping,
pan flipping) command feedforward wrenches instead.

## Human-in-the-loop procedure

1. Pick cameras: wrist cameras for ego-centric views, side cameras if the wrist can't see the whole scene. Crop each
   to the task region, resize to 128x128.
2. Reward classifier data: teleoperate, label success frames (button held). **~200 positives, ~1000 negatives**
   (≈10 trajectories of ~10 s, ~5 min). Add targeted false-positive/false-negative examples as needed. Expect >95%
   accuracy.
3. Record **20-30 demonstrations** (successful only) into the demo buffer.
4. Train: actor + learner. Intervene when the policy heads somewhere unrecoverable or is stuck; let it explore
   otherwise. Interventions are frequent early on and taper to zero.
5. Reset: scripted motion or human reset each episode.

Advice from the paper:
- Give **specific, short corrections**. Do not repeatedly take over for long stretches all the way to success: that
  overestimates Q early in training and destabilizes it.
- Use interventions to make the policy practice recovering from rare mistakes. This is what moves it from ~97% to 100%.

## Hyperparameters (supplementary tables)

Common to all tasks: **10 Hz**, proprio encoder 64, policy (and grasp critic) MLP **256x256**, Adam **3e-4**,
random-crop augmentation, 20 demos (Jenga: 30), binary-classifier reward (Jenga: human label at episode end).

| Task | Obs (images) | Action | Classifier views (acc.) | Max steps | Reset | Randomization | RL transitions | γ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAM insertion | wrist ×2 | 6D twist | wrist ×2 (97%) | 100 | scripted | 4 cm xy, 6° rz | 32k | 0.97 |
| SSD assembly | wrist ×2, side | 6D twist | wrist ×2, side (95%) | 100 | scripted | 2 cm xy, 1° rz | 21k | 0.97 |
| USB grasp-insert | wrist ×2, side | 6D twist + gripper | side (96%) | 120 | scripted | 2 cm xy, 10° rz | 50k | 0.98 |
| Cable clipping | wrist ×2 | 6D twist + gripper | wrist ×2 (97%) | 120 | human | 4 cm xy, 10° rz | 28k | 0.98 |
| IKEA side panel | wrist, side ×2 | 12D twist | side ×2 (97%) | 100 | scripted | 8 cm xy, 1° rz | 31k / 36k | 0.98 |
| IKEA top panel | side ×3 | 12D twist | side ×3 (95%) | 150 | scripted | 3 cm xy | 18k | 0.97 |
| Car dashboard | wrist ×2, side | 12D twist + gripper | wrist ×2, side (98%) | 200 | human | 2 cm xy | 36k | 0.97 |
| Object handover | wrist ×2, side | 12D twist + gripper | side (99%) | 200 | human | none | 43k | 0.97 |
| Timing belt | wrist ×2, side ×2 | 12D twist | side ×2 (96%) | 200 | human | 2 cm xy | 108k | 0.97 |
| Jenga whipping | wrist, side | wrench Fx, Fz, τz | human label | 20 | human | none | 10k | 0.96 |
| Object flipping | wrist, side | wrench Fx, Fz, τy | wrist (97%) | 100 | scripted | none | 25k | 0.985 |

All proprio sets include `tcp_pose, tcp_vel`. Contact tasks add `tcp_f/t`, grasping tasks add `gripper_pos`, and the
dynamic tasks add joint `q, dq`. Jenga's learning rate decays to 3e-5 at 70% success.

Reward classifier: frozen pretrained ResNet-10 + 2-layer MLP, cross-entropy, Adam 3e-4, 100 iterations (paper).

## Results

| Task | Train time (h) | BC (HG-DAgger) success | HIL-SERL success | BC cycle (s) | HIL-SERL cycle (s) |
| --- | --- | --- | --- | --- | --- |
| RAM insertion | 1.5 | 29% | 100% | 8.3 | 4.8 |
| SSD assembly | 1 | 79% | 100% | 6.7 | 3.3 |
| USB grasp-insert | 2.5 | 26% | 100% | 13.4 | 6.7 |
| Cable clipping | 1.25 | 95% | 100% | 7.2 | 4.2 |
| IKEA side panel 1 / 2 | 2 / 1.75 | 77% / 79% | 100% | 6.5 / 5.0 | 2.7 / 2.4 |
| IKEA top panel | 1 | 35% | 100% | 8.9 | 2.4 |
| IKEA whole (10 trials) | – | 1/10 | 10/10 | – | – |
| Car dashboard | 2 | 41% | 100% | 20.3 | 8.8 |
| Object handover | 2.5 | 79% | 100% | 16.1 | 13.6 |
| Timing belt | 6 | 2% | 100% | 9.1 | 7.2 |
| Jenga whipping | 1.25 | 8% | 100% | – | – |
| Object flipping | 1 | 46% | 100% | 3.9 | 3.8 |
| **Average** | | **49.7%** | **100%** | **9.6** | **5.4** |

100 evaluation trials per task, training time on one RTX 4090 including resets. Ablations on RAM / dashboard / flipping:

| Method | RAM | Dashboard | Flipping | Avg |
| --- | --- | --- | --- | --- |
| Diffusion Policy (200 demos) | 27 | 18 | 56 | 34 |
| HG-DAgger | 29 | 41 | 46 | 39 |
| BC (200 demos) | 12 | 35 | 46 | 31 |
| IBRL | 75 | 0 | 95 | 57 |
| Residual RL | 0 | 0 | 97 | 32 |
| DAPG | 8 | 18 | 72 | 33 |
| HIL-SERL, no demos, no interventions | 0 | 0 | 0 | 0 |
| HIL-SERL, 200 demos, no interventions | 48 | 0 | 100 | 49 |
| **HIL-SERL** | **100** | **100** | **100** | **100** |

Takeaways: demos *and* online corrections are both necessary. Ten times more demos does not replace corrections. RL
beats imitation on cycle time because γ < 1 rewards finishing sooner.

## Analysis worth remembering

- **Funnels**: state visitation contracts into a funnel from the initial states to the goal. "Critical states" (high
  Q variance under ±0.2 action noise) lie inside it and also have high Q. DAgger's visitation is more diffuse.
- **Reactive vs. predictive**: precise tasks learn high-variance actions early that shrink near the goal (visual
  servoing). Dynamic tasks learn near-zero-variance open-loop motions.
- **Limitations**: unclear for much longer horizons (sample complexity grows with horizon and action space). Little
  randomization; generalization to unstructured scenes is untested.

## Implementation details of this repo (not in the paper)

- **Agents** (`serl_launcher/agents/continuous`): `SACAgent` (fixed gripper), `SACAgentHybridSingleArm` /
  `SACAgentHybridDualArm` (learned gripper). The last action dimension is the gripper, rounded to {-1, 0, 1} →
  grasp-critic class {close, stay, open}. Chosen via `TrainConfig.setup_mode`.
- **Encoders**: `resnet-pretrained` = frozen ResNet-10 (weights downloaded to `~/.serl/resnet10_params.pkl`) +
  spatial learned embeddings (8 blocks) + 256-d bottleneck, one per image key, shared by actor/critic/grasp critic.
  Proprio goes through a 64-d dense layer.
- **SAC settings** (`utils/launcher.py`): critic ensemble 2, `temperature_init` 1e-2, `backup_entropy` False, tanh-squashed
  Gaussian (log-std in [1e-5, 5]), MLPs 256x256 with tanh + LayerNorm, Polyak 0.005, Adam 3e-4 for all networks.
- **Training loop** (`examples/train_rlpd.py`): batch 256 (128 demo + 128 online); `cta_ratio` = 2 critic updates per
  actor update; params published every 50 learner steps; learning starts after 100 online transitions; checkpoints and
  buffer dumps every `checkpoint_period` / `buffer_period` steps; restarting with the same `checkpoint_path` resumes
  both. Augmentation: random crop with 4 px padding on every image key.
- **Reward classifier** (`examples/train_reward_classifier.py`): same frozen ResNet-10 per classifier key + Dense(256)
  + dropout 0.1 + LayerNorm, sigmoid BCE, Adam 1e-4, 150 balanced batches of 256. Data: `classifier_data/*success*.pkl`
  and `*failure*.pkl` in the experiment folder. Typical success threshold: sigmoid > 0.7-0.75, often also requiring
  the gripper state to agree.
- **Recording**: `record_success_fail.py` labels a frame positive while the success button is held, and stops at
  `successes_needed` positives (default 200). `record_demos.py` keeps only episodes the reward marks successful
  (default 20).
- **BC baseline** (`examples/train_bc.py`): 512x512x512 MLP, dropout 0.25, trained on non-zero-action demo transitions.
- **Interventions** (`SpacemouseIntervention`): any input above a tiny threshold replaces the policy action and is
  reported in `info["intervene_action"]`. The actor stores those transitions in both buffers.
- **Timeouts are terminal**: hitting `MAX_EPISODE_LENGTH` sets `done` (mask 0), as do classifier successes.

## Mapping to Spot pick-and-place

| Paper (Franka) | Spot (this repo) |
| --- | --- |
| Wrist + side RGB cameras, 128x128 crops | Hand RGB + frontleft fisheye (grayscale → 3 ch) + bird's-eye webcam, 128x128 crops |
| `tcp_pose`, `tcp_vel`, `tcp_f/t`, `gripper_pos` | `hand_pose`, `hand_rot` (body frame), `gripper_pos`, `base_vel` (+ optional `base_pose` from home) |
| 6D twist in EE frame + gripper | `[vx, vy, wz]` base + `[dx, dz]` hand in body frame + pitch mode (forward/down) + roll + gripper {close, stay, open} |
| Impedance controller, reference limiting | `arm_pose_commands` targets, kept within `ARM_MAX_ERROR` of the measured hand and inside a box |
| Workspace box | Arm box in the body frame, hand ≥ 3 cm above the floor, base box around home |
| SpaceMouse intervention | Xbox gamepad (any stick/trigger/bumper outside the deadzone, or A/Y) |
| Space bar labels success | B held labels success; B press = human success, X = failure |
| Scripted/human reset | Spot walks back to home (± random offset), human re-places the plushy, Start begins |

The body frame is already ego-centric for both base and hand, so the Franka `RelativeFrame` wrapper is not used. Spot's
own arm controller handles the low level. We bound how far its target can lead the measured hand, which plays the role
of the paper's reference limiting.
