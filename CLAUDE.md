# CLAUDE.md

ROS 2 Humble workspace controlling Spot + Arm (Boston Dynamics). Python 3.10, rclpy.

## Output rules
- Comments and explanations: short, active voice, no filler, no restating code.
- Show only changed lines/diffs. No restating code, no summaries, no preamble.
- Minimal code: no boilerplate, no speculative abstractions, no unused params/imports.
- Edit existing files; create new ones only when required.
- Type hints on public functions. Comments only for non-obvious logic.
- One node per file. Config (hosts, names, speeds) via ROS params/YAML, never hardcoded.
- Ask before running anything that moves hardware.
- Do not add command-line arguments unless explicitly required.
- Do not perform visual tests unless explicitly asked otherwise. Prefer log-based tests.
- If sudo is needed, stop and ask the user to execute

## Commands
```
colcon build --symlink-install --packages-select <pkg>
source install/setup.bash
ros2 run <pkg> <node>
colcon test --packages-select <pkg>
```

## Spot Arm
- Driver: `spot_ros2` (bdaiinstitute), topics/services under `/<robot_name>/`.
- Sequence: `claim` → `power_on` → `stand` → `arm_unstow` → command → `arm_stow` → `sit` → `release`.
- Simple actions: Trigger services (`arm_stow`, `arm_unstow`, `open_gripper`, `close_gripper`).
- Custom motion: `robot_command` action, built with `bosdyn.client.robot_command.RobotCommandBuilder` + `bosdyn_msgs` conversion.
- Frames: `body`, `odom`, `vision`. Units: meters, radians.
- Quaternion order: bosdyn = (w,x,y,z), geometry_msgs = (x,y,z,w). Convert explicitly.

## Visualization (viser)
- `server = viser.ViserServer(port=8080)`; one server per process, run in its own node/thread.
- Use `server.scene.*` and `server.gui.*` (not the deprecated top-level `server.add_*`).
- Robots: `viser.extras.ViserUrdf(server, urdf_path)` → `.update_cfg(joints_rad)` from `/joint_states`.
- Frames/poses: `server.scene.add_frame(name, wxyz=, position=)`; viser quats = (w,x,y,z).
- Point clouds: `add_point_cloud(points=Nx3 float32, colors=Nx3 uint8)`; downsample first.
- Update existing handles (`handle.position = ...`) instead of re-adding nodes each tick.
- Throttle updates to ~30 Hz. Visualization is read-only: never send robot commands from GUI callbacks without explicit request.

## Safety
- No motion at import time; gate behind `main()`.
- Default to low speed / long duration. Always stop + release in `finally`.
- Use tf2 for transforms; no hand-rolled matrices when tf2 covers it.
