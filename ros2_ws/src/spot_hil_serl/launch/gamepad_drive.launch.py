"""Drive Spot with the gamepad, nothing else: joy's game_controller_node -> gamepad_teleop -> gamepad_drive, and
spot_startup (claim, power on, stand; on Ctrl-C stow the arm, sit and release). The spot_ros2 driver runs on Spot's NUC.
Don't run it next to spot_hil_serl.launch.py or a SpotEnv script.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("spot_hil_serl"), "config", "params.yaml")
    py = "python3"  # the active env's, whatever shebang --symlink-install wrote
    return LaunchDescription([
        Node(package="joy", executable="game_controller_node", parameters=[params], output="screen"),
        Node(package="spot_hil_serl", executable="gamepad_teleop", parameters=[params], output="screen", prefix=py),
        Node(package="spot_hil_serl", executable="gamepad_drive", namespace="spot", parameters=[params], output="screen",
             prefix=py),
        Node(package="spot_hil_serl", executable="spot_startup", namespace="spot", parameters=[params], output="screen",
             prefix=py, sigterm_timeout="60"),  # its shutdown sequence (stow, sit, power off) outlasts launch's default 5 s
    ])
