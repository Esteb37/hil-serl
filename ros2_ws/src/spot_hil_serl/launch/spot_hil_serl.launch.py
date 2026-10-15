"""HIL-SERL on Spot, the ROS side: the gamepad (joy's game_controller_node + gamepad_teleop), the bird's-eye webcam,
the viser GUI (http://localhost:8080) and Spot's startup (claim, power on, stand; on Ctrl-C stow the arm, sit and release). The
spot_ros2 driver runs on Spot's NUC. SpotEnv itself runs inside the recording and training scripts.

mock:=true runs spot_mock instead of the robot: give it a ROS_DOMAIN_ID of its own (it refuses to start next to a
real driver). startup:=false leaves the robot's power and lease alone (e.g. claimed from elsewhere).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("spot_hil_serl"), "config", "params.yaml")
    mock = LaunchConfiguration("mock")
    startup = LaunchConfiguration("startup")
    py = "python3"  # the active env's (hilserl: viser), whatever shebang --symlink-install wrote
    return LaunchDescription([
        DeclareLaunchArgument("mock", default_value="false"),
        DeclareLaunchArgument("startup", default_value="true"),
        Node(package="joy", executable="game_controller_node", parameters=[params]),
        Node(package="spot_hil_serl", executable="gamepad_teleop", parameters=[params], output="screen", prefix=py),
        Node(package="spot_hil_serl", executable="viser_gui", parameters=[params], output="screen", prefix=py),
        Node(package="spot_hil_serl", executable="webcam", parameters=[params], output="screen", prefix=py),
        Node(package="spot_hil_serl", executable="spot_startup", namespace="spot", parameters=[params], output="screen", prefix=py,
             condition=IfCondition(PythonExpression(["'", mock, "' != 'true' and '", startup, "' == 'true'"])),
             sigterm_timeout="60"),  # its shutdown sequence (stow, sit, power off) outlasts launch's default 5 s
        Node(package="spot_hil_serl", executable="spot_mock", namespace="spot", parameters=[params], output="screen", prefix=py,
             condition=IfCondition(mock)),
    ])
