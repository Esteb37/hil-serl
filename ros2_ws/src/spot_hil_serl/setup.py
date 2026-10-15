from glob import glob
from setuptools import setup

package_name = "spot_hil_serl"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "gamepad_teleop = spot_hil_serl.gamepad_teleop:main",
            "gamepad_drive = spot_hil_serl.gamepad_drive:main",
            "viser_gui = spot_hil_serl.viser_gui:main",
            "spot_startup = spot_hil_serl.spot_startup:main",
            "spot_mock = spot_hil_serl.spot_mock:main",
            "webcam = spot_hil_serl.webcam:main",
        ]
    },
)
