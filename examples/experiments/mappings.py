import importlib


class _ConfigMapping(dict):
    """Experiment name -> its TrainConfig, importing only the experiment asked for: each pulls in its own robot's
    packages (franka_env, spot_env)."""

    def __getitem__(self, name):
        return importlib.import_module(f"experiments.{super().__getitem__(name)}.config").TrainConfig


CONFIG_MAPPING = _ConfigMapping(
    ram_insertion="ram_insertion",
    usb_pickup_insertion="usb_pickup_insertion",
    object_handover="object_handover",
    egg_flip="egg_flip",
    spot_pick_place="spot_pick_place",
    car_nav="car_nav",
)
