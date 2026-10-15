"""Bring Spot up through the spot_ros2 driver's Trigger services (<ns>/<name>) at start, and back down on exit.

Copied from dso-demo (dso_viz/spot_startup.py). At start, the `startup` services in order (claim, power_on, stand;
SpotEnv unstows the arm itself); the first failure ends the sequence, since the rest would act on a robot in an unknown
state. On exit (SIGINT or SIGTERM, also mid-startup), if the first startup call succeeded, every `shutdown` service in
order (arm_stow, close_gripper, sit, power_off, release), failed or not, ignoring further signals until done. Each call
waits up to `timeout`. power_off waits until Spot has sat down (every knee folded past `sat_knee` and still, up to
`sit_timeout`) and is skipped if it has not, so the robot never drops from standing.
If the driver holds the body lease without this node having claimed it (e.g. claimed by hand, or `enabled` false), the
`unclaimed_shutdown` services (arm_stow, sit) run on exit instead: the robot sits before the stack exits.
Never takes control from someone else: the driver's claim force-takes the lease, so nothing is called at start if any
lease belongs to another client (e.g. the tablet) or the lease state is unknown, and nothing at exit if one does then
(a hijack must win).
Moves the robot: run online only. With `enabled` false it claims nothing at start.
"""

import signal
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import JointState
from spot_msgs.msg import LeaseArray
from std_srvs.srv import Trigger


class SpotStartup(Node):
    def __init__(self):
        super().__init__("spot_startup")
        self.enabled = self.declare_parameter("enabled", True).value
        self.timeout = float(self.declare_parameter("timeout", 20.0).value)
        self.startup = self.declare_parameter("startup", ["claim", "power_on", "stand"]).value
        self.shutdown = self.declare_parameter("shutdown", ["arm_stow", "close_gripper", "sit", "power_off", "release"]).value
        self.unclaimed_shutdown = self.declare_parameter("unclaimed_shutdown", ["arm_stow", "sit"]).value
        self.driver_client = self.declare_parameter("driver_client", "ros_spot").value
        self.sit_timeout = float(self.declare_parameter("sit_timeout", 10.0).value)
        self.sat_knee = float(self.declare_parameter("sat_knee", -2.5).value)  # rad: ~-1.5 standing, ~-2.75 sitting
        self.claimed = False

    def _owners(self) -> list[str] | None:
        """Owners of the held leases as "client (resource)", None without lease state."""
        ok, msg = wait_for_message(LeaseArray, self, "status/leases", time_to_wait=self.timeout)
        if not ok:
            return None
        return [f"{r.lease_owner.client_name} ({r.resource})" for r in msg.resources if r.lease_owner.client_name]

    def _foreign_owner(self, owners: list[str] | None) -> str | None:
        """Another client holding a lease, "unknown" without lease state, None if all are free or the driver's."""
        if owners is None:
            return "unknown"
        return next((o for o in owners if not o.startswith(self.driver_client)), None)

    def _sat(self) -> bool:
        """Wait up to `sit_timeout` for every knee folded past `sat_knee` and still."""
        deadline = time.monotonic() + self.sit_timeout
        while time.monotonic() < deadline:
            ok, msg = wait_for_message(JointState, self, "joint_states", time_to_wait=1.0)
            if not ok:
                continue
            knees = [i for i, name in enumerate(msg.name) if name.endswith("_knee")]
            still = len(msg.velocity) == len(msg.name) and all(abs(msg.velocity[i]) < 0.05 for i in knees)
            if knees and still and all(msg.position[i] < self.sat_knee for i in knees):
                return True
        return False

    def _call(self, name: str) -> bool:
        client = self.create_client(Trigger, name)
        if not client.wait_for_service(timeout_sec=self.timeout):
            self.get_logger().error(f"{self.get_namespace()}/{name}: no such service (is spot_ros2 running?)")
            return False
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout)
        res = future.result()
        if res is None or not res.success:
            self.get_logger().error(f"{self.get_namespace()}/{name} failed: {'timed out' if res is None else res.message}")
            return False
        self.get_logger().info(f"{self.get_namespace()}/{name}: {res.message}")
        return True

    def start(self):
        if not self.enabled:
            self.get_logger().info("Disabled: not claiming or moving Spot.")
            return
        if owner := self._foreign_owner(self._owners()):
            self.get_logger().error(f"Lease held by {owner}: not taking control of Spot.")
            return
        for i, name in enumerate(self.startup):
            if not self._call(name):
                return
            self.claimed = self.claimed or i == 0

    def stop(self):
        for sig in (signal.SIGINT, signal.SIGTERM):  # a second Ctrl-C must not leave it half sat down
            signal.signal(sig, signal.SIG_IGN)
        owners = self._owners()
        if owner := self._foreign_owner(owners):
            self.get_logger().error(f"Lease held by {owner}: leaving Spot alone.")
            return
        if not self.claimed and not owners:
            self.get_logger().info("Driver holds no lease: leaving Spot alone.")
            return
        for name in self.shutdown if self.claimed else self.unclaimed_shutdown:
            if name == "power_off" and not self._sat():
                self.get_logger().error(f"Spot did not sit down within {self.sit_timeout:.0f} s: not powering off.")
                continue
            self._call(name)


def main():
    # Keep the ROS context alive on SIGINT/SIGTERM: stop() still calls the driver's services.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):  # also when started with SIGINT ignored (a shell's & job)
        signal.signal(sig, signal.default_int_handler)
    node = SpotStartup()
    try:
        node.start()
        signal.pause()  # nothing to serve; rclpy.spin would block the KeyboardInterrupt in C
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        rclpy.shutdown()
