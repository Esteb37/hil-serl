"""USB webcam (the bird's-eye "eagle" view of the scene) -> sensor_msgs/CompressedImage on `topic`, at most `rate` Hz.

Opens `device` (a /dev/v4l/by-id path survives renumbering) with V4L2, MJPG at width x height. A grab thread keeps the
newest frame, so the published image is never queued behind stale ones. If the camera drops out, the thread reopens it
every second.
"""

import os
import re
import threading
import time

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class Webcam(Node):
    def __init__(self):
        super().__init__("webcam")
        self.device = self.declare_parameter("device", "/dev/video0").value
        self.size = (self.declare_parameter("width", 640).value, self.declare_parameter("height", 480).value)
        self.fps = float(self.declare_parameter("fps", 30.0).value)
        self.frame_id = self.declare_parameter("frame_id", "eagle_camera").value
        self.quality = int(self.declare_parameter("jpeg_quality", 90).value)
        rate = float(self.declare_parameter("rate", 15.0).value)
        self.pub = self.create_publisher(
            CompressedImage, self.declare_parameter("topic", "/hil_serl/camera/eagle/compressed").value,
            qos_profile_sensor_data,
        )
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._grab, daemon=True)
        self.thread.start()
        self.create_timer(1.0 / rate, self._publish)

    def _open(self) -> cv2.VideoCapture | None:
        # OpenCV's V4L2 backend opens by index only: /dev/v4l/by-id/... -> /dev/videoN -> N.
        match = re.fullmatch(r"/dev/video(\d+)", os.path.realpath(self.device))
        if match is None:
            return None
        cap = cv2.VideoCapture(int(match.group(1)), cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.get_logger().info(
            f"Opened {self.device}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
            f"at {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
        )
        return cap

    def _grab(self):
        cap = None
        while self.running:
            if cap is None and (cap := self._open()) is None:
                self.get_logger().warn(f"Cannot open {self.device}, retrying", throttle_duration_sec=10.0)
                time.sleep(1.0)
                continue
            ok, frame = cap.read()
            if not ok:
                self.get_logger().warn(f"{self.device}: no frame, reopening")
                cap.release()
                cap = None
                continue
            with self.lock:
                self.frame = frame
        if cap is not None:
            cap.release()

    def _publish(self):
        with self.lock:
            frame, self.frame = self.frame, None
        if frame is None:
            return
        msg = CompressedImage(format="jpeg")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        self.running = False
        self.thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = Webcam()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
