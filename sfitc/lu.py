#!/usr/bin/env python3
import csv
import os
import queue
import threading
import time
import itertools
from collections import deque
from datetime import datetime

import cupy as cp
import message_filters
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2 as pc2


# ----------------------------
# Minimal TF helpers (no external tf_transformations dependency)
# Quaternion is (x, y, z, w)
# ----------------------------
def quaternion_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(4, dtype=np.float32)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    m = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
            [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
            [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return m


def stamp_to_sec(stamp_msg) -> float:
    # builtin_interfaces/msg/Time
    return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9


# %% Faster PC creation from NP
def create_cloud_from_np(header, fields, np_array: np.ndarray) -> PointCloud2:
    """
    Fast version of create_cloud, using NumPy vectorized byte representation.
    Assumes np_array is (N, 3) float32 for (x, y, z).
    """
    data = np_array.astype(np.float32).tobytes()

    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = int(np_array.shape[0])
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 12  # 3 floats * 4 bytes
    cloud_msg.row_step = cloud_msg.point_step * int(np_array.shape[0])
    cloud_msg.is_dense = True
    cloud_msg.data = data
    return cloud_msg


class PointCloudTransformer(Node):
    def __init__(self):
        super().__init__("point_cloud_transformer")

        # ---- Params (ROS1 "~param" -> ROS2 node params) ----
        self.declare_parameter("pc_history_size", 10)
        self.declare_parameter("pc_topic", "/velodyne_points")
        self.declare_parameter("odom_topic", "/imu/odom")
        self.declare_parameter("publish_subtopics", False)
        self.declare_parameter("point_cloud_transformed", "/transformed_point_cloud")
        self.declare_parameter("point_cloud_cumulative", "/cumulative_point_cloud")
        self.declare_parameter("point_cloud_cumulative_origin", "/cumulative_origin_point_cloud")

        self.pc_history_size = int(self.get_parameter("pc_history_size").value)
        self.pc_topic = str(self.get_parameter("pc_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.publish_subtopics = bool(self.get_parameter("publish_subtopics").value)
        self.transformed_point_cloud = str(self.get_parameter("point_cloud_transformed").value)
        self.cumulative_point_cloud = str(self.get_parameter("point_cloud_cumulative").value)
        self.cumulative_origin_point_cloud = str(self.get_parameter("point_cloud_cumulative_origin").value)

        self.get_logger().info(
            f"Params: pc_history_size={self.pc_history_size}, pc_topic={self.pc_topic}, odom_topic={self.odom_topic}"
        )
        self.get_logger().info(f"Publish Subtopics: {self.publish_subtopics}")
        self.get_logger().info(f"Transformed Point Cloud: {self.transformed_point_cloud}")
        self.get_logger().info(f"Cumulative Point Cloud: {self.cumulative_point_cloud}")
        self.get_logger().info(f"Cumulative Origin Point Cloud: {self.cumulative_origin_point_cloud}")

        # ---- Metrics bookkeeping ----
        self.processing_times = deque(maxlen=100)
        self.input_timestamps = deque(maxlen=100)
        self.processed_timestamps = deque(maxlen=100)
        self.message_count = 0

        # ---- CSV logging ----
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = os.path.join(log_dir, f"pointcloud_metrics_{timestamp}.csv")
        self.csv_file = open(self.csv_file_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "ros_time",
                "pc_timestamp",
                "latency_ms",
                "processing_time_ms",
                "odom_callback_duration_ms",
                "pc_callback_duration_ms",
                "pc_to_points_duration_ms",
                "transform_points_duration_ms",
                "pc_create_duration_ms",
                "cumulative_points_duration_ms",
                "cumulative_points_create_cloud_duration_ms",
                "translate_points_duration_ms",
                "cumulative_origin_points_duration_ms",
                "cumulative_origin_points_create_cloud_duration_ms",
                "processing_rate_Hz",
                "input_rate_Hz",
                "throughput_ratio",
                "cumulative_points",
                "input_queue_size",
            ]
        )

        # Keep only last N frames to avoid unbounded growth
        self.cumulative_points = deque(maxlen=self.pc_history_size)
        self.cumulative_origin_points = deque(maxlen=self.pc_history_size)

        # Worker queue/thread
        self.msg_queue = queue.Queue(maxsize=100)
        self.shutdown_flag = threading.Event()
        self.worker_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker_thread.start()

        # ---- Publishers ----
        if self.publish_subtopics:
            self.point_cloud_pub = self.create_publisher(PointCloud2, self.transformed_point_cloud, 10)
            self.cumulative_cloud_pub = self.create_publisher(PointCloud2, self.cumulative_point_cloud, 10)
            self.odom_pub = self.create_publisher(Odometry, "/transformed_odom", 10)

        self.cumulative_origin_cloud_pub = self.create_publisher(
            PointCloud2, self.cumulative_origin_point_cloud, 10
        )

        # Publish xyz-only clouds (do NOT reuse velodyne fields if you only output xyz bytes)
        self.xyz_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # ---- message_filters Subscribers + Approx sync ----
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, self.pc_topic)
        self.odom_sub = message_filters.Subscriber(self, Odometry, self.odom_topic)

        # ROS2 message_filters ApproximateTimeSynchronizer signature differs from ROS1; no "reset" arg.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pc_sub, self.odom_sub], queue_size=10, slop=0.5, allow_headerless=False
        )
        self.ts.registerCallback(self.synced_callback)

    def destroy_node(self):
        # Ensure resources closed on shutdown
        try:
            self.get_logger().info("Shutting down: stopping worker and closing CSV.")
            self.shutdown_flag.set()
            if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
            if hasattr(self, "csv_file") and not self.csv_file.closed:
                self.csv_file.flush()
                self.csv_file.close()
        finally:
            super().destroy_node()

    @staticmethod
    def calculate_rate(timestamps):
        if len(timestamps) < 2:
            return 0.0
        duration = timestamps[-1] - timestamps[0]
        return (len(timestamps) / duration) if duration > 0 else 0.0

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def synced_callback(self, point_cloud_msg: PointCloud2, odom_msg: Odometry):
        self.get_logger().info(f"Received point cloud frame.")
        self.input_timestamps.append(self.now_sec())
        try:
            self.msg_queue.put_nowait((point_cloud_msg, odom_msg))
        except queue.Full:
            self.get_logger().warn("Processing queue full — dropping frame.")

    def processing_loop(self):
        while not self.shutdown_flag.is_set() and rclpy.ok():
            try:
                point_cloud_msg, odom_msg = self.msg_queue.get(timeout=0.1)

                start_time = time.time()

                pc_time = stamp_to_sec(point_cloud_msg.header.stamp)
                now = self.now_sec()
                latency_ms = (now - pc_time) * 1000.0

                self.processed_timestamps.append(pc_time)
                self.message_count += 1

                odom_callback_start_time = time.time()
                translation, rotation, transformed_odom_msg = self.odometry_callback(odom_msg)
                odom_callback_duration_ms = (time.time() - odom_callback_start_time) * 1000.0

                pc_callback_start_time = time.time()
                pc_callback_stats = self.point_cloud_callback(point_cloud_msg, translation, rotation)
                pc_callback_duration_ms = (time.time() - pc_callback_start_time) * 1000.0

                if self.publish_subtopics:
                    transformed_odom_msg.header.stamp = self.get_clock().now().to_msg()
                    self.odom_pub.publish(transformed_odom_msg)

                processing_duration_ms = (time.time() - start_time) * 1000.0
                self.processing_times.append(processing_duration_ms)
                cumulative_count = len(self.cumulative_points)

                processing_rate = self.calculate_rate(self.processed_timestamps)
                input_rate = self.calculate_rate(self.input_timestamps)
                throughput_ratio = (processing_rate / input_rate) if input_rate > 0 else 0.0

                self.csv_writer.writerow(
                    [
                        now,
                        pc_time,
                        latency_ms,
                        processing_duration_ms,
                        odom_callback_duration_ms,
                        pc_callback_duration_ms,
                        *(pc_callback_stats.values() if pc_callback_stats else [0.0] * 8),
                        processing_rate,
                        input_rate,
                        throughput_ratio,
                        cumulative_count,
                        self.msg_queue.qsize(),
                    ]
                )
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Worker exception: {e}")

    @staticmethod
    def odometry_callback(msg: Odometry):
        translation = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]

        # Your original code used identity rotation always
        rotation = [0.0, 0.0, 0.0, 1.0]  # (x, y, z, w)

        transformed_odom_msg = Odometry()
        transformed_odom_msg.header = msg.header
        transformed_odom_msg.child_frame_id = msg.child_frame_id

        transformed_odom_msg.pose.pose.position.x = translation[0]
        transformed_odom_msg.pose.pose.position.y = translation[1]
        transformed_odom_msg.pose.pose.position.z = translation[2]

        transformed_odom_msg.pose.pose.orientation.x = rotation[0]
        transformed_odom_msg.pose.pose.orientation.y = rotation[1]
        transformed_odom_msg.pose.pose.orientation.z = rotation[2]
        transformed_odom_msg.pose.pose.orientation.w = rotation[3]

        transformed_odom_msg.twist = msg.twist
        return translation, rotation, transformed_odom_msg

    def point_cloud_callback(self, point_cloud_msg: PointCloud2, translation, rotation):
        # (Keeping your structure; reduced per-frame log spam compared to ROS1 "loginfo" each callback)
        if translation is None or rotation is None:
            self.get_logger().warn("Odometry data not yet available, skipping point cloud transformation.")
            return None

        # Step 1: PointCloud2 -> xyz (CuPy)
        pc_to_points_start_time = time.time()
        # Robust for mixed datatypes/point_step: use read_points() and build Nx3 float32
        pts_iter = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        # Flatten tuples into a 1D float32 stream, then reshape to (N,3)
        flat = np.fromiter(itertools.chain.from_iterable(pts_iter), dtype=np.float32)
        xyz_np = flat.reshape((-1, 3))
        xyz_cp = cp.asarray(xyz_np, dtype=cp.float32)
        pc_to_points_duration_ms = (time.time() - pc_to_points_start_time) * 1000.0

        # Step 2: Transform points (CuPy)
        transform_points_start_time = time.time()
        ones = cp.ones((xyz_cp.shape[0], 1), dtype=cp.float32)
        homogeneous = cp.concatenate((xyz_cp, ones), axis=1)

        rot_mat_cp = cp.asarray(quaternion_matrix(rotation), dtype=cp.float32)
        transformed_cp = homogeneous @ rot_mat_cp.T
        transformed_xyz = transformed_cp[:, :3] + cp.asarray(translation, dtype=cp.float32)

        transformed_points = transformed_xyz.get()  # back to numpy
        transform_points_duration_ms = (time.time() - transform_points_start_time) * 1000.0

        # Step 4: Publish transformed cloud (optional)
        pc_create_start_time = time.time()
        if self.publish_subtopics:
            transformed_msg = create_cloud_from_np(point_cloud_msg.header, self.xyz_fields, transformed_points)
            transformed_msg.header.stamp = self.get_clock().now().to_msg()
            self.point_cloud_pub.publish(transformed_msg)
        pc_create_duration_ms = (time.time() - pc_create_start_time) * 1000.0

        # Step 5: Cumulative transformed cloud (optional)
        cumulative_points_start_time = time.time()
        self.cumulative_points.append(transformed_points)
        points_cumulative_transformed = np.vstack(self.cumulative_points) if len(self.cumulative_points) else transformed_points

        cumulative_points_create_cloud_start_time = time.time()
        if self.publish_subtopics:
            cumulative_msg = create_cloud_from_np(
                point_cloud_msg.header, self.xyz_fields, points_cumulative_transformed
            )
            cumulative_msg.header.stamp = self.get_clock().now().to_msg()
            self.cumulative_cloud_pub.publish(cumulative_msg)
        cumulative_points_create_cloud_duration_ms = (time.time() - cumulative_points_create_cloud_start_time) * 1000.0
        cumulative_points_duration_ms = (time.time() - cumulative_points_start_time) * 1000.0

        # Step 6: Translate points back to origin
        translate_points_start_time = time.time()
        translated_to_origin = transformed_points - np.array(translation, dtype=np.float32)
        translate_points_duration_ms = (time.time() - translate_points_start_time) * 1000.0

        # Step 7: Cumulative origin-aligned cloud (always published in your ROS1 code)
        cumulative_origin_points_start_time = time.time()
        self.cumulative_origin_points.append(translated_to_origin)
        points_cumulative_origin = (
            np.vstack(self.cumulative_origin_points) if len(self.cumulative_origin_points) else translated_to_origin
        )

        cum_origin_create_cloud_start_time = time.time()
        cumulative_origin_msg = create_cloud_from_np(point_cloud_msg.header, self.xyz_fields, points_cumulative_origin)
        cum_origin_points_create_cloud_duration_ms = (time.time() - cum_origin_create_cloud_start_time) * 1000.0

        cumulative_origin_msg.header.stamp = self.get_clock().now().to_msg()
        self.cumulative_origin_cloud_pub.publish(cumulative_origin_msg)
        cumulative_origin_points_duration_ms = (time.time() - cumulative_origin_points_start_time) * 1000.0

        return {
            "pc_to_points_duration_ms": pc_to_points_duration_ms,
            "transform_points_duration_ms": transform_points_duration_ms,
            "pc_create_duration_ms": pc_create_duration_ms,
            "cumulative_points_duration_ms": cumulative_points_duration_ms,
            "cumulative_points_create_cloud_duration_ms": cumulative_points_create_cloud_duration_ms,
            "translate_points_duration_ms": translate_points_duration_ms,
            "cumulative_origin_points_duration_ms": cumulative_origin_points_duration_ms,
            "cumulative_origin_points_create_cloud_duration_ms": cum_origin_points_create_cloud_duration_ms,
        }


def main():
    rclpy.init()
    node = PointCloudTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
