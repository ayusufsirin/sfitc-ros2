#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 (rclpy) PointCloud timing-safe transformer with:
- ApproximateTimeSynchronizer (PC + Odom)
- bounded-latency processing (LATEST-only by default; optional FIFO)
- dedicated worker thread (callbacks stay tiny)
- fast PointCloud2 -> numpy via frombuffer (no list(read_points))
- optional CuPy transform
- cumulative + cumulative_origin with bounded history
- CSV metrics identical to your ROS1 header

Run:
  ros2 run <your_pkg> point_cloud_transformer --ros-args \
    -p pc_topic:=/velodyne_points \
    -p odom_topic:=/jackal_velocity_controller/odom \
    -p pc_history_size:=10 \
    -p drop_policy:=latest \
    -p use_gpu:=true \
    -p publish_subtopics:=true
"""

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import message_filters

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import tf_transformations as tft
except Exception:
    tft = None


# -------------------- Fast PC creation from NP --------------------
def create_cloud_from_np(header, fields, np_array: np.ndarray) -> PointCloud2:
    """
    Fast create_cloud: assumes np_array is (N,3) float32 (x,y,z).
    Uses msg.fields as provided (common Velodyne field layout).
    """
    np_array = np_array.astype(np.float32, copy=False)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(np_array.shape[0])
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 12  # 3 floats * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = np_array.tobytes()
    return msg


# -------------------- Fast PC2 -> XYZ numpy --------------------
def pc2_to_xyz_numpy_fast(msg: PointCloud2) -> np.ndarray:
    """
    Fast XYZ extraction from PointCloud2 without Python per-point loops.
    Supports common PointField float32 x/y/z.
    """
    field_map = {f.name: f.offset for f in msg.fields}
    off_x = field_map.get("x", None)
    off_y = field_map.get("y", None)
    off_z = field_map.get("z", None)
    if off_x is None or off_y is None or off_z is None:
        raise RuntimeError("PointCloud2 missing x/y/z fields")

    # Build a "view" dtype that jumps by point_step
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [np.float32, np.float32, np.float32],
        "offsets": [off_x, off_y, off_z],
        "itemsize": msg.point_step
    })

    n = int(msg.width * msg.height)
    arr = np.frombuffer(msg.data, dtype=dtype, count=n)

    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0] = arr["x"]
    xyz[:, 1] = arr["y"]
    xyz[:, 2] = arr["z"]

    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class PointCloudTransformer(Node):
    def __init__(self):
        super().__init__("point_cloud_transformer")

        # -------------------- Params --------------------
        self.declare_parameter("pc_history_size", 10)
        self.declare_parameter("pc_topic", "/velodyne_points")
        self.declare_parameter("odom_topic", "/jackal_velocity_controller/odom")

        self.declare_parameter("publish_subtopics", False)
        self.declare_parameter("point_cloud_transformed", "/transformed_point_cloud")
        self.declare_parameter("point_cloud_cumulative", "/cumulative_point_cloud")
        self.declare_parameter("point_cloud_cumulative_origin", "/cumulative_origin_point_cloud")

        # timing safety / backpressure
        self.declare_parameter("drop_policy", "latest")   # "latest" (bounded latency) or "fifo"
        self.declare_parameter("max_queue", 100)          # used only for fifo mode
        self.declare_parameter("use_gpu", True)
        self.declare_parameter("slop_sec", 0.1)
        self.declare_parameter("sync_queue_size", 10)

        # Optional: publish cumulative clouds less often (keeps timing stable)
        self.declare_parameter("cumulative_publish_hz", 0.0)  # 0 = publish every processed frame

        self.pc_history_size = int(self.get_parameter("pc_history_size").value)
        self.pc_topic = str(self.get_parameter("pc_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.publish_subtopics = bool(self.get_parameter("publish_subtopics").value)
        self.transformed_point_cloud = str(self.get_parameter("point_cloud_transformed").value)
        self.cumulative_point_cloud = str(self.get_parameter("point_cloud_cumulative").value)
        self.cumulative_origin_point_cloud = str(self.get_parameter("point_cloud_cumulative_origin_point_cloud").value) \
            if self.has_parameter("point_cloud_cumulative_origin_point_cloud") else str(
                self.get_parameter("point_cloud_cumulative_origin").value
            )

        self.drop_policy = str(self.get_parameter("drop_policy").value).lower().strip()
        self.max_queue = int(self.get_parameter("max_queue").value)
        self.use_gpu = bool(self.get_parameter("use_gpu").value) and (cp is not None)
        self.slop_sec = float(self.get_parameter("slop_sec").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.cumulative_publish_hz = float(self.get_parameter("cumulative_publish_hz").value)

        if tft is None:
            raise RuntimeError(
                "tf_transformations not available. Install it (ROS2) or vendor it. "
                "Debian: ros-$ROS_DISTRO-tf-transformations"
            )

        self.get_logger().info(
            f"Params: pc_history_size={self.pc_history_size}, pc_topic={self.pc_topic}, odom_topic={self.odom_topic}, "
            f"publish_subtopics={self.publish_subtopics}, drop_policy={self.drop_policy}, use_gpu={self.use_gpu}"
        )

        # -------------------- Metrics buffers --------------------
        self.processing_times = deque(maxlen=100)
        self.input_timestamps = deque(maxlen=100)
        self.processed_timestamps = deque(maxlen=100)

        # -------------------- CSV log --------------------
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = os.path.join(log_dir, f"pointcloud_metrics_{ts}.csv")
        self.csv_file = open(self.csv_file_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
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
            "input_queue_size"
        ])

        # -------------------- Cumulative windows --------------------
        self.cumulative_points = deque(maxlen=self.pc_history_size)
        self.cumulative_origin_points = deque(maxlen=self.pc_history_size)

        # -------------------- Threading / queue --------------------
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

        # LATEST mode: store only newest synced pair
        self._latest_pair: Optional[Tuple[PointCloud2, Odometry]] = None
        self._latest_counter = 0
        self._processed_counter = 0
        self._wake = threading.Event()

        # FIFO mode: explicit queue
        self._fifo = deque(maxlen=self.max_queue)

        # cumulative publish throttle
        self._last_cum_pub_t = 0.0

        # -------------------- QoS --------------------
        pc_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        odom_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Callback group ensures sync callback is quick and non-overlapping with itself
        self.cb_group = MutuallyExclusiveCallbackGroup()

        # -------------------- message_filters (ROS2 style) --------------------
        pc_sub = message_filters.Subscriber(self, PointCloud2, self.pc_topic, qos_profile=pc_qos)
        odom_sub = message_filters.Subscriber(self, Odometry, self.odom_topic, qos_profile=odom_qos)

        tsync = message_filters.ApproximateTimeSynchronizer(
            [pc_sub, odom_sub],
            queue_size=self.sync_queue_size,
            slop=self.slop_sec
        )
        tsync.registerCallback(self.synced_callback)

        # -------------------- Publishers --------------------
        pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        if self.publish_subtopics:
            self.point_cloud_pub = self.create_publisher(PointCloud2, self.transformed_point_cloud, pub_qos)
            self.cumulative_cloud_pub = self.create_publisher(PointCloud2, self.cumulative_point_cloud, pub_qos)
            self.odom_pub = self.create_publisher(Odometry, "/transformed_odom", 10)

        self.cumulative_origin_cloud_pub = self.create_publisher(PointCloud2, self.cumulative_origin_point_cloud, pub_qos)

        # -------------------- Worker thread --------------------
        self.worker = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker.start()

    def calculate_rate(self, timestamps: deque) -> float:
        if len(timestamps) < 2:
            return 0.0
        duration = timestamps[-1] - timestamps[0]
        return float(len(timestamps)) / duration if duration > 0 else 0.0

    # -------------------- Sync callback: MUST be tiny --------------------
    def synced_callback(self, pc_msg: PointCloud2, odom_msg: Odometry):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.input_timestamps.append(now_sec)

        with self._lock:
            if self.drop_policy == "fifo":
                # drop-oldest when full
                if len(self._fifo) == self._fifo.maxlen:
                    self._fifo.popleft()
                self._fifo.append((pc_msg, odom_msg))
            else:
                # default: latest-only (perfect timing: bounded latency)
                self._latest_pair = (pc_msg, odom_msg)
                self._latest_counter += 1

        self._wake.set()

    # -------------------- Worker loop --------------------
    def processing_loop(self):
        while rclpy.ok() and not self._shutdown.is_set():
            # Wait until new work arrives
            self._wake.wait(timeout=0.1)
            self._wake.clear()

            # Drain work (latest) or pop one (fifo)
            pair = None
            qsize = 0

            with self._lock:
                if self.drop_policy == "fifo":
                    if len(self._fifo) > 0:
                        pair = self._fifo.popleft()
                    qsize = len(self._fifo)
                else:
                    pair = self._latest_pair
                    self._latest_pair = None
                    # approximate "queue size": how many arrivals since last processed
                    qsize = max(0, self._latest_counter - self._processed_counter - 1)

            if pair is None:
                continue

            pc_msg, odom_msg = pair

            t0 = time.perf_counter()

            # Latency
            pc_time = stamp_to_sec(pc_msg.header.stamp)
            now = self.get_clock().now().nanoseconds * 1e-9
            latency_ms = (now - pc_time) * 1000.0

            self.processed_timestamps.append(pc_time)

            # Odom transform extraction
            t_odom0 = time.perf_counter()
            translation, rotation, transformed_odom_msg = self.odometry_callback(odom_msg)
            odom_cb_ms = (time.perf_counter() - t_odom0) * 1000.0

            # PC processing
            t_pc0 = time.perf_counter()
            stats = self.point_cloud_callback(pc_msg, translation, rotation, qsize)
            pc_cb_ms = (time.perf_counter() - t_pc0) * 1000.0

            if self.publish_subtopics:
                # keep stamps consistent (use PC stamp)
                transformed_odom_msg.header.stamp = pc_msg.header.stamp
                self.odom_pub.publish(transformed_odom_msg)

            proc_ms = (time.perf_counter() - t0) * 1000.0
            self.processing_times.append(proc_ms)

            processing_rate = self.calculate_rate(self.processed_timestamps)
            input_rate = self.calculate_rate(self.input_timestamps)
            throughput_ratio = (processing_rate / input_rate) if input_rate > 0 else 0.0

            cumulative_count = len(self.cumulative_points)

            # Write CSV
            self.csv_writer.writerow([
                now,
                pc_time,
                latency_ms,
                proc_ms,
                odom_cb_ms,
                pc_cb_ms,
                stats["pc_to_points_duration_ms"],
                stats["transform_points_duration_ms"],
                stats["pc_create_duration_ms"],
                stats["cumulative_points_duration_ms"],
                stats["cumulative_points_create_cloud_duration_ms"],
                stats["translate_points_duration_ms"],
                stats["cumulative_origin_points_duration_ms"],
                stats["cumulative_origin_points_create_cloud_duration_ms"],
                processing_rate,
                input_rate,
                throughput_ratio,
                cumulative_count,
                qsize
            ])
            # flush occasionally to see live
            if (len(self.processing_times) % 20) == 0:
                self.csv_file.flush()

            with self._lock:
                self._processed_counter += 1

    # -------------------- Odom callback (extract pose) --------------------
    @staticmethod
    def odometry_callback(msg: Odometry):
        translation = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ], dtype=np.float32)

        # Use real odom orientation (ROS2)
        q = msg.pose.pose.orientation
        rotation = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

        transformed_odom_msg = Odometry()
        transformed_odom_msg.header = msg.header
        transformed_odom_msg.child_frame_id = msg.child_frame_id
        transformed_odom_msg.pose.pose.position.x = float(translation[0])
        transformed_odom_msg.pose.pose.position.y = float(translation[1])
        transformed_odom_msg.pose.pose.position.z = float(translation[2])
        transformed_odom_msg.pose.pose.orientation.x = float(rotation[0])
        transformed_odom_msg.pose.pose.orientation.y = float(rotation[1])
        transformed_odom_msg.pose.pose.orientation.z = float(rotation[2])
        transformed_odom_msg.pose.pose.orientation.w = float(rotation[3])
        transformed_odom_msg.twist = msg.twist
        return translation, rotation, transformed_odom_msg

    # -------------------- Main PC processing --------------------
    def point_cloud_callback(self, pc_msg: PointCloud2, translation: np.ndarray, rotation: np.ndarray, qsize: int):
        # Step 1: PC -> points (FAST)
        t1 = time.perf_counter()
        xyz_np = pc2_to_xyz_numpy_fast(pc_msg)  # (N,3) float32
        pc_to_points_ms = (time.perf_counter() - t1) * 1000.0

        # Step 2: transform (GPU or CPU), 3x3
        t2 = time.perf_counter()
        R = tft.quaternion_matrix(rotation)[:3, :3].astype(np.float32, copy=False)
        t = translation.astype(np.float32, copy=False)

        if self.use_gpu:
            xyz_cp = cp.asarray(xyz_np)
            R_cp = cp.asarray(R)
            t_cp = cp.asarray(t)
            transformed_xyz = xyz_cp @ R_cp.T + t_cp
            transformed_points = transformed_xyz.get()
        else:
            transformed_points = (xyz_np @ R.T) + t

        transform_ms = (time.perf_counter() - t2) * 1000.0

        # Step 3: create + publish transformed
        t3 = time.perf_counter()
        pc_create_ms = 0.0
        if self.publish_subtopics:
            transformed_msg = create_cloud_from_np(pc_msg.header, pc_msg.fields, transformed_points)
            transformed_msg.header.stamp = pc_msg.header.stamp
            self.point_cloud_pub.publish(transformed_msg)
        pc_create_ms = (time.perf_counter() - t3) * 1000.0

        # Optional throttle for cumulative publishing (keeps timing perfect under load)
        do_cum = True
        if self.cumulative_publish_hz > 0.0:
            now = self.get_clock().now().nanoseconds * 1e-9
            period = 1.0 / self.cumulative_publish_hz
            do_cum = (now - self._last_cum_pub_t) >= period
            if do_cum:
                self._last_cum_pub_t = now

        # Step 4: cumulative transformed (bounded history)
        t4 = time.perf_counter()
        cum_cloud_ms = 0.0
        cum_create_ms = 0.0
        if do_cum:
            self.cumulative_points.append(transformed_points)
            points_cum = np.vstack(self.cumulative_points) if len(self.cumulative_points) > 0 else transformed_points

            t4b = time.perf_counter()
            if self.publish_subtopics:
                cum_msg = create_cloud_from_np(pc_msg.header, pc_msg.fields, points_cum)
                cum_msg.header.stamp = pc_msg.header.stamp
                self.cumulative_cloud_pub.publish(cum_msg)
            cum_create_ms = (time.perf_counter() - t4b) * 1000.0
        cum_cloud_ms = (time.perf_counter() - t4) * 1000.0

        # Step 5: translate to origin
        t5 = time.perf_counter()
        translated_to_origin = transformed_points - t
        translate_ms = (time.perf_counter() - t5) * 1000.0

        # Step 6: cumulative origin aligned (bounded history)
        t6 = time.perf_counter()
        cum_origin_ms = 0.0
        cum_origin_create_ms = 0.0
        if do_cum:
            self.cumulative_origin_points.append(translated_to_origin)
            points_cum_origin = np.vstack(self.cumulative_origin_points) if len(self.cumulative_origin_points) > 0 else translated_to_origin

            t6b = time.perf_counter()
            cum_origin_msg = create_cloud_from_np(pc_msg.header, pc_msg.fields, points_cum_origin)
            cum_origin_msg.header.stamp = pc_msg.header.stamp
            self.cumulative_origin_cloud_pub.publish(cum_origin_msg)
            cum_origin_create_ms = (time.perf_counter() - t6b) * 1000.0
        cum_origin_ms = (time.perf_counter() - t6) * 1000.0

        return {
            "pc_to_points_duration_ms": pc_to_points_ms,
            "transform_points_duration_ms": transform_ms,
            "pc_create_duration_ms": pc_create_ms,
            "cumulative_points_duration_ms": cum_cloud_ms,
            "cumulative_points_create_cloud_duration_ms": cum_create_ms,
            "translate_points_duration_ms": translate_ms,
            "cumulative_origin_points_duration_ms": cum_origin_ms,
            "cumulative_origin_points_create_cloud_duration_ms": cum_origin_create_ms,
        }

    def close(self):
        self._shutdown.set()
        self._wake.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        self.get_logger().info(f"Closed CSV: {self.csv_file_path}")


def main():
    rclpy.init()
    node = PointCloudTransformer()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
