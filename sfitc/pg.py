#!/usr/bin/env python3
# ROS2 Humble port of the ROS1 "sf" node

import csv
import cupy as cp
import cupyx.scipy.ndimage
import cv2
import itertools
import logging
import message_filters
import numpy as np
import os
import rclpy
import time
from cv_bridge import CvBridge
from datetime import datetime
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

# ----------------------------
# Helpers / constants
# ----------------------------
ROS_DATATYPE_TO_NP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def stamp_to_sec(stamp_msg) -> float:
    return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9


def create_cloud_from_np(header, fields, np_array: np.ndarray) -> PointCloud2:
    """
    Fast PointCloud2 creation (xyz-only). np_array must be (N,3) float32.
    """
    data = np_array.astype(np.float32, copy=False).tobytes()

    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = int(np_array.shape[0])
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 12
    cloud_msg.row_step = cloud_msg.point_step * int(np_array.shape[0])
    cloud_msg.is_dense = True
    cloud_msg.data = data
    return cloud_msg


# ----------------------------
# Core math utils (unchanged)
# ----------------------------
def sph_to_cart_pts(pts):
    pts[:, 1] = cp.radians(pts[:, 1])
    pts[:, 2] = cp.radians(pts[:, 2])

    x = pts[:, 0] * cp.cos(pts[:, 1]) * cp.cos(pts[:, 2])
    y = pts[:, 0] * cp.cos(pts[:, 1]) * cp.sin(pts[:, 2])
    z = pts[:, 0] * cp.sin(pts[:, 1])

    return cp.asarray([x, y, z]).T


def cart_to_sph_pts(pts):
    pts = cp.asarray(pts)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = cp.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = cp.arctan(z / cp.sqrt(x ** 2 + y ** 2))
    phi = cp.arctan(y / x)

    return cp.column_stack((r, cp.degrees(theta), cp.degrees(phi)))


def msg2pts(msg: PointCloud2):
    wanted_fields = ["x", "y", "z"]
    field_map = {f.name: f for f in msg.fields if f.name in wanted_fields}

    offsets = [field_map[name].offset for name in wanted_fields]
    structured_dtype = np.dtype(
        {
            "names": wanted_fields,
            "formats": [ROS_DATATYPE_TO_NP[field_map[n].datatype] for n in wanted_fields],
            "offsets": offsets,
            "itemsize": msg.point_step,
        }
    )

    np_pts = np.frombuffer(msg.data, dtype=structured_dtype, count=msg.width * msg.height)
    xyz = np.stack([np_pts[name].astype(np.float32) for name in wanted_fields], axis=-1)
    return cp.asarray(xyz)


def remap(old_value, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((old_value - old_min) * new_range) / old_range) + new_min


# ----------------------------
# Filters / PG (mostly unchanged; logging adapted)
# ----------------------------
class LPFCache:
    def __init__(self):
        self.mask = None
        self.cfg = None
        self.warned_unknown = False

    @staticmethod
    def brick_wall_mask(shape, ncutoff):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2

        y = cp.arange(rows, dtype=cp.int32) - crow
        x = cp.arange(cols, dtype=cp.int32) - ccol
        Y, X = cp.meshgrid(y, x, indexing="ij")
        r2 = (X / (ccol * ncutoff)) ** 2 + (Y / (crow * ncutoff)) ** 2
        return (r2 <= 1.0).astype(cp.float32)

    @staticmethod
    def butterworth_mask(shape, D0, n=2):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        y = cp.arange(rows) - crow
        x = cp.arange(cols) - ccol
        Y, X = cp.meshgrid(y, x, indexing="ij")
        D = cp.sqrt((Y / crow) ** 2 + (X / ccol) ** 2)
        H = 1 / (1 + (D / D0) ** (2 * n))
        return H.astype(cp.float32)

    @staticmethod
    def gaussian_mask(shape, sigma):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        y = cp.arange(rows) - crow
        x = cp.arange(cols) - ccol
        Y, X = cp.meshgrid(y, x, indexing="ij")
        D2 = (Y / crow) ** 2 + (X / ccol) ** 2
        H = cp.exp(-D2 / (2 * sigma ** 2))
        return H.astype(cp.float32)

    def lpf(self, img, filter_type: str, ncutoff: float, butter_order: int, logger):
        f = cp.fft.fft2(img)
        fshift = cp.fft.fftshift(f)

        ft = filter_type.lower()
        if ft == "butterworth":
            cfg = (img.shape, "butterworth", float(ncutoff), int(butter_order))
        else:
            cfg = (img.shape, ft, float(ncutoff))

        if self.mask is None or self.cfg != cfg:
            if ft == "brick_wall":
                self.mask = self.brick_wall_mask(img.shape, ncutoff)
            elif ft == "butterworth":
                self.mask = self.butterworth_mask(img.shape, D0=ncutoff, n=butter_order)
            elif ft == "gaussian":
                self.mask = self.gaussian_mask(img.shape, sigma=ncutoff)
            else:
                if not self.warned_unknown:
                    logger.warning(f"[pg] Unknown filter_type '{filter_type}', falling back to Gaussian.")
                    self.warned_unknown = True
                self.mask = self.gaussian_mask(img.shape, sigma=ncutoff)
            self.cfg = cfg

        fshift_filtered = fshift * self.mask
        f_filtered = cp.fft.ifftshift(fshift_filtered)
        img_filtered = cp.fft.ifft2(f_filtered)
        return cp.real(img_filtered)


def pg(zed_depth, vlp_depth, lpf_cache: LPFCache, filter_type, ncutoff, butter_order, threshold, logger):
    mask = ~cp.isnan(vlp_depth)
    filtered = zed_depth

    while threshold > 0:
        filtered[mask] = vlp_depth[mask]
        filtered = lpf_cache.lpf(filtered, filter_type, ncutoff, butter_order, logger)
        threshold -= 1

    return filtered


def inpaint_depth_cupy_nanaware(cu_img, iterations=5):
    img = cu_img.copy()
    mask = ~cp.isfinite(img)

    filled = img.copy()
    filled[mask] = 0.0

    kernel = cp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=cp.float32)

    for _ in range(iterations):
        valid = cp.isfinite(filled)
        weight = cupyx.scipy.ndimage.convolve(valid.astype(cp.float32), kernel, mode="constant", cval=0)
        total = cupyx.scipy.ndimage.convolve(cp.nan_to_num(filled), kernel, mode="constant", cval=0)
        update = total / (weight + 1e-6)
        filled[mask] = update[mask]

    return filled


# ----------------------------
# Projection (unchanged)
# ----------------------------
def cart_pts_to_depth_image(cart_pts, camera_info_msg, image_shape):
    fx = camera_info_msg.k[0]
    fy = camera_info_msg.k[4]
    cx = camera_info_msg.k[2]
    cy = camera_info_msg.k[5]

    Z = cart_pts[:, 0]
    X = -cart_pts[:, 1]
    Y = -cart_pts[:, 2]

    u = cp.round((X * fx) / Z + cx).astype(cp.int32)
    v = cp.round((Y * fy) / Z + cy).astype(cp.int32)

    H, W = image_shape
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z > 0) & cp.isfinite(Z)
    u = u[valid]
    v = v[valid]
    Z = Z[valid]

    flat_idx = v * W + u
    sorted_idx = cp.argsort(flat_idx)
    flat_idx = flat_idx[sorted_idx]
    Z_sorted = Z[sorted_idx]

    unique_idx, first_pos = cp.unique(flat_idx, return_index=True)
    Z_min = Z_sorted[first_pos]

    depth_img_flat = cp.full(H * W, cp.nan, dtype=cp.float32)
    depth_img_flat[unique_idx] = Z_min
    return depth_img_flat.reshape(H, W)


def depth_to_cart_pts(depth, camera_info_msg):
    fx = camera_info_msg.k[0]
    fy = camera_info_msg.k[4]
    cx = camera_info_msg.k[2]
    cy = camera_info_msg.k[5]

    rows, cols = depth.shape
    u, v = cp.meshgrid(cp.arange(cols), cp.arange(rows))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    valid_mask = (Z.flatten() > 0) & cp.isfinite(Z.flatten())

    cart_pts = cp.stack((Z, -X, -Y), axis=-1).reshape(-1, 3)
    cart_pts = cart_pts[valid_mask]
    return cart_pts


# ----------------------------
# ROS2 Node
# ----------------------------
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__("sf")

        # ---- Params (topics) ----
        self.declare_parameter("zed_depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("zed_camera_info_topic", "/zed/zed_node/depth/camera_info")
        self.declare_parameter("zed_rgb_topic", "/zed/zed_node/left/image_rect_color")
        self.declare_parameter("vlp_topic", "/cumulative_origin_point_cloud")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")

        self.declare_parameter("vlp_depth_topic", "/islam/vlp_depth")
        self.declare_parameter("pg_depth_topic", "/islam/pg_depth")
        self.declare_parameter("pg_camera_info_topic", "/islam/pg_camera_info")
        self.declare_parameter("pg_rgb_topic", "/islam/pg_rgb")
        self.declare_parameter("pg_odom_topic", "/islam/pg_odom")
        self.declare_parameter("pg_fused_pc_topic", "/islam/pg_fused_pointcloud")
        self.declare_parameter("zed_pc_topic", "/islam/zed_pointcloud")
        self.declare_parameter("zed_original_pc_topic", "/islam/zed_original_pointcloud")
        self.declare_parameter("vlp_filtered_pc_topic", "/islam/vlp_filtered_pointcloud")
        self.declare_parameter("vlp_debug_pc_topic", "/islam/vlp_debug_pointcloud")

        # ---- Params (algo) ----
        self.declare_parameter("current_ncutoff", 0.16)
        self.declare_parameter("current_ncutoff_l", 0.08)
        self.declare_parameter("current_ncutoff_h", 0.08)
        self.declare_parameter("current_threshold", 10)
        self.declare_parameter("mortal_rows_top", 250)
        self.declare_parameter("mortal_rows_bottom", 320)
        self.declare_parameter("mortal_columns_left", 70)
        self.declare_parameter("mortal_columns_right", 10)
        self.declare_parameter("zed_vlp_diff_max", 500.0)

        self.declare_parameter("filter_type", "gaussian")  # gaussian|butterworth|brick_wall
        self.declare_parameter("butterworth_order", 3)

        # ---- Params (hardware specs) ----
        self.declare_parameter("zed_h_angle", 110.0)
        self.declare_parameter("zed_v_angle", 70.0)
        self.declare_parameter("zed_max", 50.0)
        self.declare_parameter("zed_min", 0.0)

        self.declare_parameter("lidar_v", 16)
        self.declare_parameter("lidar_angle", 30.2)
        self.declare_parameter("lidar_max", 100.0)
        self.declare_parameter("lidar_min", 0.5)

        # ---- Debug toggles (ROS1 globals -> ROS2 params) ----
        self.declare_parameter("synthetic_data", False)
        self.declare_parameter("publish_pc", True)
        self.declare_parameter("publish_aux", False)

        # ---- Read params ----
        self.ZED_DEPTH_TOPIC = self.get_parameter("zed_depth_topic").value
        self.ZED_CAMERA_INFO_TOPIC = self.get_parameter("zed_camera_info_topic").value
        self.ZED_RGB_TOPIC = self.get_parameter("zed_rgb_topic").value
        self.VLP_TOPIC = self.get_parameter("vlp_topic").value
        self.ODOM_TOPIC = self.get_parameter("odom_topic").value

        self.VLP_DEPTH_TOPIC = self.get_parameter("vlp_depth_topic").value
        self.PG_DEPTH_TOPIC = self.get_parameter("pg_depth_topic").value
        self.PG_CAMERA_INFO_TOPIC = self.get_parameter("pg_camera_info_topic").value
        self.PG_RGB_TOPIC = self.get_parameter("pg_rgb_topic").value
        self.PG_ODOM_TOPIC = self.get_parameter("pg_odom_topic").value
        self.PG_FUSED_PC_TOPIC = self.get_parameter("pg_fused_pc_topic").value
        self.ZED_PC_TOPIC = self.get_parameter("zed_pc_topic").value
        self.ZED_ORIGINAL_PC_TOPIC = self.get_parameter("zed_original_pc_topic").value
        self.VLP_FILTERED_PC_TOPIC = self.get_parameter("vlp_filtered_pc_topic").value
        self.VLP_DEBUG_PC_TOPIC = self.get_parameter("vlp_debug_pc_topic").value

        self.CURRENT_NCUTOFF = float(self.get_parameter("current_ncutoff").value)
        self.CURRENT_THRESHOLD = int(self.get_parameter("current_threshold").value)
        self.MORTAL_ROWS_TOP = int(self.get_parameter("mortal_rows_top").value)
        self.MORTAL_ROWS_BOTTOM = int(self.get_parameter("mortal_rows_bottom").value)
        self.MORTAL_COLUMNS_LEFT = int(self.get_parameter("mortal_columns_left").value)
        self.MORTAL_COLUMNS_RIGHT = int(self.get_parameter("mortal_columns_right").value)
        self.ZED_VLP_DIFF_MAX = float(self.get_parameter("zed_vlp_diff_max").value)

        self.FILTER_TYPE = str(self.get_parameter("filter_type").value)
        self.BUTTERWORTH_ORDER = int(self.get_parameter("butterworth_order").value)

        self.ZED_H_ANGLE = float(self.get_parameter("zed_h_angle").value)
        self.ZED_V_ANGLE = float(self.get_parameter("zed_v_angle").value)
        self.ZED_MAX = float(self.get_parameter("zed_max").value)
        self.ZED_MIN = float(self.get_parameter("zed_min").value)

        self.SYNTHETIC_DATA = bool(self.get_parameter("synthetic_data").value)
        self.PUBLISH_PC = bool(self.get_parameter("publish_pc").value)
        self.PUBLISH_AUX = bool(self.get_parameter("publish_aux").value)

        # ---- Runtime state ----
        self.bridge = CvBridge()
        self.ZED_V = None
        self.ZED_H = None
        self.zed_depth_frame_id = None

        self.cp_to_np_time_ms = 0.0
        self.pc_to_msg_time_ms = 0.0

        self.lpf_cache = LPFCache()

        # ---- CSV logging ----
        log_dir = os.path.expanduser("./logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"fusion_performance_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.log_file = open(log_filename, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(
            [
                "Timestamp",  # time.time(),
                "FrameNumber",  # self.frame_counter,

                "fusion_time_ms",  # fusion_time_ms,
                "zed_preproc_time_ms",  # zed_preproc_time_ms,
                "vlp_preproc_time_ms",  # vlp_preproc_time_ms,
                "msg_to_pts_time_ms",  # msg_to_pts_time_ms,
                "cart_to_pts_time_ms",  # cart_to_pts_time_ms,
                "remap_time_ms",  # remap_time_ms,
                "vlp_mean_time_ms",  # float(vlp_mean.get()) if hasattr(vlp_mean, "get") else float(vlp_mean),
                "scatter_time_ms",  # scatter_time_ms,

                "vlp_filtered_time_ms",  # vlp_filtered_time_ms,
                "sph_to_cart_pts_time_ms",  # sph_to_cart_pts_time_ms,
                "filtered_publish_time_ms",  # filtered_publish_time_ms,

                "zed_pc_time_ms",  # zed_pc_time_ms,
                "depth_to_cart_time_ms",  # depth_to_cart_time_ms,
                "pts_to_pc_time_ms",  # pts_to_pc_time_ms,
                "cp_to_np_time_ms",  # self.cp_to_np_time_ms,
                "pc_to_msg_time_ms",  # self.pc_to_msg_time_ms,
                "vlp_depth_time_ms",  # vlp_depth_time_ms,
                "pg_depth_time_ms",  # pg_depth_time_ms,
                "aux_time_ms",  # aux_time_ms,

                "TotalCallbackTime_ms",  # total_processing_time_ms,
                "ncutoff",  # self.CURRENT_NCUTOFF,
                "threshold",  # self.CURRENT_THRESHOLD,
            ]
        )

        # perf report
        self.processing_times = []
        self.frame_counter = 0
        self.last_report_time = time.time()
        self.report_interval_frames = 100
        self.report_interval_seconds = 5

        # xyz-only fields (for created clouds)
        self.xyz_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # ---- Publishers ----
        self.vlp_depth_p = self.create_publisher(Image, self.VLP_DEPTH_TOPIC, 50)
        self.pg_depth_p = self.create_publisher(Image, self.PG_DEPTH_TOPIC, 50)
        self.pg_camera_info_p = self.create_publisher(CameraInfo, self.PG_CAMERA_INFO_TOPIC, 50)
        self.pg_rgb_p = self.create_publisher(Image, self.PG_RGB_TOPIC, 50)
        self.pg_odom_p = self.create_publisher(Odometry, self.PG_ODOM_TOPIC, 10)

        self.pg_fused_pc_p = self.create_publisher(PointCloud2, self.PG_FUSED_PC_TOPIC, 10)
        self.zed_pc_p = self.create_publisher(PointCloud2, self.ZED_PC_TOPIC, 10)
        self.zed_original_pc_p = self.create_publisher(PointCloud2, self.ZED_ORIGINAL_PC_TOPIC, 10)
        self.vlp_filtered_pc_p = self.create_publisher(PointCloud2, self.VLP_FILTERED_PC_TOPIC, 10)
        self.vlp_debug_pc_p = self.create_publisher(PointCloud2, self.VLP_DEBUG_PC_TOPIC, 10)

        # ---- message_filters subs + Approx sync ----
        self.depth_sub = message_filters.Subscriber(self, Image, self.ZED_DEPTH_TOPIC)
        self.vlp_sub = message_filters.Subscriber(self, PointCloud2, self.VLP_TOPIC)
        self.rgb_sub = message_filters.Subscriber(self, Image, self.ZED_RGB_TOPIC)
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, self.ZED_CAMERA_INFO_TOPIC)
        self.odom_sub = message_filters.Subscriber(self, Odometry, self.ODOM_TOPIC)

        self.get_logger().info(
            "Subscribing to topics:\n"
            f"{self.ZED_DEPTH_TOPIC}\n"
            f"{self.VLP_TOPIC}\n"
            f"{self.ZED_RGB_TOPIC}\n"
            f"{self.ZED_CAMERA_INFO_TOPIC}\n"
            f"{self.ODOM_TOPIC}\n"
        )

        self.ats = message_filters.ApproximateTimeSynchronizer([
            self.depth_sub,
            self.vlp_sub,
            self.rgb_sub,
            self.cam_info_sub,
            # self.odom_sub
        ],
            queue_size=100,
            slop=100.0,
            allow_headerless=False,
        )
        self.ats.registerCallback(self.synchronized_callback)

    def destroy_node(self):
        try:
            self.get_logger().info("Shutting down sensor fusion node. Closing log file.")
            if hasattr(self, "log_file") and self.log_file and (not self.log_file.closed):
                self.log_file.flush()
                self.log_file.close()
        finally:
            super().destroy_node()

    def cart_pts_to_pc_msg(self, cart_pts: cp.ndarray, header):
        self.cp_to_np_time_ms = 0.0
        self.pc_to_msg_time_ms = 0.0

        t0 = time.time()
        cart_pts_np = cart_pts.get()
        self.cp_to_np_time_ms = (time.time() - t0) * 1000.0

        t1 = time.time()
        msg = create_cloud_from_np(header, self.xyz_fields, cart_pts_np)
        self.pc_to_msg_time_ms = (time.time() - t1) * 1000.0
        return msg

    def synchronized_callback(
        self,
        zed_img_msg: Image,
        vlp_pc_msg: PointCloud2,
        pg_rgb_msg: Image,
        pg_camera_info_msg: CameraInfo,
        # pg_odom_msg: Odometry,
    ):
        self.get_logger().info("synchronized_callback")

        total_start_time = time.time()

        # lazy init: image size + frame_id
        if self.ZED_V is None or self.ZED_H is None:
            zed_init = cp.array(self.bridge.imgmsg_to_cv2(zed_img_msg, desired_encoding="32FC1"))
            self.ZED_V, self.ZED_H = zed_init.shape
            self.zed_depth_frame_id = zed_img_msg.header.frame_id
            self.get_logger().info(f"Detected ZED Depth Frame ID: {self.zed_depth_frame_id}")
            del zed_init

        # %% VLP Preproc
        vlp_preproc_start_time = time.time()

        msg_to_pts_start_time = time.time()
        vlp_pts = msg2pts(vlp_pc_msg)
        msg_to_pts_time_ms = (time.time() - msg_to_pts_start_time) * 1000.0

        cart_to_pts_start_time = time.time()
        vlp_sph_pts_raw = cart_to_sph_pts(vlp_pts[vlp_pts[:, 0] > 0])
        cart_to_pts_time_ms = (time.time() - cart_to_pts_start_time) * 1000.0

        remap_start_time = time.time()
        mask = (vlp_sph_pts_raw[:, 2] < self.ZED_H_ANGLE / 2) & (vlp_sph_pts_raw[:, 2] > -self.ZED_H_ANGLE / 2)
        vlp_sph_pts = vlp_sph_pts_raw[mask]
        remap_time_ms = (time.time() - remap_start_time) * 1000.0

        vlp_mean_start_time = time.time()
        vlp_mean = cp.mean(vlp_sph_pts[:, 0]) if len(vlp_sph_pts) > 0 else cp.float32(0.0)
        vlp_mean_time_ms = (time.time() - vlp_mean_start_time) * 1000.0

        scatter_start_time = time.time()
        vlp_depth = cart_pts_to_depth_image(vlp_pts, pg_camera_info_msg, (self.ZED_V, self.ZED_H))
        scatter_time_ms = (time.time() - scatter_start_time) * 1000.0

        vlp_preproc_time_ms = (time.time() - vlp_preproc_start_time) * 1000.0

        # %% Publish filtered vlp point cloud
        vlp_filtered_start_time = time.time()
        sph_to_cart_pts_time_ms = 0.0
        filtered_publish_time_ms = 0.0

        if len(vlp_sph_pts) > 0 and self.PUBLISH_PC:
            sph_to_cart_pts_start_time = time.time()
            # vlp_filtered_cart_pts = sph_to_cart_pts(vlp_sph_pts.copy())
            # vlp_filtered_cart_pts_np = vlp_filtered_cart_pts.get()
            sph_to_cart_pts_time_ms = (time.time() - sph_to_cart_pts_start_time) * 1000.0

            filtered_publish_start_time = time.time()
            # header = vlp_pc_msg.header
            # header.frame_id = self.zed_depth_frame_id
            #
            # vlp_filtered_pc_msg = create_cloud_from_np(header, self.xyz_fields, vlp_filtered_cart_pts_np)
            # self.vlp_filtered_pc_p.publish(vlp_filtered_pc_msg)
            filtered_publish_time_ms = (time.time() - filtered_publish_start_time) * 1000.0

            self.get_logger().debug("vlp_filtered_pc_p.publish")

        vlp_filtered_time_ms = (time.time() - vlp_filtered_start_time) * 1000.0

        # %% ZED Preproc
        zed_preproc_start_time = time.time()

        zed_depth_original = cp.array(self.bridge.imgmsg_to_cv2(zed_img_msg, desired_encoding="32FC1"))

        # synthetic
        if self.SYNTHETIC_DATA:
            img_size = zed_depth_original.shape
            square_px = 100
            colors = (4, 6)
            rows, cols = cp.indices(img_size, dtype=cp.int32)
            checker = ((rows // square_px) ^ (cols // square_px)) & 1
            x = cp.where(checker, colors[1], colors[0]).astype(cp.float32)
            x -= 2
            zed_depth_original = x

        mask_nan_zed = cp.isnan(zed_depth_original)
        mask_inf_zed = cp.isinf(zed_depth_original)
        filled_mask = mask_nan_zed | mask_inf_zed

        zed_depth = zed_depth_original.copy()
        zed_depth = inpaint_depth_cupy_nanaware(zed_depth)
        zed_depth[~cp.isfinite(zed_depth)] = self.ZED_MAX

        zed_preproc_time_ms = (time.time() - zed_preproc_start_time) * 1000.0

        # %% Sensor Fusion
        fusion_start_time = time.time()

        t, b = self.MORTAL_ROWS_TOP, self.MORTAL_ROWS_BOTTOM
        l, r = self.MORTAL_COLUMNS_LEFT, self.MORTAL_COLUMNS_RIGHT

        zed_depth_cropped = zed_depth[t:-b, l:-r].copy()
        vlp_depth_cropped = vlp_depth[t:-b, l:-r].copy()

        vlp_depth_cropped[cp.abs(zed_depth_cropped - vlp_depth_cropped) > self.ZED_VLP_DIFF_MAX] = cp.nan

        # PG (your ROS1 currently bypassed PG by copying vlp_depth; keep same behavior)
        pg_depth_cropped = pg(
            zed_depth_cropped.copy(),
            vlp_depth_cropped.copy(),
            self.lpf_cache,
            self.FILTER_TYPE,
            float(self.CURRENT_NCUTOFF),
            int(self.BUTTERWORTH_ORDER),
            int(self.CURRENT_THRESHOLD),
            self.get_logger(),
        )
        # pg_depth_cropped = vlp_depth_cropped.copy()

        zed_depth = cp.pad(
            zed_depth_cropped,
            ((t, b), (l, r)),
            mode="constant",
            constant_values=cp.nan,
        )
        vlp_depth = cp.pad(
            vlp_depth_cropped,
            ((t, b), (l, r)),
            mode="constant",
            constant_values=cp.nan,
        )

        h, w = pg_depth_cropped.shape
        pg_depth_img = zed_depth_original.copy()
        region = pg_depth_img[t: t + h, l: l + w]
        pg_depth_img[t: t + h, l: l + w] = cp.where(cp.isnan(pg_depth_cropped), region, pg_depth_cropped)

        # remove filled pixels
        pg_depth_img[filled_mask] = cp.nan

        fusion_time_ms = (time.time() - fusion_start_time) * 1000.0

        self.processing_times.append(fusion_time_ms)
        self.frame_counter += 1

        # %% Convert fused depth to point clouds (optional)
        zed_pc_start_time = time.time()
        depth_to_cart_time_ms = 0.0
        pts_to_pc_time_ms = 0.0

        # In ROS2, CameraInfo fields are lower-case in Python: k, p, etc.
        # CameraInfo.k and .p are fixed-length arrays; don't use them directly in boolean context.
        k = np.asarray(pg_camera_info_msg.k, dtype=np.float64)
        p = np.asarray(pg_camera_info_msg.p, dtype=np.float64)
        cam_ok = (k.size == 9) and (p.size == 12) and (np.any(k != 0.0)) and (np.any(p != 0.0))

        if cam_ok and self.PUBLISH_PC:
            t0 = time.time()
            # fused_cart_pts = depth_to_cart_pts(pg_depth_img, camera_info_msg=pg_camera_info_msg)
            # zed_cart_pts = depth_to_cart_pts(zed_depth, camera_info_msg=pg_camera_info_msg)
            # zed_original_cart_pts = depth_to_cart_pts(zed_depth_original, camera_info_msg=pg_camera_info_msg)
            # vlp_cart_pts = depth_to_cart_pts(vlp_depth, camera_info_msg=pg_camera_info_msg)
            depth_to_cart_time_ms = (time.time() - t0) * 1000.0

            header = zed_img_msg.header
            header.frame_id = self.zed_depth_frame_id

            t1 = time.time()
            # pg_fused_pc_msg = self.cart_pts_to_pc_msg(fused_cart_pts, header)
            # zed_pc_msg = self.cart_pts_to_pc_msg(zed_cart_pts, header)
            # zed_original_pc_msg = self.cart_pts_to_pc_msg(zed_original_cart_pts, header)
            # vlp_pc_dbg_msg = self.cart_pts_to_pc_msg(vlp_cart_pts, header)
            pts_to_pc_time_ms = (time.time() - t1) * 1000.0

            # self.pg_fused_pc_p.publish(pg_fused_pc_msg)
            # self.zed_pc_p.publish(zed_pc_msg)
            # self.zed_original_pc_p.publish(zed_original_pc_msg)
            # self.vlp_debug_pc_p.publish(vlp_pc_dbg_msg)
            self.vlp_debug_pc_p.publish(vlp_pc_msg)
            self.get_logger().info("publish_pg")
        elif self.PUBLISH_PC and (not cam_ok):
            self.get_logger().warning("CameraInfo invalid/not ready (k/p all zeros); skipping fused point cloud publication.")

        zed_pc_time_ms = (time.time() - zed_pc_start_time) * 1000.0

        # %% Publish VLP depth image (optional)
        vlp_depth_start_time = time.time()
        if self.PUBLISH_AUX:
            vlp_depth_msg = self.bridge.cv2_to_imgmsg(vlp_depth.get(), encoding="32FC1")
            vlp_depth_msg.header.stamp = zed_img_msg.header.stamp
            vlp_depth_msg.header.frame_id = self.zed_depth_frame_id
            self.vlp_depth_p.publish(vlp_depth_msg)
        vlp_depth_time_ms = (time.time() - vlp_depth_start_time) * 1000.0
        # %% Publish PG depth image (always, like your ROS1)
        pg_depth_start_time = time.time()
        pg_depth_msg = self.bridge.cv2_to_imgmsg(pg_depth_img.get(), encoding="32FC1")
        pg_depth_msg.header.stamp = self.get_clock().now().to_msg()
        pg_depth_msg.header.frame_id = self.zed_depth_frame_id
        self.pg_depth_p.publish(pg_depth_msg)
        pg_depth_time_ms = (time.time() - pg_depth_start_time) * 1000.0
        # %% Publish aux info
        aux_start_time = time.time()
        pg_rgb_msg.header.stamp = self.get_clock().now().to_msg()
        self.pg_rgb_p.publish(pg_rgb_msg)

        if self.PUBLISH_AUX:
            pg_rgb_msg.header.stamp = zed_img_msg.header.stamp
            pg_camera_info_msg.header.stamp = zed_img_msg.header.stamp
            # pg_odom_msg.header.stamp = zed_img_msg.header.stamp
            self.pg_camera_info_p.publish(pg_camera_info_msg)
            # self.pg_odom_p.publish(pg_odom_msg)
            self.get_logger().info("pg_odom published")
        aux_time_ms = (time.time() - aux_start_time) * 1000.0

        total_processing_time_ms = (time.time() - total_start_time) * 1000.0

        # CSV row
        self.csv_writer.writerow(
            [
                time.time(),
                self.frame_counter,
                fusion_time_ms,
                zed_preproc_time_ms,
                vlp_preproc_time_ms,
                msg_to_pts_time_ms,
                cart_to_pts_time_ms,
                remap_time_ms,
                vlp_mean_time_ms,
                scatter_time_ms,
                vlp_filtered_time_ms,
                sph_to_cart_pts_time_ms,
                filtered_publish_time_ms,
                zed_pc_time_ms,
                depth_to_cart_time_ms,
                pts_to_pc_time_ms,
                self.cp_to_np_time_ms,
                self.pc_to_msg_time_ms,
                vlp_depth_time_ms,
                pg_depth_time_ms,
                aux_time_ms,
                total_processing_time_ms,
                self.CURRENT_NCUTOFF,
                self.CURRENT_THRESHOLD,
            ]
        )
        self.log_file.flush()

        # periodic report
        now = time.time()
        if self.frame_counter >= self.report_interval_frames or (
                now - self.last_report_time) >= self.report_interval_seconds:
            if self.processing_times:
                avg_fusion = sum(self.processing_times) / len(self.processing_times)
                self.get_logger().info(
                    f"--- Performance Report (Last {len(self.processing_times)} frames) ---\n"
                    f"  Parameters: ncutoff={self.CURRENT_NCUTOFF}, threshold={self.CURRENT_THRESHOLD}\n"
                    f"  Avg Fusion Time: {avg_fusion:.2f} ms\n"
                    f"  Max Fusion Time: {max(self.processing_times):.2f} ms\n"
                    f"  Min Fusion Time: {min(self.processing_times):.2f} ms\n"
                    f"  Estimated Fusion Frame Rate: {1000.0 / avg_fusion:.2f} Hz\n"
                    f"--------------------------------------------------"
                )
            self.processing_times = []
            self.frame_counter = 0
            self.last_report_time = now


def main():
    rclpy.init()
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
