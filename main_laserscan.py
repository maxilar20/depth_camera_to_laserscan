import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from include.camera_stream import RealSenseStream
from include.obstacle_detector import CreatureDetector
from include.pointcloud_processor import PointCloudProcessor


class SharedData:
    """Thread-safe container for data shared between threads"""

    def __init__(self):
        # Data storage
        self._rgb_frame = None
        self._depth_frame = None
        self._pointcloud = None
        self._pointcloud_world = None
        self._laserscan = None
        self._slowdown_flag = False
        self._detections = []

        # Thread synchronization
        self._rgb_depth_lock = threading.Lock()
        self._pointcloud_lock = threading.Lock()
        self._pointcloud_world_lock = threading.Lock()
        self._laserscan_lock = threading.Lock()
        self._slowdown_lock = threading.Lock()
        self._detections_lock = threading.Lock()

        # Control flags
        self.stop_flag = False
        self.new_frame_event = threading.Event()
        self.new_pointcloud_event = threading.Event()

    # RGB-Depth frame methods
    def set_frames(self, rgb_frame, depth_frame):
        with self._rgb_depth_lock:
            self._rgb_frame = rgb_frame.copy() if rgb_frame is not None else None
            self._depth_frame = depth_frame.copy() if depth_frame is not None else None
        self.new_frame_event.set()

    def get_frames(self):
        with self._rgb_depth_lock:
            if self._rgb_frame is None or self._depth_frame is None:
                return None, None
            return self._rgb_frame.copy(), self._depth_frame.copy()

    # Pointcloud methods
    def set_pointcloud(self, pointcloud):
        with self._pointcloud_lock:
            self._pointcloud = pointcloud
        self.new_pointcloud_event.set()

    def get_pointcloud(self):
        with self._pointcloud_lock:
            return self._pointcloud

    # World pointcloud methods
    def set_pointcloud_world(self, pointcloud_world):
        with self._pointcloud_world_lock:
            self._pointcloud_world = pointcloud_world

    def get_pointcloud_world(self):
        with self._pointcloud_world_lock:
            return self._pointcloud_world

    # Laserscan methods
    def set_laserscan(self, laserscan):
        with self._laserscan_lock:
            self._laserscan = laserscan

    def get_laserscan(self):
        with self._laserscan_lock:
            if self._laserscan is None:
                return None
            return self._laserscan.copy()

    # Slowdown flag methods
    def set_slowdown(self, slowdown):
        with self._slowdown_lock:
            self._slowdown_flag = slowdown

    def get_slowdown(self):
        with self._slowdown_lock:
            return self._slowdown_flag

    # Detections methods
    def set_detections(self, detections):
        with self._detections_lock:
            self._detections = detections

    def get_detections(self):
        with self._detections_lock:
            return self._detections.copy()


class ThreadedSystem:
    def __init__(
        self,
        camera_stream,
        pointcloud_processor,
        creature_detector,
        window_name="Multi-threaded Stream",
    ):
        self.camera_stream = camera_stream
        self.pointcloud_processor = pointcloud_processor
        self.creature_detector = creature_detector
        self.window_name = window_name

        # Shared data between threads
        self.shared_data = SharedData()

        # Camera pose parameters
        self.camera_position = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self.camera_orientation = {
            "yaw": 0,
            "pitch": -90,
            "roll": 0,
        }
        self.camera_to_world = self._build_transformation_matrix()

        # Thread objects
        self.threads = []

    def _build_transformation_matrix(self):
        """
        Build a 4x4 transformation matrix to convert points from camera frame to world frame.
        """
        # Convert angles from degrees to radians
        yaw_rad = math.radians(self.camera_orientation["yaw"])
        pitch_rad = math.radians(self.camera_orientation["pitch"])
        roll_rad = math.radians(self.camera_orientation["roll"])

        # Extract position values
        x = self.camera_position["x"]
        y = self.camera_position["y"]
        z = self.camera_position["z"]

        # Rotation matrices
        R_yaw = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1],
            ]
        )
        R_pitch = np.array(
            [
                [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                [0, 1, 0],
                [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
            ]
        )
        R_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0, np.sin(roll_rad), np.cos(roll_rad)],
            ]
        )

        # Combined rotation matrix (order: yaw → pitch → roll)
        R_combined = R_yaw @ R_pitch @ R_roll

        # Build full transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R_combined
        transform[:3, 3] = [x, y, z]

        return transform

    def camera_thread_function(self):
        """Thread for camera image acquisition"""
        print("Camera thread starting...")

        # Wait for camera to warm up
        time.sleep(1.0)

        while not self.shared_data.stop_flag:
            # Get the latest RGBD frame
            rgb_frame, depth_frame = self.camera_stream.read_rgbd()

            # Check if frames are valid
            if rgb_frame is None or depth_frame is None:
                print("Warning: Empty frame received from camera")
                time.sleep(0.01)
                continue

            # Store frames in shared data
            self.shared_data.set_frames(rgb_frame, depth_frame)

            # Sleep briefly to avoid maxing out CPU
            time.sleep(0.01)

        print("Camera thread stopped")

    def pointcloud_thread_function(self):
        """Thread for pointcloud generation"""
        print("Pointcloud thread starting...")

        while not self.shared_data.stop_flag:
            # Wait for a new frame
            if not self.shared_data.new_frame_event.wait(timeout=0.1):
                continue

            # Clear the event
            self.shared_data.new_frame_event.clear()

            # Get frames
            rgb_frame, depth_frame = self.shared_data.get_frames()

            if rgb_frame is None or depth_frame is None:
                continue

            # Process frames to create pointcloud (in camera coordinates)
            pointcloud = self.pointcloud_processor.process_rgbd_image(
                rgb_frame, depth_frame
            )

            # Store pointcloud in shared data
            self.shared_data.set_pointcloud(pointcloud)

            # Transform pointcloud from camera to world coordinates
            pointcloud_copy = o3d.geometry.PointCloud(pointcloud)
            pointcloud_world = pointcloud_copy.transform(self.camera_to_world)

            # Store world-frame pointcloud in shared data
            self.shared_data.set_pointcloud_world(pointcloud_world)

        print("Pointcloud thread stopped")

    def laserscan_thread_function(self):
        """Thread for laserscan generation"""
        print("Laserscan thread starting...")

        while not self.shared_data.stop_flag:
            # Wait for a new pointcloud
            if not self.shared_data.new_pointcloud_event.wait(timeout=0.1):
                continue

            # Clear the event
            self.shared_data.new_pointcloud_event.clear()

            # Get pointcloud in world frame
            pointcloud_world = self.shared_data.get_pointcloud_world()

            if pointcloud_world is None:
                continue

            # Generate laser scan
            laser_scan = self.get_laserscan(
                pointcloud_world, num_beams=36, max_range=3.0, angle_range=180
            )

            # Store laser scan in shared data
            self.shared_data.set_laserscan(laser_scan)

        print("Laserscan thread stopped")

    def slowdown_thread_function(self):
        """Thread for slowdown flag calculation"""
        print("Slowdown thread starting...")

        while not self.shared_data.stop_flag:
            # Get pointcloud in world frame
            pointcloud_world = self.shared_data.get_pointcloud_world()

            if pointcloud_world is None:
                time.sleep(0.1)
                continue

            # Calculate slowdown flag
            slowdown = self.get_slowdown(pointcloud_world)

            # Store slowdown flag in shared data
            self.shared_data.set_slowdown(slowdown)

            # Sleep briefly to avoid constant recalculation
            time.sleep(0.1)

        print("Slowdown thread stopped")

    def detection_thread_function(self):
        """Thread for creature detection"""
        print("Detection thread starting...")

        while not self.shared_data.stop_flag:
            # Get frames
            rgb_frame, depth_frame = self.shared_data.get_frames()

            if rgb_frame is None or depth_frame is None:
                time.sleep(0.1)
                continue

            # Detect creatures
            detections = self.creature_detector.detect(rgb_frame, depth_frame)

            # Store detections in shared data
            self.shared_data.set_detections(detections)

            # Sleep to limit detection rate (creature detection is expensive)
            time.sleep(0.1)

        print("Detection thread stopped")

    def visualization_thread_function(self):
        """Thread for visualization"""
        print("Visualization thread starting...")

        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)

        # Track the geometries we've added to the visualizer
        last_pointcloud = None

        try:
            while not self.shared_data.stop_flag:
                # Get all data for visualization
                pointcloud_world = self.shared_data.get_pointcloud_world()
                laser_scan = self.shared_data.get_laserscan()
                detections = self.shared_data.get_detections()
                slowdown = self.shared_data.get_slowdown()

                # Clear the visualizer for new frame
                vis.clear_geometries()

                # Add a coordinate frame
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5
                )
                vis.add_geometry(coord_frame)

                # Add pointcloud if available
                if pointcloud_world is not None:
                    vis.add_geometry(pointcloud_world)
                    last_pointcloud = pointcloud_world

                # Add laser scan if available
                if laser_scan is not None:
                    self.visualize_laser_scan(vis, laser_scan)

                # Visualize detections if available
                if detections:
                    self.visualize_creatures(vis, detections)

                # Show slowdown indicator
                if slowdown:
                    slowdown_sphere = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.1
                    )
                    slowdown_sphere.paint_uniform_color([1, 0, 0])  # Red for slowdown
                    slowdown_sphere.translate([0, 0.5, 0])  # Position above origin
                    vis.add_geometry(slowdown_sphere)

                # Set camera view position
                ctr = vis.get_view_control()
                ctr.set_lookat([0, 0, 0])
                ctr.set_front([0, 1, 0])  # Top-down view
                ctr.set_up([0, 0, 1])

                # Update visualization
                vis.poll_events()
                vis.update_renderer()

                # Check for user input (q to quit)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.shared_data.stop_flag = True
                    break

                # Sleep briefly to limit refresh rate
                time.sleep(0.03)

        finally:
            vis.destroy_window()
            print("Visualization thread stopped")

    def start_threads(self):
        """Start all processing threads"""
        self.threads = []

        # Create and start all threads
        self.threads.append(
            threading.Thread(target=self.camera_thread_function, daemon=True)
        )
        self.threads.append(
            threading.Thread(target=self.pointcloud_thread_function, daemon=True)
        )
        self.threads.append(
            threading.Thread(target=self.laserscan_thread_function, daemon=True)
        )
        self.threads.append(
            threading.Thread(target=self.slowdown_thread_function, daemon=True)
        )
        self.threads.append(
            threading.Thread(target=self.detection_thread_function, daemon=True)
        )
        self.threads.append(
            threading.Thread(target=self.visualization_thread_function, daemon=True)
        )

        for thread in self.threads:
            thread.start()

        # Wait for stop flag or keyboard interrupt
        try:
            while not self.shared_data.stop_flag:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping threads...")
            self.shared_data.stop_flag = True

        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)

        # Clean up camera resources
        if hasattr(self.camera_stream, "stop"):
            self.camera_stream.stop()
            print("Camera resources released")

    def get_slowdown(
        self, pointcloud, max_range=1.5, z_range=(-0.35, 0.35), y_range=(-0.1, 0.1)
    ):
        """Generate a boolean slowdown flag based on the minimum distance to obstacles"""
        # Get points as numpy array
        points = np.asarray(pointcloud.points)

        if len(points) == 0:
            return False

        # Filter points within the height range (Y-axis)
        height_mask = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        filtered_points = points[height_mask]

        if len(filtered_points) == 0:
            return False

        # Filter points within the Z range
        z_mask = (filtered_points[:, 2] >= z_range[0]) & (
            filtered_points[:, 2] <= z_range[1]
        )
        filtered_points = filtered_points[z_mask]

        if len(filtered_points) == 0:
            return False

        # Calculate distances in the X direction
        distances = filtered_points[:, 0]

        # Check if any point is within max_range
        return np.any(distances <= max_range)

    def get_laserscan(
        self,
        pointcloud,
        num_beams=360,
        max_range=3.0,
        angle_range=180,
        height_range=(-0.1, 0.1),
    ):
        """Generate a 2D laser scan from a 3D point cloud"""
        # Get points as numpy array
        points = np.asarray(pointcloud.points)

        if len(points) == 0:
            return np.full(num_beams, max_range)

        # Initialize the laser scan as an array of maximum range values
        laser_scan = np.full(num_beams, max_range)

        # Define the min and max angles in degrees
        min_angle = -angle_range / 2
        max_angle = angle_range / 2
        angular_res = angle_range / (num_beams - 1) if num_beams > 1 else angle_range

        # Filter points within the height range (Y-axis)
        height_mask = (points[:, 1] >= height_range[0]) & (
            points[:, 1] <= height_range[1]
        )
        filtered_points = points[height_mask]

        if len(filtered_points) == 0:
            return laser_scan

        # Calculate distances in the XZ plane (vectorized)
        distances = np.sqrt(filtered_points[:, 0] ** 2 + filtered_points[:, 2] ** 2)

        # Filter points within max_range
        range_mask = distances <= max_range
        filtered_points = filtered_points[range_mask]
        distances = distances[range_mask]

        if len(filtered_points) == 0:
            return laser_scan

        # Calculate angles in degrees (vectorized)
        angles = np.degrees(np.arctan2(filtered_points[:, 2], filtered_points[:, 0]))

        # Filter points within the angle range
        angle_mask = (angles >= min_angle) & (angles <= max_angle)
        angles = angles[angle_mask]
        distances = distances[angle_mask]

        if len(distances) == 0:
            return laser_scan

        # Map angles to beam indices
        beam_indices = ((angles - min_angle) / angular_res).astype(int)
        beam_indices = np.clip(beam_indices, 0, num_beams - 1)

        # Use np.minimum.at for efficient in-place minimization
        np.minimum.at(laser_scan, beam_indices, distances)

        return laser_scan

    def visualize_laser_scan(self, vis, laser_scan, angle_range=180, max_range=3.0):
        """Visualize the laser scan as lines in the 3D scene"""
        num_beams = len(laser_scan)
        min_angle = -angle_range / 2
        max_angle = angle_range / 2

        # Create a line set for visualization
        points = []
        lines = []
        colors = []

        # Add origin point
        points.append([0, 0, 0])

        # For each beam in the scan
        for i, distance in enumerate(laser_scan):
            # Calculate the angle for this beam
            angle = (
                min_angle + (i / (num_beams - 1)) * angle_range if num_beams > 1 else 0
            )
            angle_rad = math.radians(angle)

            # Calculate endpoint coordinates
            x = distance * math.cos(angle_rad)
            z = distance * math.sin(angle_rad)

            # Add endpoint
            points.append([x, 0, z])

            # Add line from origin to endpoint
            lines.append([0, i + 1])

            # Color based on distance (green to red)
            normalized_dist = min(1.0, distance / max_range)
            colors.append([normalized_dist, 1 - normalized_dist, 0])  # R,G,B

        # Create and add the line set to visualizer
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(line_set)

    def visualize_creatures(self, vis, detections):
        """Visualize the detected creatures in the 3D scene"""
        # For each detected creature
        for detection in detections:
            position = detection["position"]

            # Add sphere at the position of the detection
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.compute_vertex_normals()

            # Position sphere at detection location
            sphere.translate([position["z"], 0, position["x"]])

            # Color based on class
            if detection["class"] == "person":
                sphere.paint_uniform_color([1, 0, 0])  # Red for person
            elif detection["class"] == "dog":
                sphere.paint_uniform_color([0, 1, 0])  # Green for dog
            else:
                sphere.paint_uniform_color([0, 0, 1])  # Blue for other creatures

            vis.add_geometry(sphere)


def main():
    # Initialize and start the camera stream
    camera = RealSenseStream().start()

    # Load camera intrinsics from yaml
    camera_params_file_path = Path("data/camera_calibration.yaml")
    with open(camera_params_file_path, "r") as file:
        camera_params = yaml.safe_load(file)

    camera_intrinsics = camera_params.get("intrinsics")

    # Initialize processors
    pointcloud_processor = PointCloudProcessor(camera_intrinsics)
    creature_detector = CreatureDetector(confidence_threshold=0.75)

    # Initialize the threaded system
    system = ThreadedSystem(camera, pointcloud_processor, creature_detector)

    # Start all threads
    system.start_threads()


if __name__ == "__main__":
    main()
