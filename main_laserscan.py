import copy
import cProfile
import math
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from camera_stream import RealSenseStream
from pointcloud_processor import PointCloudProcessor


class PointcloudViewer:
    def __init__(
        self,
        camera_stream,
        pointcloud_processor,
        window_name="Live Stream",
    ):
        self.camera_stream = camera_stream
        self.pointcloud_processor = pointcloud_processor
        self.window_name = window_name
        self.stopped = False

        # Camera pose parameters - updated to align camera properly
        self.camera_position = {
            "x": 0.0,  # Camera position along x-axis (meters)
            "y": 0.0,  # Camera position along y-axis (meters)
            "z": 0.0,  # Camera height above ground (meters)
        }

        self.camera_orientation = {
            "yaw": 0,  # Rotation around vertical axis (degrees)
            "pitch": -90,  # Rotation around lateral axis (degrees)
            "roll": 0,  # Rotation around longitudinal axis (degrees)
        }

        # Build the transformation matrix
        self.camera_to_world = self._build_transformation_matrix()

    def _build_transformation_matrix(self):
        """
        Build a 4x4 transformation matrix to convert points from camera frame to world frame.

        The transformation includes:
        1. Rotation defined by yaw, pitch, roll angles
        2. Translation defined by x, y, z coordinates

        Returns:
            4x4 numpy array representing the transformation matrix
        """
        # Convert angles from degrees to radians
        yaw_rad = math.radians(self.camera_orientation["yaw"])
        pitch_rad = math.radians(self.camera_orientation["pitch"])
        roll_rad = math.radians(self.camera_orientation["roll"])

        # Extract position values
        x = self.camera_position["x"]
        y = self.camera_position["y"]
        z = self.camera_position["z"]

        # Rotation matrices (using right-hand rule)
        # Yaw rotation (around z-axis)
        R_yaw = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Pitch rotation (around y-axis)
        R_pitch = np.array(
            [
                [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                [0, 1, 0],
                [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
            ]
        )

        # Roll rotation (around x-axis)
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
        transform = np.eye(4)  # Start with identity matrix
        transform[:3, :3] = R_combined  # Set rotation part
        transform[:3, 3] = [x, y, z]  # Set translation part

        return transform

    def transform_pointcloud_to_world(self, pointcloud):
        """
        Transform a pointcloud from camera coordinates to world coordinates using open3d functions

        Args:
            pointcloud: Open3D pointcloud in camera frame

        Returns:
            Open3D pointcloud in world frame
        """
        # Create a copy of the input pointcloud
        pointcloud_world = copy.copy(pointcloud)

        # Apply the transformation matrix
        pointcloud_world.transform(self.camera_to_world)

        return pointcloud_world

    def get_laserscan(
        self,
        pointcloud,
        num_beams=360,
        max_range=3.0,
        angle_range=180,  # Changed to 180 degrees for wider coverage
        height_range=(-0.1, 0.1),  # Narrower height range to focus on a plane
    ):
        """
        Generate a 2D laser scan from a 3D point cloud using vectorized operations

        Args:
            pointcloud: Open3D pointcloud in world frame
            num_beams: Number of laser beams to simulate
            max_range: Maximum range of the laser scan
            angle_range: Total angular range in degrees
            height_range: Range of heights to consider (min, max) along Y-axis

        Returns:
            1D numpy array representing the laser scan distances
        """
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
        """
        Visualize the laser scan as lines in the 3D scene

        Args:
            vis: Open3D visualizer
            laser_scan: Laser scan distances
            angle_range: Total angular range in degrees
            max_range: Maximum range of the laser scan
        """
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

        return line_set

    def start_display(self):
        """Display the camera stream in an Open3D window until 'q' is pressed"""
        # Allow the camera sensor to warm up
        time.sleep(2.0)

        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)

        try:
            # Keep looping until 'q' is pressed
            while not self.stopped:
                self.update_display(vis)
        finally:
            # Clean up resources
            if hasattr(self.camera_stream, "stop"):
                self.camera_stream.stop()
            vis.destroy_window()
            print("Camera resources released")

    def update_display(self, vis):
        # Get the latest frame (RGBD)
        rgb_frame, depth_frame = self.camera_stream.read_rgbd()

        # Check if frames are valid
        if rgb_frame is None or depth_frame is None:
            print("Error: Empty frame received from the camera.")
            return  # Skip this iteration if frames are invalid

        # Process frames to create pointcloud (in camera coordinates)
        pointcloud = self.pointcloud_processor.process_rgbd_image(
            rgb_frame, depth_frame
        )

        # Transform pointcloud from camera to world coordinates
        pointcloud_world = self.transform_pointcloud_to_world(pointcloud)

        # Clear the visualizer and update it with the new point cloud
        vis.clear_geometries()
        vis.add_geometry(pointcloud_world)

        # Add xyz axis to the visualization
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        vis.add_geometry(axis)  # Red axis is x, green axis is y, blue axis is z

        # Get the laser scan from the point cloud
        laser_scan = self.get_laserscan(
            pointcloud_world, num_beams=36, max_range=3.0, angle_range=180
        )

        # Visualize the laser scan
        self.visualize_laser_scan(vis, laser_scan, angle_range=180, max_range=3.0)

        # Change view position
        ctr = vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_front(
            [0, 1, 0]
        )  # Changed to top-down view for better laserscan visibility
        ctr.set_up([0, 0, 1])

        # Update the visualization
        vis.poll_events()
        vis.update_renderer()

        # Check for user input - exit on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.stop_display()

    def stop_display(self):
        """Stop the display loop"""
        self.stopped = True


def main():
    # Initialize and start the camera stream
    camera = RealSenseStream().start()

    # Load camera intrinsics from yaml
    camera_params_file_path = Path("data/camera_calibration.yaml")
    with open(camera_params_file_path, "r") as file:
        camera_params = yaml.safe_load(file)

    camera_intrinsics = camera_params.get("intrinsics")

    # Initialize the PointCloudProcessor
    pointcloud_processor = PointCloudProcessor(camera_intrinsics)

    # Initialize and start the stream viewer
    viewer = PointcloudViewer(camera, pointcloud_processor)

    # Profile the start_display method
    profiler = cProfile.Profile()
    profiler.enable()
    viewer.start_display()
    profiler.disable()
    profiler.print_stats(sort="cumtime")


if __name__ == "__main__":
    main()
