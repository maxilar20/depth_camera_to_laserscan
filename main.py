import math  # Added for angle conversions
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from include.camera_stream import RealSenseStream
from include.pointcloud_processor import PointCloudProcessor


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

        # Camera pose parameters
        # The camera is mounted at position (x,y,z) in the world coordinate system
        # and rotated by (yaw, pitch, roll) angles
        self.camera_position = {
            "x": 0.0,  # Camera position along x-axis (meters)
            "y": 0.72,  # Camera position along y-axis (meters)
            "z": 0.0,  # Camera height above ground (meters)
        }

        self.camera_orientation = {
            "yaw": 0,  # Rotation around vertical axis (degrees)
            "pitch": 0,  # Rotation around lateral axis (degrees)
            "roll": -90,  # Rotation around longitudinal axis (degrees)
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
        Transform a pointcloud from camera coordinates to world coordinates

        Args:
            pointcloud: Open3D pointcloud in camera frame

        Returns:
            Open3D pointcloud in world frame
        """
        # Create new pointcloud for transformed points
        pointcloud_world = o3d.geometry.PointCloud()

        # Get points as numpy array
        points = np.asarray(pointcloud.points)

        # Add homogeneous coordinate (convert from Nx3 to Nx4)
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

        # Apply transformation (matrix multiplication)
        # Then slice to get just the XYZ coordinates (drop homogeneous coordinate)
        transformed_points = (self.camera_to_world @ points_homogeneous.T).T[:, :3]

        # Set the transformed points in the new pointcloud
        pointcloud_world.points = o3d.utility.Vector3dVector(transformed_points)

        # Copy colors if they exist
        if hasattr(pointcloud, "colors") and len(pointcloud.colors) > 0:
            pointcloud_world.colors = pointcloud.colors

        return pointcloud_world

    def start_display(self):
        """Display the camera stream in an Open3D window until 'q' is pressed"""
        # Allow the camera sensor to warm up
        time.sleep(2.0)

        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)

        print(vis)

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

        # Add the point cloud
        vis.add_geometry(pointcloud_world)


        # Add xyz axis to the visualization
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        vis.add_geometry(axis)
        # Red axis is x, green axis is y, blue axis is z

        # Change view position
        try:
            ctr = vis.get_view_control()
            ctr.set_lookat([0, 0, 0])
            ctr.set_front([1, 1, -1])
            ctr.set_up([0, 1, 0])
        except Exception as e:
            print(f"Error setting view control: {e}")

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
    viewer.start_display()


if __name__ == "__main__":
    main()
