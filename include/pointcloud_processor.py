import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from include.camera_stream import RealSenseStream


class PointCloudProcessor:
    def __init__(
        self,
        camera_intrinsics=None,
        depth_scale=1000.0,
        depth_trunc=3.0,
    ):  # Maximum depth in meters
        """
        Initialize a PointCloud processor with camera parameters.

        Args:
            camera_intrinsics: Camera intrinsic parameters (fx, fy, cx, cy)
                               If None, default parameters will be used
            depth_scale: Scale factor to convert depth values to meters
            depth_trunc: Maximum depth in meters
        """
        # Default camera intrinsics if none provided (can be calibrated later)
        if camera_intrinsics is None:
            self.fx = 525.0  # focal length x
            self.fy = 525.0  # focal length y
            self.cx = 319.5  # principal point x
            self.cy = 239.5  # principal point y
        else:
            self.fx = camera_intrinsics["fx"]
            self.fy = camera_intrinsics["fy"]
            self.cx = camera_intrinsics["cx"]
            self.cy = camera_intrinsics["cy"]

        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.vis = None

    def create_pointcloud(self, color_image, depth_image):
        """
        Convert RGB-D image to point cloud.

        Args:
            color_image: RGB image (HxWx3 numpy array)
            depth_image: Depth image (HxW numpy array)
                         Values should be in the depth_scale units

        Returns:
            open3d.geometry.PointCloud: The created point cloud
        """
        # Create Open3D images
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        # Create camera intrinsics object
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            width=color_image.shape[1],
            height=color_image.shape[0],
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )

        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

        # Flip the orientation for better visualization
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return pcd

    def display_pointcloud(self, pointcloud):
        """
        Display the point cloud in a 3D viewer.

        Args:
            pointcloud: The Open3D point cloud to display
        """
        if self.vis is None:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window()
            self.vis.add_geometry(pointcloud)

            # Set some reasonable default options for viewing
            render_option = self.vis.get_render_option()
            render_option.point_size = 1.0
            render_option.background_color = np.array([0, 0, 0])

            # Adjust the view control
            view_control = self.vis.get_view_control()
            view_control.set_zoom(0.7)
        else:
            self.vis.update_geometry(pointcloud)

        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()

    def process_rgbd_image(self, color_image, depth_image):
        """
        Process an RGB-D image pair to create a point cloud.

        Args:
            color_image: RGB image (HxWx3 numpy array)
            depth_image: Depth image (HxW numpy array)

        Returns:
            open3d.geometry.PointCloud: The created point cloud
        """
        pointcloud = self.create_pointcloud(color_image, depth_image)

        return pointcloud

    def close(self):
        """Close the visualization window."""
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None


def main():
    try:
        # Capture RGB-D image from RealSense camera
        camera_stream = RealSenseStream().start()

        time.sleep(2)
        color_image, depth_image = camera_stream.read_rgbd()

        # Load camera intrinsics from yaml
        camera_params_file_path = Path("data/camera_calibration.yaml")
        with open(camera_params_file_path, "r") as file:
            camera_params = yaml.safe_load(file)

        camera_intrinsics = camera_params.get("intrinsics")

        # Initialize the PointCloudProcessor
        processor = PointCloudProcessor(camera_intrinsics)

        # Process the RGB-D image and display the point cloud
        pointcloud = processor.process_rgbd_image(color_image, depth_image)

        processor.display_pointcloud(pointcloud)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        processor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
