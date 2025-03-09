import cProfile
import time
from pathlib import Path

import cv2
import open3d as o3d
import yaml

from camera_stream import RealSenseStream
from vslam import VisualSLAM


class PointcloudViewer:
    def __init__(
        self,
        camera_stream,
        vslam,
        window_name="Live Stream",
    ):
        self.camera_stream = camera_stream
        self.vslam = vslam
        self.window_name = window_name
        self.stopped = False

    def start_display(self):
        """Display the camera stream in an Open3D window until 'q' is pressed or Ctrl+C is received"""
        # Allow the camera sensor to warm up
        time.sleep(2.0)

        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)

        try:
            # Keep looping until 'q' is pressed
            while not self.stopped:
                self.update_display(vis)
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping display.")
            self.stop_display()
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

        vis.clear_geometries()

        pose, updated = self.vslam.process_frame(rgb_frame, depth_frame)

        print("Pose: ", pose)

        map = self.vslam.get_map()
        if map is not None:
            vis.add_geometry(map)

        # # Change view position
        # ctr = vis.get_view_control()
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_front(
        #     [0, 1, 0]
        # )  # Changed to top-down view for better laserscan visibility
        # ctr.set_up([0, 0, 1])

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

    # Initialize the VSLAM module
    vslam = VisualSLAM(camera_intrinsics)

    # Initialize and start the stream viewer
    viewer = PointcloudViewer(camera, vslam)

    # Profile the start_display method
    profiler = cProfile.Profile()
    profiler.enable()
    viewer.start_display()
    profiler.disable()
    profiler.print_stats(sort="cumtime")


if __name__ == "__main__":
    main()
