import threading

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseStream:
    def __init__(self, name="RealSense Stream"):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        self.name = name
        self.stopped = False
        self.lock = threading.Lock()

        self.depth_frame = None
        self.color_frame = None

        self.pipeline.start(self.config)

        # Align depth to color
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def start(self):
        """Start the thread to read frames from the RealSense camera"""
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        """Update frames in a continuous loop"""
        while not self.stopped:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = cv2.cvtColor(
                np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB
            )

            with self.lock:
                self.depth_frame = depth_image
                self.color_frame = color_image

    def read(self):
        """Return the current frames"""
        try:
            with self.lock:
                return True, self.color_frame
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None

    def read_color(self):
        """Return the current color frame"""
        with self.lock:
            return self.color_frame

    def read_depth(self):
        """Return the current depth frame"""
        with self.lock:
            return self.depth_frame

    def read_rgbd(self):
        """Return the current RGB-D frames"""
        with self.lock:
            return self.color_frame, self.depth_frame

    def stop(self):
        """Stop the thread and release resources"""
        self.stopped = True
        self.pipeline.stop()


if __name__ == "__main__":
    rs_stream = RealSenseStream().start()

    while True:
        color_frame, depth_frame = rs_stream.read_rgbd()

        if color_frame is not None and depth_frame is not None:
            cv2.imshow("Color Image", color_frame)
            cv2.imshow("Depth Image", depth_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    rs_stream.stop()
    cv2.destroyAllWindows()
