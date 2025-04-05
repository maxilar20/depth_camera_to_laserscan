import glob
import os
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

from include.camera_stream import RealSenseStream


class CameraIntrinsicCalibrator:
    def __init__(
        self,
        camera_stream,
        checkerboard_size=(9, 6),  # Number of inner corners (width, height)
        square_size=0.025,  # Physical size of squares in meters
    ):
        """
        Initialize a camera calibrator to determine intrinsic parameters.

        Args:
            checkerboard_size: Tuple with number of inner corners of the calibration checkerboard
            square_size: Physical size of each square in meters
        """

        self.camera_stream = camera_stream

        self.checkerboard_size = checkerboard_size
        self.square_size = square_size

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... in real world space
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : checkerboard_size[0], 0 : checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size  # Scale to real-world size

        # Arrays to store object points and image points from all images
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None
        self.calibration_error = None
        self.calibration_flags = cv2.CALIB_FIX_ASPECT_RATIO

    def capture_calibration_images(self, num_images=20, delay=1):
        """
        Capture images for calibration from the camera.

        Args:
            num_images: Number of images to capture for calibration
            delay: Delay between captures in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        print("Press 'c' to capture an image, 'q' to quit...")

        count = 0
        saved_frames = []

        time.sleep(1.0)

        while count < num_images:
            ret, frame = self.camera_stream.read()

            if not ret:
                print("Error reading frame")
                break

            # Draw frame count
            cv2.putText(
                frame,
                f"Captured: {count}/{num_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show live view
            cv2.imshow("Camera Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Save current frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, self.checkerboard_size, None
                )

                if ret:
                    saved_frames.append(frame.copy())
                    print(f"Image {count + 1} captured. Checkerboard detected!")
                    count += 1

                    # Draw and display the corners
                    corners_refined = cv2.cornerSubPix(
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )
                    cv2.drawChessboardCorners(
                        frame, self.checkerboard_size, corners_refined, ret
                    )
                    cv2.imshow("Corners Detected", frame)
                    cv2.waitKey(500)
                else:
                    print("Checkerboard not detected. Please adjust the position.")

        self.camera_stream.stop()
        cv2.destroyAllWindows()

        # Process all saved frames
        if saved_frames:
            self._process_calibration_images(saved_frames)
            return True
        else:
            return False

    def calibrate_from_images(self, image_folder):
        """
        Calibrate camera using existing images from a folder.

        Args:
            image_folder: Path to folder containing calibration images

        Returns:
            bool: True if successful, False otherwise
        """
        # Get all image files
        image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(
            os.path.join(image_folder, "*.png")
        )

        if not image_paths:
            print(f"No images found in {image_folder}")
            return False

        print(f"Found {len(image_paths)} images for calibration")

        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)

        if images:
            self._process_calibration_images(images)
            return True
        else:
            return False

    def _process_calibration_images(self, images):
        """
        Process images to find checkerboard corners and perform calibration.

        Args:
            images: List of images (numpy arrays)
        """
        self.objpoints = []
        self.imgpoints = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            # If found, add object points, image points
            if ret:
                self.objpoints.append(self.objp)

                # Refine corner locations
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                self.imgpoints.append(corners_refined)

                # Save image size for calibration
                if self.image_size is None:
                    self.image_size = gray.shape[::-1]

        # Perform calibration if we have enough data points
        if self.imgpoints:
            self._calibrate()
        else:
            print("No checkerboard corners detected in any image")

    def _calibrate(self):
        """Perform camera calibration using collected points."""
        if self.image_size is None or not self.objpoints or not self.imgpoints:
            print("Not enough data for calibration")
            return

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.image_size,
            None,
            None,
            flags=self.calibration_flags,
        )

        # Calculate re-projection error
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(
                imgpoints2
            )
            mean_error += error

        self.calibration_error = mean_error / len(self.objpoints)
        self.camera_matrix = mtx
        self.dist_coeffs = dist

        print(f"Calibration complete! Re-projection error: {self.calibration_error}")
        print(f"Camera matrix:\n{self.camera_matrix}")
        print(f"Distortion coefficients:\n{self.dist_coeffs}")

    def save_calibration(self, filename=None):
        """
        Save calibration parameters to a YAML file.

        Args:
            filename: Optional filename to save calibration to.
                     If None, a default name with timestamp is used.

        Returns:
            str: Path to the saved file if successful, None otherwise
        """
        if self.camera_matrix is None:
            print("No calibration data to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_calibration_{timestamp}.yaml"

        # Create a dictionary with calibration data
        calibration_data = {
            "image_width": int(self.image_size[0]),
            "image_height": int(self.image_size[1]),
            "camera_matrix": {
                "rows": int(self.camera_matrix.shape[0]),
                "cols": int(self.camera_matrix.shape[1]),
                "data": self.camera_matrix.flatten().tolist(),
            },
            "distortion_coefficients": {
                "rows": int(self.dist_coeffs.shape[0]),
                "cols": int(self.dist_coeffs.shape[1]),
                "data": self.dist_coeffs.flatten().tolist(),
            },
            "calibration_error": float(self.calibration_error),
            "calibration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Extract intrinsics in a more direct format for easy access
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        calibration_data["intrinsics"] = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }

        # Save to file
        try:
            with open(filename, "w") as f:
                yaml.dump(calibration_data, f, default_flow_style=False)
            print(f"Calibration saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return None

    def load_calibration(self, filename):
        """
        Load calibration parameters from a YAML file.

        Args:
            filename: Path to the calibration YAML file

        Returns:
            dict: Dictionary with intrinsic parameters if successful, None otherwise
        """
        try:
            with open(filename, "r") as f:
                calibration_data = yaml.safe_load(f)

            # Extract camera matrix
            camera_matrix_data = calibration_data.get("camera_matrix", {}).get(
                "data", []
            )
            rows = calibration_data.get("camera_matrix", {}).get("rows", 3)
            cols = calibration_data.get("camera_matrix", {}).get("cols", 3)
            self.camera_matrix = np.array(camera_matrix_data).reshape(rows, cols)

            # Extract distortion coefficients
            dist_data = calibration_data.get("distortion_coefficients", {}).get(
                "data", []
            )
            rows = calibration_data.get("distortion_coefficients", {}).get("rows", 1)
            cols = calibration_data.get("distortion_coefficients", {}).get("cols", 5)
            self.dist_coeffs = np.array(dist_data).reshape(rows, cols)

            # Extract other parameters
            self.image_size = (
                calibration_data.get("image_width", 640),
                calibration_data.get("image_height", 480),
            )
            self.calibration_error = calibration_data.get("calibration_error", None)

            print(f"Calibration loaded from {filename}")

            # Return intrinsics as a dict for easy use
            return calibration_data.get(
                "intrinsics",
                {
                    "fx": float(self.camera_matrix[0, 0]),
                    "fy": float(self.camera_matrix[1, 1]),
                    "cx": float(self.camera_matrix[0, 2]),
                    "cy": float(self.camera_matrix[1, 2]),
                },
            )
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return None


def main():
    # Initialize the stream
    rs_stream = RealSenseStream().start()

    # Initialize the calibrator
    calibrator = CameraIntrinsicCalibrator(
        rs_stream,
        checkerboard_size=(10, 7),  # Adjust according to your checkerboard
        square_size=0.025,  # Size in meters
    )

    # Capture calibration images
    print("Starting camera calibration...")
    success = calibrator.capture_calibration_images(num_images=15)

    if success:
        # Save calibration to file
        calibrator.save_calibration()


if __name__ == "__main__":
    main()
