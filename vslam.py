import time
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d


class VisualSLAM:
    def __init__(self, camera_intrinsics, min_matches=10, use_ransac=True):
        """
        Initialize the Visual SLAM system

        Args:
            camera_intrinsics: Dictionary containing camera intrinsics parameters:
                - fx: focal length x
                - fy: focal length y
                - cx: principal point x
                - cy: principal point y
            min_matches: Minimum number of feature matches to estimate motion
            use_ransac: Whether to use RANSAC for robust motion estimation
        """
        # Camera parameters
        self.K = np.array(
            [
                [camera_intrinsics["fx"], 0, camera_intrinsics["cx"]],
                [0, camera_intrinsics["fy"], camera_intrinsics["cy"]],
                [0, 0, 1],
            ]
        )
        self.fx = camera_intrinsics["fx"]
        self.fy = camera_intrinsics["fy"]
        self.cx = camera_intrinsics["cx"]
        self.cy = camera_intrinsics["cy"]

        # SLAM parameters
        self.min_matches = min_matches
        self.use_ransac = use_ransac

        # Feature detection and matching
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # State variables
        self.prev_rgb = None
        self.prev_depth = None
        self.prev_kp = None
        self.prev_des = None
        self.curr_pose = np.eye(4)  # Initial pose is identity
        self.poses = [np.eye(4)]  # Store all poses

        # Map representation
        self.global_map = o3d.geometry.PointCloud()
        self.keyframes = []  # Store key frames for loop closure
        self.frame_count = 0

        # Feature tracking
        self.feature_tracks = defaultdict(list)  # Track features across frames
        self.next_feature_id = 0

        print("Visual SLAM initialized with camera intrinsics:")
        print(f"fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}")

    def process_frame(self, rgb_image, depth_image):
        """
        Process a new RGB-D frame

        Args:
            rgb_image: RGB image as numpy array (H x W x 3)
            depth_image: Depth image as numpy array (H x W), values in mm

        Returns:
            current_pose: 4x4 transformation matrix representing current camera pose
            updated: True if pose was successfully updated, False otherwise
        """
        start_time = time.time()

        # Convert RGB to grayscale for feature detection
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image

        # Extract features
        kp, des = self.orb.detectAndCompute(gray, None)

        # First frame - just store it
        if self.prev_rgb is None:
            self.prev_rgb = gray
            self.prev_depth = depth_image
            self.prev_kp = kp
            self.prev_des = des
            self.frame_count += 1
            print(f"First frame initialized with {len(kp)} features")
            return self.curr_pose, False

        # Match features with previous frame
        if (
            len(kp) > 0
            and len(self.prev_kp) > 0
            and des is not None
            and self.prev_des is not None
        ):
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # Filter matches by distance
            good_matches = matches[: min(100, len(matches))]

            print(f"Found {len(good_matches)} good matches")

            if len(good_matches) < self.min_matches:
                print(
                    f"Not enough good matches: {len(good_matches)}/{self.min_matches}"
                )
                self.prev_rgb = gray
                self.prev_depth = depth_image
                self.prev_kp = kp
                self.prev_des = des
                self.frame_count += 1
                return self.curr_pose, False

            # Extract matched keypoint positions
            prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
            curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])

            # Get 3D coordinates from previous frame
            prev_points_3d = []
            valid_matches = []

            for i, (u, v) in enumerate(prev_pts):
                u, v = int(u), int(v)
                if (
                    0 <= v < self.prev_depth.shape[0]
                    and 0 <= u < self.prev_depth.shape[1]
                ):
                    z = self.prev_depth[v, u] / 1000.0  # Convert mm to meters
                    if z > 0 and z < 5.0:  # Add maximum depth check
                        # Convert to camera coordinates with Y-flip
                        x = (u - self.cx) * z / self.fx
                        y = -((v - self.cy) * z / self.fy)  # Add negative sign
                        prev_points_3d.append([x, y, z])
                        valid_matches.append(i)

            if len(valid_matches) < self.min_matches:
                print(
                    f"Not enough valid 3D points: {len(valid_matches)}/{self.min_matches}"
                )
                self.prev_rgb = gray
                self.prev_depth = depth_image
                self.prev_kp = kp
                self.prev_des = des
                self.frame_count += 1
                return self.curr_pose, False

            # Extract only valid matches
            prev_points_3d = np.array(prev_points_3d)
            curr_pts = curr_pts[valid_matches]

            # Estimate motion using PnP (Perspective-n-Point)
            if self.use_ransac:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(
                    prev_points_3d, curr_pts, self.K, None
                )
                if inliers is None or len(inliers) < self.min_matches:
                    print(
                        f"Not enough PnP inliers: {0 if inliers is None else len(inliers)}/{self.min_matches}"
                    )
                    self.prev_rgb = gray
                    self.prev_depth = depth_image
                    self.prev_kp = kp
                    self.prev_des = des
                    self.frame_count += 1
                    return self.curr_pose, False
            else:
                _, rvec, tvec = cv2.solvePnP(prev_points_3d, curr_pts, self.K, None)

            # Convert rotation vector to rotation matrix
            R_mat, _ = cv2.Rodrigues(rvec)

            # Create transformation matrix from current to previous frame
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = tvec.flatten()

            # Update global pose (invert T because we want prev_frame -> curr_frame)
            T_inv = np.eye(4)
            T_inv[:3, :3] = R_mat.T
            T_inv[:3, 3] = -R_mat.T @ tvec.flatten()

            # Update current pose
            self.curr_pose = self.curr_pose @ T_inv
            self.poses.append(self.curr_pose.copy())

            # Update map with new points from current frame
            if self.frame_count % 5 == 0:  # Update map every 5 frames
                self._update_map(rgb_image, depth_image)

            # Check if this should be a keyframe
            if self.frame_count % 10 == 0:
                self._add_keyframe(rgb_image, depth_image, kp, des)

            # Update previous frame data
            self.prev_rgb = gray
            self.prev_depth = depth_image
            self.prev_kp = kp
            self.prev_des = des

            self.frame_count += 1
            elapsed = time.time() - start_time
            print(
                f"Frame {self.frame_count}: Motion estimated with {len(valid_matches)} points in {elapsed:.3f}s"
            )

            return self.curr_pose, True
        else:
            print("No features detected or matched")
            self.prev_rgb = gray
            self.prev_depth = depth_image
            self.prev_kp = kp
            self.prev_des = des
            self.frame_count += 1
            return self.curr_pose, False

    def _update_map(self, rgb_image, depth_image):
        """
        Update the global map with points from current RGB-D frame with fixed coordinate system

        Args:
            rgb_image: RGB image
            depth_image: Depth image
        """
        # Skip if depth image is empty
        if depth_image is None:
            return

        # Downsample depth image for efficiency
        step = 8  # Process every 8th pixel
        height, width = depth_image.shape

        # Create point cloud from depth image
        points = []
        colors = []

        for v in range(0, height, step):
            for u in range(0, width, step):
                z = depth_image[v, u] / 1000.0  # Convert mm to meters
                # Filter out invalid or too far points
                if z > 0 and z < 5.0:
                    # Convert to camera coordinates
                    # BUT flip the Y axis to match the world coordinate system
                    x = (u - self.cx) * z / self.fx
                    y = -((v - self.cy) * z / self.fy)  # Note the negative sign here!
                    z = z  # z remains the same

                    points.append([x, y, z])

                    # Add color
                    if len(rgb_image.shape) == 3:  # Color image
                        b, g, r = rgb_image[v, u] / 255.0
                        colors.append([r, g, b])
                    else:  # Grayscale image
                        gray = rgb_image[v, u] / 255.0
                        colors.append([gray, gray, gray])

        if len(points) == 0:
            return

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Filter outlier points
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Transform to global coordinates
        pcd.transform(self.curr_pose)

        # Add to global map
        self.global_map += pcd

        # Voxel downsample the global map to keep size manageable
        self.global_map = self.global_map.voxel_down_sample(voxel_size=0.01)

    def _add_keyframe(self, rgb_image, depth_image, keypoints, descriptors):
        """
        Add a keyframe for loop closure detection

        Args:
            rgb_image: RGB image
            depth_image: Depth image
            keypoints: Detected keypoints
            descriptors: Feature descriptors
        """
        keyframe = {
            "frame_id": self.frame_count,
            "pose": self.curr_pose.copy(),
            "keypoints": keypoints,
            "descriptors": descriptors,
        }
        self.keyframes.append(keyframe)
        print(f"Added keyframe {self.frame_count}")

    def get_current_pose(self):
        """
        Get the current camera pose

        Returns:
            4x4 transformation matrix representing current camera pose
        """
        return self.curr_pose

    def get_trajectory(self):
        """
        Get the camera trajectory

        Returns:
            List of 4x4 transformation matrices representing camera poses
        """
        return self.poses

    def get_map(self):
        """
        Get the global point cloud map

        Returns:
            Open3D point cloud
        """
        return self.global_map

    def visualize(self, window_name="VSLAM"):
        """
        Visualize the current map and trajectory

        Args:
            window_name: Name of visualization window
        """
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coord_frame)

        # Add global map
        vis.add_geometry(self.global_map)

        # Add trajectory
        trajectory = o3d.geometry.LineSet()
        points = []
        lines = []
        colors = []

        for i, pose in enumerate(self.poses):
            points.append(pose[:3, 3])
            if i > 0:
                lines.append([i - 1, i])
                colors.append([1, 0, 0])  # Red trajectory

        trajectory.points = o3d.utility.Vector3dVector(points)
        trajectory.lines = o3d.utility.Vector2iVector(lines)
        trajectory.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(trajectory)

        # Set view
        vis.poll_events()
        vis.update_renderer()

        while True:
            vis.poll_events()
            vis.update_renderer()
            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:  # 'q' or ESC
                break

        vis.destroy_window()


# Example usage:
# camera_intrinsics = {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}
# vslam = VisualSLAM(camera_intrinsics)
#
# for frame_idx in range(num_frames):
#     rgb_image = read_rgb_image(frame_idx)
#     depth_image = read_depth_image(frame_idx)
#
#     pose, updated = vslam.process_frame(rgb_image, depth_image)
#
#     if frame_idx % 10 == 0:
#         print(f"Current pose:\n{pose}")
#
# vslam.visualize()
