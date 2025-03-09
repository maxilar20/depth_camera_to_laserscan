import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


class CreatureDetector:
    def __init__(self, confidence_threshold=0.5, device=None):
        """
        Initialize the creature detector with a pre-trained object detection model

        Args:
            confidence_threshold: Minimum confidence score to consider a detection valid
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.confidence_threshold = confidence_threshold

        # Set device (GPU if available, otherwise CPU)
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Load a pre-trained Faster R-CNN model
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # COCO dataset class labels that we're interested in
        self.target_classes = {
            1: "person",
            16: "cat",
            17: "dog",
            19: "sheep",
        }

        # Image transformation for the model
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def detect(self, rgb_image, depth_image):
        """
        Detect creatures (humans or animals) in the RGB image and determine their positions
        using the depth image.

        Args:
            rgb_image: RGB image as a numpy array (H x W x 3)
            depth_image: Depth image as a numpy array (H x W)

        Returns:
            List of dictionaries containing creature detections:
                [
                    {
                        'class': str,        # Class name of detected object
                        'confidence': float, # Detection confidence
                        'box': [x1, y1, x2, y2],  # Bounding box coordinates
                        'position': {
                            'x': float,      # X position in meters
                            'y': float,      # Y position in meters
                            'z': float       # Z position in meters (depth)
                        }
                    },
                    ...
                ]
        """
        start_time = time.time()

        # Convert numpy array to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

        # Transform the image as required by the model
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Get detections
        with torch.no_grad():
            predictions = self.model(input_tensor)

        detections = []

        # Process predictions
        for i, score in enumerate(predictions[0]["scores"]):
            score_value = score.item()

            # Filter by confidence and class
            if score_value >= self.confidence_threshold:
                label_idx = predictions[0]["labels"][i].item()

                # Check if the detected class is one we're interested in
                if label_idx in self.target_classes:
                    box = predictions[0]["boxes"][i].cpu().numpy().astype(int)
                    class_name = self.target_classes[label_idx]

                    # Get position from depth image
                    position = self._get_3d_position(box, depth_image)

                    detections.append(
                        {
                            "class": class_name,
                            "confidence": score_value,
                            "box": box.tolist(),
                            "position": position,
                        }
                    )

        elapsed_time = time.time() - start_time
        print(
            f"Detection took {elapsed_time:.3f} seconds. Found {len(detections)} creatures."
        )

        return detections

    def _get_3d_position(self, box, depth_image):
        """
        Calculate the 3D position of a detected object using the depth image

        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            depth_image: Depth image as numpy array (H x W)

        Returns:
            Dictionary with x, y, z positions in meters
        """
        x1, y1, x2, y2 = box

        # Calculate center position of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Get depth at the center of the bounding box
        # Handle possible out-of-bounds errors
        if (0 <= center_y < depth_image.shape[0]) and (
            0 <= center_x < depth_image.shape[1]
        ):
            # Get a small region around the center for robustness
            depth_region = depth_image[
                max(0, center_y - 5) : min(depth_image.shape[0], center_y + 5),
                max(0, center_x - 5) : min(depth_image.shape[1], center_x + 5),
            ]

            # Filter out zero values (invalid depth measurements)
            valid_depths = depth_region[depth_region > 0]

            if len(valid_depths) > 0:
                # Use median depth for robustness against outliers
                depth = np.median(valid_depths)
            else:
                # No valid depth measurements
                depth = 0
        else:
            # Center is out of bounds
            depth = 0

        # Convert depth to meters (assuming depth is in millimeters)
        depth_meters = depth / 1000.0

        # Calculate X and Y world coordinates
        # This is a simplified calculation and would need camera intrinsics for accuracy
        # Using a simple pinhole camera model approximation
        # fx and fy are the focal lengths, cx and cy are the principal points
        fx = depth_image.shape[1] / 2  # approximation
        fy = depth_image.shape[0] / 2  # approximation
        cx = depth_image.shape[1] / 2
        cy = depth_image.shape[0] / 2

        x_meters = (center_x - cx) * depth_meters / fx
        y_meters = (center_y - cy) * depth_meters / fy

        return {"x": x_meters, "y": y_meters, "z": depth_meters}

    def visualize_detections(self, rgb_image, detections):
        """
        Draw bounding boxes and position information on the RGB image

        Args:
            rgb_image: RGB image as numpy array
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        image = rgb_image.copy()

        for detection in detections:
            box = detection["box"]
            label = f"{detection['class']} ({detection['confidence']:.2f})"
            position = detection["position"]

            # Draw bounding box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Draw label
            cv2.putText(
                image,
                label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw position
            pos_text = f"X: {position['x']:.2f}m, Y: {position['y']:.2f}m, Z: {position['z']:.2f}m"
            cv2.putText(
                image,
                pos_text,
                (box[0], box[3] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return image


# Example usage:
# detector = CreatureDetector()
# detections = detector.detect(rgb_image, depth_image)
# annotated_image = detector.visualize_detections(rgb_image, detections)
# cv2.imshow("Detections", annotated_image)
# cv2.waitKey(0)
