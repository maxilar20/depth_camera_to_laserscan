import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101


class FloorSegmentation:
    def __init__(self, model_path=None, use_cpu=False, max_image_size=None, target_class_id=None, debug=False):
        # Initialize device based on parameters and availability
        if use_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load pre-trained DeepLabV3+ model
        try:
            # Use the updated API to avoid deprecated warning
            from torchvision.models.segmentation.deeplabv3 import (
                DeepLabV3_ResNet101_Weights,
            )
            self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        except ImportError:
            # Fall back to the older API if the new one is not available
            self.model = deeplabv3_resnet101(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        # Store the maximum image size for resizing large images
        self.max_image_size = max_image_size
        self.debug = debug

        # In the DeepLabV3 model, these classes correspond to:
        # 0-background, 1-aeroplane, ..., 15-person, ..., etc.
        # See the full list at torchvision documentation for the model
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
            'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
            'tv/monitor', 'grass', 'ground', 'road', 'floor'
        ]

        # Default target class is floor (index depends on model)
        # For COCO+VOC labels, floor is usually around index 3-5
        # For ADE20K dataset, floor is index 3
        # For custom models, this may vary
        self.target_class_id = target_class_id if target_class_id is not None else 3
        print(f"Targeting class ID: {self.target_class_id}")

        # Image transformation pipeline
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _resize_if_needed(self, image):
        """Resize the image if it exceeds the maximum size."""
        if self.max_image_size is None:
            return image

        h, w = image.shape[:2]
        if h > self.max_image_size or w > self.max_image_size:
            # Calculate new dimensions while preserving aspect ratio
            if h > w:
                new_h = self.max_image_size
                new_w = int(w * new_h / h)
            else:
                new_w = self.max_image_size
                new_h = int(h * new_w / w)

            print(f"Resizing image from {w}x{h} to {new_w}x{new_h} to fit in memory")
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def process_image(self, image):
        """
        Process an image to segment and remove floor regions.

        Args:
            image: RGB image as numpy array (H, W, C)

        Returns:
            Processed image with floor removed and segmentation mask
        """
        # Resize image if needed
        original_image = image.copy()
        image = self._resize_if_needed(image)
        original_size = original_image.shape[:2]
        current_size = image.shape[:2]

        # Convert to RGB if in BGR format (OpenCV default)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        try:
            # Prepare image for the model
            input_tensor = self.transform(rgb_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Disable gradient calculation for inference
            with torch.no_grad():
                output = self.model(input_batch)['out'][0]

            # Get segmentation map
            segmentation = torch.argmax(output, dim=0).cpu().numpy()

            # Debug: Display unique classes found in the image
            if self.debug:
                unique_classes = np.unique(segmentation)
                print(f"Detected classes: {unique_classes}")
                for class_id in unique_classes:
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Unknown ({class_id})"
                    class_pixels = np.sum(segmentation == class_id)
                    percent = 100 * class_pixels / segmentation.size
                    print(f"  Class {class_id} ({class_name}): {percent:.2f}% of image")

            # Create mask where target class is False (to be removed)
            target_mask = segmentation != self.target_class_id

            # If no target class pixels were found, try a different approach
            if np.all(target_mask) and self.debug:
                print(f"Warning: No pixels of class {self.target_class_id} found in the image.")

                # Try to find a class that might be the floor/ground
                for potential_id in [3, 4, 5, 9, 11, 22, 23, 24]:
                    if potential_id in np.unique(segmentation):
                        print(f"Suggestion: Try using class {potential_id} instead (found in image)")

            # If the image was resized, resize the mask back to the original size
            if original_size != current_size:
                target_mask = cv2.resize(
                    target_mask.astype(np.uint8),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Create a color-coded segmentation image for debugging
            if self.debug:
                # Create a colorful visualization of all segmentation classes
                segmentation_vis = np.zeros((*original_size, 3), dtype=np.uint8)
                for class_id in np.unique(segmentation):
                    # Generate a random color for each class
                    color = np.random.randint(0, 255, 3).tolist()
                    mask = cv2.resize(
                        (segmentation == class_id).astype(np.uint8),
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    segmentation_vis[mask == 1] = color

                # Save this visualization
                cv2.imwrite("segmentation_classes.jpg", segmentation_vis)
                print("Saved class visualization to segmentation_classes.jpg")

            # Convert mask to 3 channels for multiplication with RGB image
            target_mask_3ch = np.stack([target_mask] * 3, axis=2)
            if original_size != current_size:
                target_mask_3ch = np.stack([cv2.resize(
                    target_mask.astype(np.uint8),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )] * 3, axis=2).astype(bool)

            # Remove floor from image by setting floor pixels to black
            result_image = original_image.copy()
            result_image[~target_mask_3ch] = 0

            return result_image, target_mask

        except torch.cuda.OutOfMemoryError as e:
            if self.device.type == 'cuda':
                print("CUDA out of memory error. Try using --cpu option or --max-size to reduce image dimensions.")
                # Clean up GPU memory
                torch.cuda.empty_cache()
                raise
            else:
                # This shouldn't happen if already on CPU
                raise

    def visualize_segmentation(self, image, mask):
        """
        Create a visualization of the segmentation result.

        Args:
            image: Original RGB image
            mask: Segmentation mask

        Returns:
            Visualization image
        """
        # Create a copy of the original image
        vis_image = image.copy()

        # Create a red overlay for floor regions
        overlay = np.zeros_like(image)

        # Create boolean mask for floor regions (inverted because floor_mask is True where floor is NOT present)
        floor_regions = ~mask

        # Set floor regions to red in the overlay
        overlay[floor_regions, 2] = 255  # Red channel

        # Blend the original image and overlay
        alpha = 0.5
        vis_image = cv2.addWeighted(vis_image, 1.0, overlay, alpha, 0)

        return vis_image


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Floor Segmentation')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, help='Path to save output image')
    parser.add_argument('--show', action='store_true', help='Display the results')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing (slower but uses less memory)')
    parser.add_argument('--max-size', type=int, default=1024, help='Maximum image dimension for processing')
    parser.add_argument('--class-id', type=int, help='Class ID to remove (default is 3 for floor)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return

    try:
        # Initialize the segmentation model with updated parameters
        segmentor = FloorSegmentation(
            use_cpu=args.cpu,
            max_image_size=args.max_size,
            target_class_id=args.class_id,
            debug=args.debug
        )

        # Load image
        print(f"Loading image from {args.input}")
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not load image from {args.input}")
            return

        # Process image
        print("Processing image...")
        result_image, floor_mask = segmentor.process_image(image)

        # Create visualization with floor highlighted
        vis_image = segmentor.visualize_segmentation(image, floor_mask)

        # Save output images
        output_dir = os.path.dirname(args.output) if args.output else '.'
        if args.output:
            # Make sure directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Get base output name without extension
            output_base = os.path.splitext(args.output)[0]

            # Save the floor-removed image
            cv2.imwrite(args.output, result_image)
            print(f"Saved floor-removed image to {args.output}")

            # Save visualization image
            vis_path = f"{output_base}_visualization.jpg"
            cv2.imwrite(vis_path, vis_image)
            print(f"Saved visualization image to {vis_path}")

        # Display results if requested
        if args.show:
            cv2.imshow('Original Image', image)
            cv2.imshow('Segmented Image (Floor Removed)', result_image)
            cv2.imshow('Visualization (Floor Highlighted)', vis_image)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except torch.cuda.OutOfMemoryError:
        print("Error: Not enough GPU memory. Please try again with the --cpu option or reduce image size with --max-size.")
        return
    except Exception as e:
        print(f"Error processing image: {e}")
        return


if __name__ == "__main__":
    main()
