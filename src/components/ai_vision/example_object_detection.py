#!/usr/bin/env python3
# src/components/ai_vision/example_object_detection.py

"""
Example usage of YOLOv8 Object Detection Module

This script demonstrates how to use the object detection module for:
1. Single frame detection
2. Batch image processing
3. Video stream detection with OpenCV

Requirements:
    pip install ultralytics torch torchvision opencv-python numpy
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import time

# Import our object detection module
from object_detection import (
    ObjectDetector, 
    VideoStreamDetector, 
    create_detector,
    detect_objects,
    detect_objects_batch
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a sample image with some geometric shapes for testing"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colorful shapes
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(image, (400, 200), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(image, (300, 300), (500, 400), (0, 0, 255), -1)  # Red rectangle
    
    # Add some text
    cv2.putText(image, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image


async def example_single_frame_detection():
    """Example of single frame object detection"""
    logger.info("=== Single Frame Detection Example ===")
    
    try:
        # Create detector
        detector = await create_detector(model_name="yolov8n", device="auto")
        
        # Create or load a test image
        test_image = create_sample_image()
        
        # Detect objects
        start_time = time.time()
        detections = detector.detect_objects(test_image, conf_threshold=0.25)
        processing_time = time.time() - start_time
        
        # Print results
        logger.info(f"Processing time: {processing_time:.3f} seconds")
        logger.info(f"Found {len(detections)} objects:")
        
        for i, detection in enumerate(detections):
            logger.info(f"  {i+1}. {detection.class_name} - "
                       f"Confidence: {detection.confidence:.3f} - "
                       f"BBox: {detection.bbox}")
        
        # Save result image with bounding boxes
        result_image = test_image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, f"{detection.class_name}: {detection.confidence:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite("single_frame_result.jpg", result_image)
        logger.info("Result saved as 'single_frame_result.jpg'")
        
    except Exception as e:
        logger.error(f"Single frame detection failed: {e}")


async def example_batch_detection():
    """Example of batch image processing"""
    logger.info("=== Batch Detection Example ===")
    
    try:
        # Create detector
        detector = await create_detector(model_name="yolov8n", device="auto")
        
        # Create multiple test images
        batch_images = []
        for i in range(3):
            img = create_sample_image()
            # Add variation to each image
            if i == 1:
                cv2.circle(img, (200, 300), 30, (255, 255, 0), -1)  # Yellow circle
            elif i == 2:
                cv2.rectangle(img, (400, 100), (550, 150), (255, 0, 255), -1)  # Magenta rectangle
            batch_images.append(img)
        
        # Process batch
        start_time = time.time()
        batch_result = detector.detect_objects_batch(batch_images, conf_threshold=0.25)
        
        # Print results
        logger.info(f"Batch processing time: {batch_result.processing_time:.3f} seconds")
        logger.info(f"Average time per image: {batch_result.processing_time/batch_result.batch_size:.3f} seconds")
        
        for i, detections in enumerate(batch_result.detections):
            logger.info(f"Image {i+1}: Found {len(detections)} objects")
            for j, detection in enumerate(detections):
                logger.info(f"  {j+1}. {detection.class_name} - "
                           f"Confidence: {detection.confidence:.3f}")
        
        # Save batch results
        for i, (img, detections) in enumerate(zip(batch_images, batch_result.detections)):
            result_image = img.copy()
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_image, f"{detection.class_name}: {detection.confidence:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(f"batch_result_{i+1}.jpg", result_image)
        
        logger.info("Batch results saved as 'batch_result_*.jpg'")
        
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")


async def example_video_stream_detection():
    """Example of video stream detection (requires webcam or video file)"""
    logger.info("=== Video Stream Detection Example ===")
    
    try:
        # Create detector
        detector = await create_detector(model_name="yolov8n", device="auto")
        
        # Create video stream detector (0 = default webcam)
        # You can also use a video file path instead of 0
        stream_detector = VideoStreamDetector(detector, source=0)
        
        # Start stream
        if not stream_detector.start_stream():
            logger.error("Failed to start video stream. Skipping video stream example.")
            return
        
        logger.info("Starting video stream detection. Press 'q' to quit.")
        
        frame_count = 0
        total_detections = 0
        
        # Process stream frames
        for frame, detections in stream_detector.detect_stream(conf_threshold=0.25):
            frame_count += 1
            total_detections += len(detections)
            
            # Draw bounding boxes
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detection.class_name}: {detection.confidence:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}, Objects: {len(detections)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("YOLOv8 Object Detection", frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Stop after 100 frames for demo purposes
            if frame_count >= 100:
                logger.info("Reached frame limit (100 frames). Stopping...")
                break
        
        # Cleanup
        cv2.destroyAllWindows()
        stream_detector.stop_stream()
        
        logger.info(f"Processed {frame_count} frames with {total_detections} total detections")
        logger.info(f"Average detections per frame: {total_detections/frame_count:.2f}")
        
    except Exception as e:
        logger.error(f"Video stream detection failed: {e}")


def example_convenience_functions():
    """Example using convenience functions with pre-loaded model"""
    logger.info("=== Convenience Functions Example ===")
    
    try:
        # Note: This example shows the function signatures, but requires a pre-loaded model
        # In practice, you would use the ObjectDetector class as shown in other examples
        
        test_image = create_sample_image()
        
        # These functions require a pre-loaded YOLOv8 model
        # model = ... # Your pre-loaded YOLO model
        # detections = detect_objects(test_image, model, conf_threshold=0.25)
        # batch_result = detect_objects_batch([test_image], model, conf_threshold=0.25)
        
        logger.info("Convenience functions are available when you have a pre-loaded model.")
        logger.info("Use ObjectDetector class for complete functionality.")
        
    except Exception as e:
        logger.error(f"Convenience functions example failed: {e}")


async def run_all_examples():
    """Run all examples"""
    logger.info("Starting YOLOv8 Object Detection Examples")
    logger.info("="*50)
    
    # Run examples
    await example_single_frame_detection()
    print()
    
    await example_batch_detection()
    print()
    
    example_convenience_functions()
    print()
    
    # Uncomment to test video stream (requires webcam)
    # await example_video_stream_detection()
    
    logger.info("All examples completed!")


def test_installation():
    """Test if required packages are installed"""
    logger.info("Testing installation...")
    
    try:
        import ultralytics
        logger.info("✓ Ultralytics installed")
    except ImportError:
        logger.error("✗ Ultralytics not installed. Run: pip install ultralytics")
        return False
    
    try:
        import torch
        logger.info("✓ PyTorch installed")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        logger.error("✗ PyTorch not installed. Run: pip install torch torchvision")
        return False
    
    try:
        import cv2
        logger.info("✓ OpenCV installed")
    except ImportError:
        logger.error("✗ OpenCV not installed. Run: pip install opencv-python")
        return False
    
    try:
        import numpy
        logger.info("✓ NumPy installed")
    except ImportError:
        logger.error("✗ NumPy not installed. Run: pip install numpy")
        return False
    
    logger.info("All required packages are installed!")
    return True


if __name__ == "__main__":
    # Test installation first
    if not test_installation():
        logger.error("Please install missing packages before running examples.")
        exit(1)
    
    # Run examples
    asyncio.run(run_all_examples())
