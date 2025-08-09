#!/usr/bin/env python3
"""
Vision Orchestrator Demo

This script demonstrates how to use the new vision task routing functionality
in the Orchestrator class for object detection and computer vision analytics.

Usage:
    python vision_orchestrator_demo.py
"""

import sys
import os
import asyncio
import numpy as np
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.orchestration import Orchestrator
except ImportError:
    print("Error: Could not import Orchestrator. Make sure you're running from the correct directory.")
    sys.exit(1)


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a test image with some basic shapes for object detection testing
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Test image as numpy array
    """
    # Create image with random noise to simulate real data
    image = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    
    # Add some geometric shapes that might trigger detection
    # Blue rectangle
    image[100:200, 100:200] = [255, 100, 100]
    
    # Green circle (approximated as rectangle for simplicity)
    image[300:350, 400:450] = [100, 255, 100] 
    
    # Red rectangle
    image[350:400, 200:300] = [100, 100, 255]
    
    return image


async def demo_object_detection(orchestrator: Orchestrator):
    """
    Demonstrate object detection functionality
    
    Args:
        orchestrator: Orchestrator instance
    """
    print("=== Object Detection Demo ===")
    
    # Create test image
    test_frame = create_test_image()
    print(f"Created test image: {test_frame.shape}")
    
    # Single frame detection
    print("\n1. Single Frame Detection:")
    result = await orchestrator.handle_request(
        type='vision',
        task='object_detection',
        payload={
            'frame': test_frame,
            'conf_threshold': 0.1,  # Lower threshold for demo
            'model_name': 'yolov8n'
        }
    )
    
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Confidence: {result['confidence']}")
        print(f"Detections: {len(result['data'])} objects found")
        for i, detection in enumerate(result['data'][:3]):  # Show first 3
            print(f"  - {detection['class_name']}: {detection['confidence']:.3f}")
    else:
        print(f"Error: {result['error']}")
    
    # Batch detection
    print("\n2. Batch Frame Detection:")
    batch_frames = [create_test_image() for _ in range(3)]
    
    batch_result = await orchestrator.handle_request(
        type='vision',
        task='object_detection',
        payload={
            'batch_frames': batch_frames,
            'conf_threshold': 0.1,
            'model_name': 'yolov8n'
        }
    )
    
    print(f"Status: {batch_result['status']}")
    if batch_result['status'] == 'success':
        batch_data = batch_result['data']
        print(f"Processed {batch_data['batch_size']} frames")
        print(f"Processing time: {batch_data['processing_time']:.3f}s")
        for i, detections in enumerate(batch_data['detections'][:2]):  # Show first 2
            print(f"  Frame {i+1}: {len(detections)} objects")
    else:
        print(f"Error: {batch_result['error']}")


async def demo_vision_analytics(orchestrator: Orchestrator):
    """
    Demonstrate computer vision analytics functionality
    
    Args:
        orchestrator: Orchestrator instance
    """
    print("\n=== Vision Analytics Demo ===")
    
    # Create test image
    test_frame = create_test_image()
    print(f"Created test image for analytics: {test_frame.shape}")
    
    # Analytics processing
    print("\n1. Single Frame Analytics:")
    result = await orchestrator.handle_request(
        type='vision',
        task='analytics',
        payload={
            'input_data': test_frame,
            'analytics_config': {
                'motion_config': {
                    'threshold': 25,
                    'min_area': 500,
                    'blur_size': 21
                },
                'reasoning_config': {
                    'activity_detection': {
                        'enabled': True,
                        'motion_threshold': 0.05
                    },
                    'door_detection': {
                        'enabled': True,
                        'region': {'x': 100, 'y': 50, 'width': 200, 'height': 300},
                        'threshold': 0.3
                    }
                }
            }
        }
    )
    
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Confidence: {result['confidence']}")
        data = result['data']
        print(f"Events generated: {len(data.get('events', []))}")
        
        # Show motion detection results
        motion_result = data.get('motion_result', {})
        print(f"Motion detected: {motion_result.get('motion_detected', False)}")
        print(f"Motion area: {motion_result.get('motion_area', 0)}")
        
        # Show reasoning results
        reasoning_result = data.get('reasoning_result', {})
        print(f"Rules evaluated: {len(reasoning_result.get('rules_evaluated', []))}")
        print(f"Rules triggered: {len(reasoning_result.get('triggered_rules', []))}")
        
    else:
        print(f"Error: {result['error']}")


async def demo_error_handling(orchestrator: Orchestrator):
    """
    Demonstrate error handling for various scenarios
    
    Args:
        orchestrator: Orchestrator instance
    """
    print("\n=== Error Handling Demo ===")
    
    # Test unsupported request type
    print("\n1. Unsupported request type:")
    result = await orchestrator.handle_request(
        type='unsupported_type',
        task='some_task',
        payload={}
    )
    print(f"Status: {result['status']}")
    print(f"Error: {result['error']}")
    
    # Test unsupported vision task
    print("\n2. Unsupported vision task:")
    result = await orchestrator.handle_request(
        type='vision',
        task='unsupported_vision_task',
        payload={}
    )
    print(f"Status: {result['status']}")
    print(f"Error: {result['error']}")
    
    # Test missing required parameters
    print("\n3. Missing required parameters:")
    result = await orchestrator.handle_request(
        type='vision',
        task='object_detection',
        payload={
            'conf_threshold': 0.25
            # Missing frame or batch_frames
        }
    )
    print(f"Status: {result['status']}")
    print(f"Error: {result['error']}")


def demo_response_format():
    """
    Show the standardized response format
    """
    print("\n=== Standardized Response Format ===")
    
    example_success_response = {
        'status': 'success',
        'data': {
            'detections': [
                {
                    'bbox': [100, 150, 200, 250],
                    'confidence': 0.85,
                    'class_id': 0,
                    'class_name': 'person'
                }
            ]
        },
        'confidence': 0.85,
        'metadata': {
            'task_id': 'obj_det_2024-01-01T12:00:00',
            'vision_task': 'object_detection',
            'model_name': 'yolov8n',
            'conf_threshold': 0.25,
            'processing_type': 'single_frame',
            'timestamp': '2024-01-01T12:00:00'
        }
    }
    
    example_error_response = {
        'status': 'error',
        'error': 'Either frame or batch_frames must be provided',
        'data': None,
        'metadata': {
            'vision_task': 'object_detection',
            'timestamp': '2024-01-01T12:00:00'
        }
    }
    
    print("Success Response Format:")
    print(json.dumps(example_success_response, indent=2))
    
    print("\nError Response Format:")
    print(json.dumps(example_error_response, indent=2))


async def main():
    """
    Main demonstration function
    """
    print("Vision Orchestrator Demo")
    print("=" * 50)
    
    try:
        # Create orchestrator
        orchestrator = Orchestrator()
        print("Orchestrator initialized successfully!")
        
        # Show response format
        demo_response_format()
        
        # Run object detection demo
        await demo_object_detection(orchestrator)
        
        # Run analytics demo
        await demo_vision_analytics(orchestrator)
        
        # Show error handling
        await demo_error_handling(orchestrator)
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nTo use in your own code:")
        print("```python")
        print("orchestrator = Orchestrator()")
        print("result = await orchestrator.handle_request(")
        print("    type='vision',")
        print("    task='object_detection',")
        print("    payload={'frame': your_image, 'conf_threshold': 0.25}")
        print(")")
        print("```")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("This might be due to missing dependencies or import issues.")
        print("Make sure all required packages are installed:")
        print("  pip install ultralytics torch torchvision opencv-python numpy")


if __name__ == "__main__":
    # Check if we can import required dependencies
    try:
        import cv2
        import torch
        import ultralytics
        print("All dependencies available - running full demo")
        asyncio.run(main())
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install ultralytics torch torchvision opencv-python numpy")
        print("\nRunning limited demo without actual model execution...")
        
        # Run a limited demo showing the API structure
        async def limited_demo():
            print("\n=== Limited Demo (API Structure Only) ===")
            orchestrator = Orchestrator()
            
            # This will show the error handling for missing dependencies
            result = await orchestrator.handle_request(
                type='vision',
                task='object_detection',
                payload={
                    'frame': np.zeros((480, 640, 3), dtype=np.uint8),
                    'conf_threshold': 0.25
                }
            )
            
            print("Example API call:")
            print(json.dumps(result, indent=2, default=str))
        
        asyncio.run(limited_demo())
