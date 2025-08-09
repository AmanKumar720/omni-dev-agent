#!/usr/bin/env python3
# src/components/ai_vision/example_face_recognition.py

"""
Example usage of the Face Recognition module

This script demonstrates:
1. Face detection in images
2. Identity enrollment with multiple images  
3. Face recognition on new images
4. Database persistence
5. API usage examples

Run with: python example_face_recognition.py
"""

import os
import cv2
import numpy as np
import requests
import base64
import json
from typing import List, Optional
import time

from face_recognition import (
    create_face_recognizer,
    FaceRecognizer,
    recognize_faces,
    enroll_person
)
# from face_api import create_api  # Skip API demo if Flask not available


def create_sample_faces() -> List[np.ndarray]:
    """Create sample face images for testing (synthetic data)"""
    print("Creating sample face images...")
    
    face_images = []
    
    # Create simple synthetic "faces" for demo
    for i in range(3):
        # Create a 200x200 image with different patterns
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Add some pattern to simulate a face
        center_x, center_y = 100, 100
        
        # Face outline (circle)
        cv2.circle(img, (center_x, center_y), 80, (100, 100, 100), 2)
        
        # Eyes
        cv2.circle(img, (center_x - 25, center_y - 20), 8, (255, 255, 255), -1)
        cv2.circle(img, (center_x + 25, center_y - 20), 8, (255, 255, 255), -1)
        
        # Nose
        cv2.line(img, (center_x, center_y - 5), (center_x, center_y + 10), (150, 150, 150), 2)
        
        # Mouth
        cv2.ellipse(img, (center_x, center_y + 25), (20, 10), 0, 0, 180, (200, 200, 200), 2)
        
        # Add some variation for each image
        cv2.circle(img, (center_x + i*10 - 10, center_y + i*5 - 5), 5, (50 + i*50, 50 + i*50, 100), -1)
        
        face_images.append(img)
    
    print(f"Created {len(face_images)} sample face images")
    return face_images


def save_sample_image(image: np.ndarray, filename: str) -> str:
    """Save image to file and return path"""
    cv2.imwrite(filename, image)
    return filename


def test_basic_face_recognition():
    """Test basic face recognition functionality"""
    print("\n=== Testing Basic Face Recognition ===")
    
    # Create face recognizer
    recognizer = create_face_recognizer(
        db_path="test_faces.db",
        detection_threshold=0.3,  # Lower threshold for our simple synthetic faces
        recognition_threshold=0.5
    )
    
    # Create sample faces for "John"
    john_faces = create_sample_faces()
    
    print("Enrolling 'John Doe'...")
    identity_id = recognizer.enroll_identity(
        name="John Doe",
        face_images=john_faces,
        metadata={"department": "Engineering", "employee_id": "EMP001"}
    )
    
    if identity_id:
        print(f"✓ Successfully enrolled John Doe with ID: {identity_id}")
    else:
        print("✗ Failed to enroll John Doe")
        return
    
    # Create sample faces for "Jane"
    jane_faces = []
    for i in range(2):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        center_x, center_y = 100, 100
        
        # Different pattern for Jane
        cv2.rectangle(img, (60, 60), (140, 140), (80, 80, 80), 2)
        cv2.circle(img, (center_x - 20, center_y - 15), 6, (255, 255, 255), -1)
        cv2.circle(img, (center_x + 20, center_y - 15), 6, (255, 255, 255), -1)
        cv2.rectangle(img, (center_x - 15, center_y + 15), (center_x + 15, center_y + 25), (180, 180, 180), -1)
        
        jane_faces.append(img)
    
    print("Enrolling 'Jane Smith'...")
    jane_id = recognizer.enroll_identity(
        name="Jane Smith", 
        face_images=jane_faces,
        metadata={"department": "Marketing", "employee_id": "EMP002"}
    )
    
    if jane_id:
        print(f"✓ Successfully enrolled Jane Smith with ID: {jane_id}")
    else:
        print("✗ Failed to enroll Jane Smith")
    
    # Test recognition
    print("\nTesting recognition on John's first image...")
    test_image = john_faces[0]
    results = recognizer.recognize_faces(test_image)
    
    print(f"Recognition results: {len(results)} faces found")
    for result in results:
        print(f"  - Name: {result.name}")
        print(f"  - Confidence: {result.confidence:.3f}")
        print(f"  - Face ID: {result.face_id}")
        print(f"  - Bounding Box: ({result.detection.bbox.x}, {result.detection.bbox.y}, "
              f"{result.detection.bbox.width}, {result.detection.bbox.height})")
    
    # List all identities
    print("\nAll enrolled identities:")
    identities = recognizer.list_identities()
    for identity in identities:
        print(f"  - {identity['name']} (ID: {identity['id'][:8]}...)")
        print(f"    Enrollments: {identity['num_embeddings']}")
        print(f"    Created: {identity['created_at']}")
    
    return recognizer


def test_convenience_functions():
    """Test convenience functions"""
    print("\n=== Testing Convenience Functions ===")
    
    # Create sample images and save them
    sample_faces = create_sample_faces()
    image_paths = []
    
    for i, face in enumerate(sample_faces):
        path = f"temp_face_{i}.jpg"
        save_sample_image(face, path)
        image_paths.append(path)
    
    print("Testing enroll_person function...")
    identity_id = enroll_person(
        name="Bob Wilson",
        face_images=image_paths,  # Using file paths
        metadata={"department": "Sales", "employee_id": "EMP003"}
    )
    
    if identity_id:
        print(f"✓ Successfully enrolled Bob Wilson with ID: {identity_id}")
    else:
        print("✗ Failed to enroll Bob Wilson")
    
    print("Testing recognize_faces function...")
    recognition_results = recognize_faces(image_paths[0])  # Using file path
    
    print(f"Recognition results: {len(recognition_results)} faces found")
    for result in recognition_results:
        print(f"  - Name: {result.name}")
        print(f"  - Confidence: {result.confidence:.3f}")
    
    # Clean up temporary files
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)
    
    print("✓ Cleaned up temporary files")


def test_api_server():
    """Test the REST API server"""
    print("\n=== Testing Face Recognition API ===")
    
    # Start API server in background (for demo, we'll just show how to use it)
    print("To test the API server, run in another terminal:")
    print("python face_api.py --host 127.0.0.1 --port 5000")
    print()
    
    # Example API calls (would work if server is running)
    api_examples = {
        "health_check": {
            "method": "GET",
            "url": "http://127.0.0.1:5000/api/health",
            "description": "Check API health status"
        },
        "enroll_identity": {
            "method": "POST", 
            "url": "http://127.0.0.1:5000/api/enroll",
            "description": "Enroll new identity with JSON payload",
            "payload": {
                "name": "Alice Johnson",
                "images": ["base64_encoded_image1", "base64_encoded_image2"],
                "metadata": {"department": "HR", "employee_id": "EMP004"}
            }
        },
        "recognize_face": {
            "method": "POST",
            "url": "http://127.0.0.1:5000/api/recognize", 
            "description": "Recognize faces in image",
            "payload": {
                "image": "base64_encoded_image"
            }
        },
        "list_identities": {
            "method": "GET",
            "url": "http://127.0.0.1:5000/api/identities",
            "description": "List all enrolled identities"
        },
        "detect_only": {
            "method": "POST",
            "url": "http://127.0.0.1:5000/api/detect",
            "description": "Detect faces without recognition",
            "payload": {
                "image": "base64_encoded_image"
            }
        }
    }
    
    print("Available API endpoints:")
    for name, info in api_examples.items():
        print(f"\n{name}:")
        print(f"  {info['method']} {info['url']}")
        print(f"  {info['description']}")
        if 'payload' in info:
            print(f"  Payload: {json.dumps(info['payload'], indent=2)}")


def demo_api_client():
    """Demonstrate API client usage (assumes server is running)"""
    print("\n=== API Client Demo (requires running server) ===")
    
    base_url = "http://127.0.0.1:5000"
    
    try:
        # Health check
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ API server is running")
            print(f"  Response: {response.json()}")
            
            # Create a sample image and encode it
            sample_image = create_sample_faces()[0]
            _, buffer = cv2.imencode('.jpg', sample_image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Test face detection
            detect_response = requests.post(
                f"{base_url}/api/detect",
                json={"image": image_b64},
                timeout=10
            )
            
            if detect_response.status_code == 200:
                detect_data = detect_response.json()
                print(f"✓ Face detection successful: {detect_data['count']} faces detected")
            else:
                print(f"✗ Face detection failed: {detect_response.status_code}")
        
        else:
            print(f"✗ API server health check failed: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"✗ API server not accessible: {e}")
        print("  Start the server with: python face_api.py")


def cleanup_test_files():
    """Clean up test files"""
    print("\n=== Cleaning Up Test Files ===")
    
    test_files = ["test_faces.db", "faces.db"]
    
    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"✓ Removed {filename}")
    
    # Remove temp uploads directory if exists
    if os.path.exists("temp_uploads"):
        import shutil
        shutil.rmtree("temp_uploads")
        print("✓ Removed temp_uploads directory")


def main():
    """Main demo function"""
    print("Face Recognition Module Demo")
    print("=" * 50)
    
    try:
        # Test basic functionality
        recognizer = test_basic_face_recognition()
        
        # Test convenience functions
        test_convenience_functions()
        
        # Show API examples
        test_api_server()
        
        # Try to connect to API (if running)
        demo_api_client()
        
        print("\n=== Demo Summary ===")
        print("✓ Face detection using OpenCV")
        print("✓ Face embedding extraction (demo implementation)")
        print("✓ Identity enrollment with multiple images")
        print("✓ Face recognition with cosine similarity")
        print("✓ SQLite database persistence")
        print("✓ REST API endpoints")
        print("✓ Convenience functions for easy usage")
        
        print("\n=== Production Notes ===")
        print("- Replace simple feature extractor with real FaceNet/InsightFace model")
        print("- Use proper DNN face detector (download models from OpenCV)")
        print("- Add face alignment and preprocessing")
        print("- Implement batch processing for better performance")
        print("- Add authentication and rate limiting to API")
        print("- Use proper database with connection pooling")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cleanup_test_files()


if __name__ == "__main__":
    main()
