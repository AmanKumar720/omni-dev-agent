#!/usr/bin/env python3
"""
Test script for the Vision Processing Web Interface

This script tests the basic functionality of the web interface
including API endpoints and template rendering.
"""

import requests
import json
import base64
import cv2
import numpy as np
from datetime import datetime
import os

class VisionInterfaceTest:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name, passed, message=""):
        """Log test results"""
        status = "PASS" if passed else "FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = f"[{timestamp}] {test_name}: {status}"
        if message:
            result += f" - {message}"
        print(result)
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'message': message,
            'timestamp': timestamp
        })
    
    def create_test_image(self):
        """Create a simple test image"""
        # Create a 300x200 test image with text
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img.fill(255)  # White background
        
        # Add some text
        cv2.putText(img, "TEST IMAGE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Hello World!", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add a rectangle (simulate an object)
        cv2.rectangle(img, (20, 20), (100, 80), (0, 255, 0), 2)
        
        return img
    
    def image_to_base64(self, image):
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def test_server_connection(self):
        """Test if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                self.log_test("Server Connection", True, f"Status: {response.status_code}")
                return True
            else:
                self.log_test("Server Connection", False, f"Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("Server Connection", False, f"Error: {str(e)}")
            return False
    
    def test_ui_endpoint(self):
        """Test the main UI endpoint"""
        try:
            response = requests.get(f"{self.base_url}/ui", timeout=5)
            if response.status_code == 200 and "Vision Interface" in response.text:
                self.log_test("UI Endpoint", True, "Main interface accessible")
                return True
            else:
                self.log_test("UI Endpoint", False, f"Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("UI Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_demo_endpoint(self):
        """Test the demo page endpoint"""
        try:
            response = requests.get(f"{self.base_url}/demo", timeout=5)
            if response.status_code == 200 and "Demo" in response.text:
                self.log_test("Demo Endpoint", True, "Demo page accessible")
                return True
            else:
                self.log_test("Demo Endpoint", False, f"Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("Demo Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_camera_list_api(self):
        """Test camera list API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/camera/list", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Camera List API", True, f"Found {data.get('count', 0)} cameras")
                return True
            else:
                self.log_test("Camera List API", False, f"Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("Camera List API", False, f"Error: {str(e)}")
            return False
        except json.JSONDecodeError:
            self.log_test("Camera List API", False, "Invalid JSON response")
            return False
    
    def test_camera_discover_api(self):
        """Test camera discover API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/camera/discover", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Camera Discovery API", True, f"Discovered {data.get('count', 0)} cameras")
                return True
            else:
                self.log_test("Camera Discovery API", False, f"Status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("Camera Discovery API", False, f"Error: {str(e)}")
            return False
        except json.JSONDecodeError:
            self.log_test("Camera Discovery API", False, "Invalid JSON response")
            return False
    
    def test_object_detection_api(self):
        """Test object detection API with test image"""
        try:
            test_image = self.create_test_image()
            image_base64 = self.image_to_base64(test_image)
            
            payload = {
                "image": image_base64,
                "confidence": 0.5
            }
            
            response = requests.post(
                f"{self.base_url}/vision/detect", 
                json=payload, 
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                detection_count = len(data.get('detections', []))
                self.log_test("Object Detection API", True, f"Detected {detection_count} objects")
                return True
            else:
                self.log_test("Object Detection API", False, f"Status: {response.status_code}, Response: {response.text[:100]}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Object Detection API", False, f"Error: {str(e)}")
            return False
        except json.JSONDecodeError:
            self.log_test("Object Detection API", False, "Invalid JSON response")
            return False
        except Exception as e:
            self.log_test("Object Detection API", False, f"Unexpected error: {str(e)}")
            return False
    
    def test_ocr_api(self):
        """Test OCR API with test image containing text"""
        try:
            test_image = self.create_test_image()
            image_base64 = self.image_to_base64(test_image)
            
            payload = {
                "image": image_base64,
                "language": "eng"
            }
            
            response = requests.post(
                f"{self.base_url}/vision/ocr", 
                json=payload, 
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                extracted_text = data.get('text', '').strip()
                has_text = len(extracted_text) > 0
                self.log_test("OCR API", has_text, f"Extracted: '{extracted_text[:50]}...' " if has_text else "No text detected")
                return True
            else:
                self.log_test("OCR API", False, f"Status: {response.status_code}, Response: {response.text[:100]}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("OCR API", False, f"Error: {str(e)}")
            return False
        except json.JSONDecodeError:
            self.log_test("OCR API", False, "Invalid JSON response")
            return False
        except Exception as e:
            self.log_test("OCR API", False, f"Unexpected error: {str(e)}")
            return False
    
    def test_face_recognition_api(self):
        """Test face recognition API"""
        try:
            test_image = self.create_test_image()
            image_base64 = self.image_to_base64(test_image)
            
            payload = {
                "image": image_base64
            }
            
            response = requests.post(
                f"{self.base_url}/vision/face/identify", 
                json=payload, 
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                face_count = len(data.get('faces', []))
                self.log_test("Face Recognition API", True, f"Detected {face_count} faces")
                return True
            else:
                self.log_test("Face Recognition API", False, f"Status: {response.status_code}, Response: {response.text[:100]}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Face Recognition API", False, f"Error: {str(e)}")
            return False
        except json.JSONDecodeError:
            self.log_test("Face Recognition API", False, "Invalid JSON response")
            return False
        except Exception as e:
            self.log_test("Face Recognition API", False, f"Unexpected error: {str(e)}")
            return False
    
    def test_classification_api(self):
        """Test image classification API"""
        try:
            test_image = self.create_test_image()
            image_base64 = self.image_to_base64(test_image)
            
            payload = {
                "image": image_base64,
                "top_k": 5
            }
            
            response = requests.post(
                f"{self.base_url}/vision/classify", 
                json=payload, 
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                classifications = data.get('classifications', [])
                self.log_test("Classification API", True, f"Got {len(classifications)} classifications")
                return True
            else:
                self.log_test("Classification API", False, f"Status: {response.status_code}, Response: {response.text[:100]}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.log_test("Classification API", False, f"Error: {str(e)}")
            return False
        except json.JSONDecodeError:
            self.log_test("Classification API", False, "Invalid JSON response")
            return False
        except Exception as e:
            self.log_test("Classification API", False, f"Unexpected error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests and return summary"""
        print("=" * 60)
        print("VISION PROCESSING WEB INTERFACE TEST SUITE")
        print("=" * 60)
        print()
        
        tests = [
            self.test_server_connection,
            self.test_ui_endpoint,
            self.test_demo_endpoint,
            self.test_camera_list_api,
            self.test_camera_discover_api,
            self.test_object_detection_api,
            self.test_ocr_api,
            self.test_face_recognition_api,
            self.test_classification_api,
        ]
        
        for test in tests:
            test()
        
        # Summary
        print()
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print()
            print("FAILED TESTS:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['name']}: {result['message']}")
        
        print()
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Web interface is working correctly.")
        else:
            print(f"⚠️  {failed_tests} test(s) failed. Check the error messages above.")
        
        return failed_tests == 0

def main():
    """Main test function"""
    import sys
    
    # Check if server URL is provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    print(f"Testing Vision Interface at: {base_url}")
    print("Make sure the Flask server is running!")
    print()
    
    tester = VisionInterfaceTest(base_url)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
