# src/components/ai_vision/face_api.py

import os
import cv2
import numpy as np
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import uuid

# from face_recognition import (
#     FaceRecognizer, 
#     create_face_recognizer,
#     FaceRecognitionResult
# )


# class FaceRecognitionAPI:
#     """REST API for face enrollment and recognition"""
    
#     def __init__(self, 
#                  db_path: str = "faces.db",
#                  detection_threshold: float = 0.5,
#                  recognition_threshold: float = 0.7,
#                  upload_folder: str = "temp_uploads"):
#         """
#         Initialize Face Recognition API
#         
#         Args:
#             db_path: Path to faces database
#             detection_threshold: Minimum confidence for face detection  
#             recognition_threshold: Minimum similarity for face recognition
#             upload_folder: Folder for temporary file uploads
#         """
#         self.app = Flask(__name__)
#         self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
#         
#         # Create upload folder if it doesn't exist
#         self.upload_folder = upload_folder
#         os.makedirs(upload_folder, exist_ok=True)
#         
#         # Initialize face recognizer
#         self.recognizer = create_face_recognizer(
#             db_path=db_path,
#             detection_threshold=detection_threshold,
#             recognition_threshold=recognition_threshold
#         )
#         
#         self.logger = logging.getLogger(f"{__name__}.FaceRecognitionAPI")
#         
#         # Register API routes
#         self._register_routes()
    
#     def _register_routes(self):
#         """Register all API routes"""
        
#         @self.app.route('/api/health', methods=['GET'])
#         def health_check():
#             """Health check endpoint"""
#             return jsonify({
#                 'status': 'healthy',
#                 'timestamp': datetime.now().isoformat(),
#                 'version': '1.0.0'
#             })
        
#         @self.app.route('/api/enroll', methods=['POST'])
#         def enroll_identity():
#             """
#             Enroll a new identity
#             
#             Expected JSON payload:
#             {
#                 "name": "John Doe",
#                 "images": ["base64_image1", "base64_image2", ...],
#                 "metadata": {"department": "Engineering", "employee_id": "EMP001"}
#             }
#             
#             Or multipart form with:
#             - name: Person's name
#             - files: Multiple image files
#             - metadata: JSON string (optional)
#             """
#             try:
#                 if request.content_type and 'application/json' in request.content_type:
#                     return self._enroll_from_json()
#                 else:
#                     return self._enroll_from_form()
                    
#             except Exception as e:
#                 self.logger.error(f"Enrollment failed: {e}")
#                 return jsonify({'error': str(e)}), 500
        
#         @self.app.route('/api/recognize', methods=['POST'])
#         def recognize_faces():
#             """
#             Recognize faces in image
#             
#             Expected JSON payload:
#             {
#                 "image": "base64_encoded_image"
#             }
#             
#             Or multipart form with image file
#             """
#             try:
#                 if request.content_type and 'application/json' in request.content_type:
#                     return self._recognize_from_json()
#                 else:
#                     return self._recognize_from_form()
                    
#             except Exception as e:
#                 self.logger.error(f"Recognition failed: {e}")
#                 return jsonify({'error': str(e)}), 500
        
#         @self.app.route('/api/identities', methods=['GET'])
#         def list_identities():
#             """List all enrolled identities"""
#             try:
#                 identities = self.recognizer.list_identities()
#                 return jsonify({
#                     'identities': identities,
#                     'total': len(identities)
#                 })
#             except Exception as e:
#                 self.logger.error(f"Failed to list identities: {e}")
#                 return jsonify({'error': str(e)}), 500
        
#         @self.app.route('/api/identities/<identity_id>', methods=['GET'])
#         def get_identity(identity_id):
#             """Get identity information by ID"""
#             try:
#                 identity_info = self.recognizer.get_identity_info(identity_id)
#                 if identity_info:
#                     return jsonify(identity_info)
#                 else:
#                     return jsonify({'error': 'Identity not found'}), 404
#             except Exception as e:
#                 self.logger.error(f"Failed to get identity {identity_id}: {e}")
#                 return jsonify({'error': str(e)}), 500
        
#         @self.app.route('/api/identities/<identity_id>', methods=['DELETE'])
#         def delete_identity(identity_id):
#             """Delete identity by ID"""
#             try:
#                 success = self.recognizer.delete_identity(identity_id)
#                 if success:
#                     return jsonify({'message': 'Identity deleted successfully'})
#                 else:
#                     return jsonify({'error': 'Identity not found'}), 404
#             except Exception as e:
#                 self.logger.error(f"Failed to delete identity {identity_id}: {e}")
#                 return jsonify({'error': str(e)}), 500
        
#         @self.app.route('/api/detect', methods=['POST'])
#         def detect_faces_only():
#             """
#             Detect faces without recognition
#             
#             Expected JSON payload:
#             {
#                 "image": "base64_encoded_image"
#             }
#             
#             Or multipart form with image file
#             """
#             try:
#                 if request.content_type and 'application/json' in request.content_type:
#                     image = self._decode_base64_image(request.json.get('image'))
#                 else:
#                     if 'image' not in request.files:
#                         return jsonify({'error': 'No image provided'}), 400
#                     image = self._read_uploaded_image(request.files['image'])
                
#                 # Detect faces only
#                 faces = self.recognizer.detector.detect_faces(image)
                
#                 # Convert to JSON serializable format
#                 face_data = []
#                 for face in faces:
#                     face_data.append({
#                         'bbox': {
#                             'x': face.bbox.x,
#                             'y': face.bbox.y,
#                             'width': face.bbox.width,
#                             'height': face.bbox.height
#                         },
#                         'confidence': float(face.confidence)
#                     })
                
#                 return jsonify({
#                     'faces': face_data,
#                     'count': len(face_data)
#                 })
                
#             except Exception as e:
#                 self.logger.error(f"Face detection failed: {e}")
#                 return jsonify({'error': str(e)}), 500
    
#     def _enroll_from_json(self):
#         """Handle enrollment from JSON payload"""
#         data = request.json
#         
#         if not data or 'name' not in data or 'images' not in data:
#             return jsonify({'error': 'Name and images are required'}), 400
#         
#         name = data['name']
#         base64_images = data['images']
#         metadata = data.get('metadata', {})
#         
#         if not isinstance(base64_images, list) or len(base64_images) == 0:
#             return jsonify({'error': 'At least one image is required'}), 400
#         
#         # Decode base64 images
#         face_images = []
#         for i, b64_image in enumerate(base64_images):
#             try:
#                 image = self._decode_base64_image(b64_image)
#                 face_images.append(image)
#             except Exception as e:
#                 self.logger.error(f"Failed to decode image {i}: {e}")
#                 return jsonify({'error': f'Failed to decode image {i}'}), 400
#         
#         # Enroll identity
#         identity_id = self.recognizer.enroll_identity(name, face_images, metadata)
#         
#         if identity_id:
#             return jsonify({
#                 'message': 'Identity enrolled successfully',
#                 'identity_id': identity_id,
#                 'name': name,
#                 'num_images': len(face_images)
#             })
#         else:
#             return jsonify({'error': 'Failed to enroll identity'}), 500
    
#     def _enroll_from_form(self):
#         """Handle enrollment from multipart form"""
#         if 'name' not in request.form:
#             return jsonify({'error': 'Name is required'}), 400
#         
#         name = request.form['name']
#         metadata_str = request.form.get('metadata', '{}')
#         
#         try:
#             metadata = json.loads(metadata_str)
#         except json.JSONDecodeError:
#             metadata = {}
#         
#         # Get uploaded images
#         if 'images' not in request.files:
#             return jsonify({'error': 'At least one image is required'}), 400
#         
#         files = request.files.getlist('images')
#         if len(files) == 0:
#             return jsonify({'error': 'At least one image is required'}), 400
#         
#         # Process uploaded images
#         face_images = []
#         for file in files:
#             if file.filename == '':
#                 continue
#             
#             try:
#                 image = self._read_uploaded_image(file)
#                 face_images.append(image)
#             except Exception as e:
#                 self.logger.error(f"Failed to process uploaded image: {e}")
#                 return jsonify({'error': 'Failed to process uploaded image'}), 400
#         
#         if len(face_images) == 0:
#             return jsonify({'error': 'No valid images provided'}), 400
#         
#         # Enroll identity
#         identity_id = self.recognizer.enroll_identity(name, face_images, metadata)
#         
#         if identity_id:
#             return jsonify({
#                 'message': 'Identity enrolled successfully',
#                 'identity_id': identity_id,
#                 'name': name,
#                 'num_images': len(face_images)
#             })
#         else:
#             return jsonify({'error': 'Failed to enroll identity'}), 500
    
#     def _recognize_from_json(self):
#         """Handle recognition from JSON payload"""
#         data = request.json
#         
#         if not data or 'image' not in data:
#             return jsonify({'error': 'Image is required'}), 400
#         
#         # Decode base64 image
#         image = self._decode_base64_image(data['image'])
#         
#         # Perform recognition
#         results = self.recognizer.recognize_faces(image)
#         
#         # Convert to JSON serializable format
#         recognition_data = []
#         for result in results:
#             recognition_data.append({
#                 'face_id': result.face_id,
#                 'name': result.name,
#                 'confidence': float(result.confidence),
#                 'bbox': {
#                     'x': result.detection.bbox.x,
#                     'y': result.detection.bbox.y,
#                     'width': result.detection.bbox.width,
#                     'height': result.detection.bbox.height
#                 },
#                 'detection_confidence': float(result.detection.confidence)
#             })
#         
#         return jsonify({
#             'results': recognition_data,
#             'count': len(recognition_data)
#         })
    
#     def _recognize_from_form(self):
#         """Handle recognition from multipart form"""
#         if 'image' not in request.files:
#             return jsonify({'error': 'Image is required'}), 400
#         
#         file = request.files['image']
#         if file.filename == '':
#             return jsonify({'error': 'No image selected'}), 400
#         
#         # Process uploaded image
#         image = self._read_uploaded_image(file)
#         
#         # Perform recognition
#         results = self.recognizer.recognize_faces(image)
#         
#         # Convert to JSON serializable format
#         recognition_data = []
#         for result in results:
#             recognition_data.append({
#                 'face_id': result.face_id,
#                 'name': result.name,
#                 'confidence': float(result.confidence),
#                 'bbox': {
#                     'x': result.detection.bbox.x,
#                     'y': result.detection.bbox.y,
#                     'width': result.detection.bbox.width,
#                     'height': result.detection.bbox.height
#                 },
#                 'detection_confidence': float(result.detection.confidence)
#             })
#         
#         return jsonify({
#             'results': recognition_data,
#             'count': len(recognition_data)
#         })
    
#     def _decode_base64_image(self, base64_str: str) -> np.ndarray:
#         """Decode base64 string to image array"""
#         if base64_str.startswith('data:image'):
#             # Remove data URL prefix
#             base64_str = base64_str.split(',', 1)[1]
#         
#         # Decode base64
#         image_data = base64.b64decode(base64_str)
#         
#         # Convert to numpy array
#         nparr = np.frombuffer(image_data, np.uint8)
#         
#         # Decode image
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         
#         if image is None:
#             raise ValueError("Invalid image data")
#         
#         return image
    
#     def _read_uploaded_image(self, file) -> np.ndarray:
#         """Read uploaded file as image array"""
#         # Save temporarily
#         filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
#         temp_path = os.path.join(self.upload_folder, filename)
#         
#         try:
#             file.save(temp_path)
#             
#             # Read image
#             image = cv2.imread(temp_path)
#             
#             if image is None:
#                 raise ValueError("Invalid image file")
#             
#             return image
#             
#         finally:
#             # Clean up temporary file
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)
    
#     def run(self, host='127.0.0.1', port=5000, debug=False):
#         """Run the Flask application"""
#         self.logger.info(f"Starting Face Recognition API on {host}:{port}")
#         self.app.run(host=host, port=port, debug=debug)


# def create_api(db_path: str = "faces.db",
#                detection_threshold: float = 0.5,
#                recognition_threshold: float = 0.7,
#                upload_folder: str = "temp_uploads") -> FaceRecognitionAPI:
#     """Create Face Recognition API instance"""
#     return FaceRecognitionAPI(
#         db_path=db_path,
#         detection_threshold=detection_threshold,
#         recognition_threshold=recognition_threshold,
#         upload_folder=upload_folder
#     )


# # CLI entry point
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Face Recognition API Server')
#     parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
#     parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
#     parser.add_argument('--db-path', default='faces.db', help='Path to faces database')
#     parser.add_argument('--detection-threshold', type=float, default=0.5, 
#                        help='Face detection confidence threshold')
#     parser.add_argument('--recognition-threshold', type=float, default=0.7,
#                        help='Face recognition similarity threshold')
#     parser.add_argument('--upload-folder', default='temp_uploads',
#                        help='Temporary upload folder')
#     parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
#     args = parser.parse_args()
    
#     # Set up logging
#     logging.basicConfig(level=logging.INFO)
    
#     # Create and run API
#     api = create_api(
#         db_path=args.db_path,
#         detection_threshold=args.detection_threshold,
#         recognition_threshold=args.recognition_threshold,
#         upload_folder=args.upload_folder
#     )
    
#     api.run(host=args.host, port=args.port, debug=args.debug)