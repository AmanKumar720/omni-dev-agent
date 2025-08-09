import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import asyncio
from datetime import datetime
import json
import threading
from typing import Dict, Any, Optional

# Import vision components
from src.components.ai_vision.ocr import OCREngine, create_ocr_engine
from src.components.ai_vision.object_detection import ObjectDetector, detect_objects
# from src.components.ai_vision.face_recognition import FaceRecognizer, create_face_recognizer
from src.components.ai_vision.image_classification import ImageClassifier, classify_image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize vision components
ocr_processor = create_ocr_engine()
object_detector = ObjectDetector()
# face_recognizer = create_face_recognizer()
image_classifier = ImageClassifier()

# Camera integration
from components.camera_integration.camera_manager import CameraManager, CameraConfig, CameraCredentials, CameraCapabilities
camera_manager = CameraManager(storage_path="camera_data")

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',', 1)[1]
    
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Invalid image data")
    
    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"

@app.route("/")
def hello_world():
    return "Omni-Dev Agent Vision API - Ready"

@app.route("/ui")
def vision_interface():
    """Serve the vision interface web UI"""
    return render_template('index.html')

@app.route("/demo")
def demo_page():
    """Serve the demo page"""
    return render_template('demo.html')

# Vision API Endpoints

@app.route('/vision/detect', methods=['POST'])
def vision_detect():
    """Object detection endpoint"""
    try:
        # Handle both JSON and form data
        if request.content_type and 'application/json' in request.content_type:
            data = request.json
            if 'image' not in data:
                return jsonify({'error': 'Image data required'}), 400
            
            image = decode_base64_image(data['image'])
            confidence_threshold = data.get('confidence', 0.25)
            
        else:
            if 'image' not in request.files:
                return jsonify({'error': 'Image file required'}), 400
            
            file = request.files['image']
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            confidence_threshold = float(request.form.get('confidence', 0.25))
        
        # Perform object detection
        detections = detect_objects(image, confidence_threshold=confidence_threshold)
        
        # Convert results to JSON
        result = {
            'timestamp': datetime.now().isoformat(),
            'detections': [
                {
                    'class_name': det.class_name,
                    'confidence': float(det.confidence),
                    'bbox': {
                        'x1': int(det.bbox[0]),
                        'y1': int(det.bbox[1]),
                        'x2': int(det.bbox[2]),
                        'y2': int(det.bbox[3])
                    }
                }
                for det in detections
            ],
            'count': len(detections)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vision/ocr', methods=['POST'])
def vision_ocr():
    """OCR text extraction endpoint"""
    try:
        # Handle both JSON and form data
        if request.content_type and 'application/json' in request.content_type:
            data = request.json
            if 'image' not in data:
                return jsonify({'error': 'Image data required'}), 400
            
            image = decode_base64_image(data['image'])
            language = data.get('language', 'eng')
            
        else:
            if 'image' not in request.files:
                return jsonify({'error': 'Image file required'}), 400
            
            file = request.files['image']
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            language = request.form.get('language', 'eng')
        
        # Perform OCR
        result = ocr_processor.extract_text(image, language=language)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'text': result.text if hasattr(result, 'text') else str(result),
            'confidence': getattr(result, 'confidence', 0.0),
            'language': language
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vision/face/identify', methods=['POST'])
def vision_face_identify():
    """Face identification endpoint"""
    try:
        # Handle both JSON and form data
        if request.content_type and 'application/json' in request.content_type:
            data = request.json
            if 'image' not in data:
                return jsonify({'error': 'Image data required'}), 400
            
            image = decode_base64_image(data['image'])
            
        else:
            if 'image' not in request.files:
                return jsonify({'error': 'Image file required'}), 400
            
            file = request.files['image']
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform face recognition
        results = face_recognizer.recognize_faces(image)
        
        # Convert results to JSON
        face_results = []
        for result in results:
            face_results.append({
                'face_id': result.face_id,
                'name': result.name,
                'confidence': float(result.confidence),
                'bbox': {
                    'x': result.detection.bbox.x,
                    'y': result.detection.bbox.y,
                    'width': result.detection.bbox.width,
                    'height': result.detection.bbox.height
                }
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'faces': face_results,
            'count': len(face_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vision/classify', methods=['POST'])
def vision_classify():
    """Image classification endpoint"""
    try:
        # Handle both JSON and form data
        if request.content_type and 'application/json' in request.content_type:
            data = request.json
            if 'image' not in data:
                return jsonify({'error': 'Image data required'}), 400
            
            image = decode_base64_image(data['image'])
            top_k = data.get('top_k', 5)
            
        else:
            if 'image' not in request.files:
                return jsonify({'error': 'Image file required'}), 400
            
            file = request.files['image']
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            top_k = int(request.form.get('top_k', 5))
        
        # Perform classification
        results = classify_image(image, top_k=top_k)
        
        # Convert results to JSON
        classifications = []
        for result in results.predictions:
            classifications.append({
                'class_name': result.class_name,
                'confidence': float(result.confidence),
                'class_id': result.class_id
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'classifications': classifications,
            'processing_time': results.processing_time
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Camera Integration API Endpoints

@app.route('/camera/discover', methods=['GET'])
def camera_discover():
    """Discover CP Plus cameras on network"""
    try:
        cameras = camera_manager.discover_cameras()
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'cameras_found': cameras,
            'count': len(cameras)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera/add', methods=['POST'])
def camera_add():
    """Add a new camera to the system"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Camera configuration required'}), 400
        
        # Create camera configuration
        credentials = CameraCredentials(
            username=data['credentials']['username'],
            password=data['credentials']['password'],
            ip_address=data['credentials']['ip_address'],
            port=data['credentials'].get('port', 80),
            rtsp_port=data['credentials'].get('rtsp_port', 554),
            http_port=data['credentials'].get('http_port', 80)
        )
        
        capabilities = CameraCapabilities(
            has_360_view=data['capabilities'].get('has_360_view', True),
            has_motion_detection=data['capabilities'].get('has_motion_detection', True),
            has_motion_tracking=data['capabilities'].get('has_motion_tracking', True),
            has_privacy_mode=data['capabilities'].get('has_privacy_mode', True),
            has_ir_lights=data['capabilities'].get('has_ir_lights', True),
            has_night_vision=data['capabilities'].get('has_night_vision', True)
        )
        
        config = CameraConfig(
            camera_id=data['camera_id'],
            name=data['name'],
            credentials=credentials,
            capabilities=capabilities,
            motion_detection_enabled=data.get('motion_detection_enabled', True),
            ai_analysis_enabled=data.get('ai_analysis_enabled', True)
        )
        
        success = camera_manager.add_camera(config)
        
        if success:
            return jsonify({
                'message': 'Camera added successfully',
                'camera_id': config.camera_id,
                'status': 'success'
            })
        else:
            return jsonify({'error': 'Failed to add camera'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera/<camera_id>/status', methods=['GET'])
def camera_status(camera_id):
    """Get camera status and information"""
    try:
        status = camera_manager.get_camera_status(camera_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera/<camera_id>/control', methods=['POST'])
def camera_control(camera_id):
    """Control camera functions (PTZ, privacy mode, etc.)"""
    try:
        data = request.json
        if not data or 'action' not in data:
            return jsonify({'error': 'Action required'}), 400
        
        action = data['action']
        kwargs = {k: v for k, v in data.items() if k != 'action'}
        
        success = camera_manager.control_camera(camera_id, action, **kwargs)
        
        return jsonify({
            'camera_id': camera_id,
            'action': action,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera/<camera_id>/privacy/schedule', methods=['POST'])
def camera_set_privacy_schedule(camera_id):
    """Set privacy mode schedule for camera"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Schedule configuration required'}), 400
        
        camera_manager.set_privacy_schedule(camera_id, data)
        
        return jsonify({
            'message': 'Privacy schedule set successfully',
            'camera_id': camera_id,
            'schedule': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/camera/list', methods=['GET'])
def camera_list():
    """List all configured cameras"""
    try:
        cameras = []
        for camera_id in camera_manager.cameras:
            status = camera_manager.get_camera_status(camera_id)
            cameras.append(status)
        
        return jsonify({
            'cameras': cameras,
            'count': len(cameras),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket streaming endpoint
@socketio.on('connect')
def handle_connect():
    print('Client connected to vision stream')
    emit('status', {'message': 'Connected to vision stream'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from vision stream')

@socketio.on('start_stream')
def handle_start_stream(data):
    """Start vision processing stream"""
    stream_type = data.get('type', 'camera')  # camera, file, etc.
    processing_options = data.get('options', {})
    
    if stream_type == 'camera':
        # Start camera stream with vision processing
        start_camera_stream(processing_options)
    
    emit('stream_status', {'status': 'started', 'type': stream_type})

def start_camera_stream(options):
    """Start camera streaming with real-time vision processing"""
    def stream_worker():
        cap = cv2.VideoCapture(0)  # Default camera
        
        enable_detection = options.get('enable_detection', True)
        enable_face_recognition = options.get('enable_face_recognition', True)
        enable_classification = options.get('enable_classification', False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with vision capabilities
            vision_data = {}
            
            if enable_detection:
                try:
                    detections = detect_objects(frame, confidence_threshold=0.5)
                    vision_data['detections'] = [
                        {
                            'class_name': det.class_name,
                            'confidence': float(det.confidence),
                            'bbox': [int(x) for x in det.bbox]
                        }
                        for det in detections
                    ]
                    
                    # Draw bounding boxes
                    for det in detections:
                        x1, y1, x2, y2 = [int(x) for x in det.bbox]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except:
                    pass
            
            # if enable_face_recognition:
            #     try:
            #         faces = face_recognizer.recognize_faces(frame)
            #         vision_data['faces'] = [
            #             {
            #                 'name': face.name,
            #                 'confidence': float(face.confidence),
            #                 'bbox': [face.detection.bbox.x, face.detection.bbox.y, 
            #                        face.detection.bbox.width, face.detection.bbox.height]
            #             }
            #             for face in faces
            #         ]
                    
            #         # Draw face boxes
            #         for face in faces:
            #             x, y, w, h = face.detection.bbox.x, face.detection.bbox.y, face.detection.bbox.width, face.detection.bbox.height
            #             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #             name = face.name or "Unknown"
            #             cv2.putText(frame, f"{name} {face.confidence:.2f}", 
            #                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #     except:
            #         pass
            
            # Encode frame for streaming
            encoded_frame = encode_image_to_base64(frame)
            
            # Emit frame and vision data
            socketio.emit('vision_frame', {
                'image': encoded_frame,
                'vision_data': vision_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Control frame rate
            socketio.sleep(0.1)  # ~10 FPS
        
        cap.release()
    
    # Start streaming in background thread
    stream_thread = threading.Thread(target=stream_worker, daemon=True)
    stream_thread.start()

if __name__ == "__main__":
    print("Starting Omni-Dev Agent Vision API...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
