API Reference
=============

This section provides comprehensive documentation for all Omni-Dev Agent APIs, including REST endpoints, WebSocket connections, and Python SDK methods.

REST API Overview
-----------------

The Omni-Dev Agent provides a comprehensive REST API for all its capabilities. All endpoints return JSON responses and support both form data and JSON payloads where appropriate.

**Base URL**: ``http://localhost:5000``

**Authentication**: Currently, no authentication is required for local development. See :doc:`deployment` for production authentication setup.

Vision API Endpoints
--------------------

Object Detection
~~~~~~~~~~~~~~~~

.. http:post:: /vision/detect

   Perform object detection on an image using YOLOv8 models.

   **Request Format**:

   .. sourcecode:: http

      POST /vision/detect HTTP/1.1
      Host: localhost:5000
      Content-Type: application/json

      {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
        "confidence": 0.25
      }

   **Parameters**:
   
   :json string image: Base64-encoded image data with data URL prefix
   :json float confidence: Confidence threshold (0.0-1.0, default: 0.25)

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "timestamp": "2024-01-01T12:00:00",
        "detections": [
          {
            "class_name": "person",
            "confidence": 0.85,
            "bbox": {
              "x1": 100,
              "y1": 150,
              "x2": 300,
              "y2": 400
            }
          }
        ],
        "count": 1
      }

   :statuscode 200: Detection completed successfully
   :statuscode 400: Invalid image data or parameters
   :statuscode 500: Internal server error

   **Example Usage**:

   .. code-block:: python

      import requests
      import base64

      with open('image.jpg', 'rb') as f:
          image_data = base64.b64encode(f.read()).decode()

      response = requests.post(
          'http://localhost:5000/vision/detect',
          json={
              'image': f'data:image/jpeg;base64,{image_data}',
              'confidence': 0.25
          }
      )
      result = response.json()

OCR Text Extraction
~~~~~~~~~~~~~~~~~~~

.. http:post:: /vision/ocr

   Extract text from images using OCR (Optical Character Recognition).

   **Request Format**:

   .. sourcecode:: http

      POST /vision/ocr HTTP/1.1
      Host: localhost:5000
      Content-Type: application/json

      {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
        "language": "eng"
      }

   **Parameters**:
   
   :json string image: Base64-encoded image data
   :json string language: OCR language code (default: "eng")

   **Supported Languages**:
   
   * ``eng`` - English
   * ``fra`` - French  
   * ``deu`` - German
   * ``spa`` - Spanish
   * ``chi_sim`` - Chinese Simplified
   * ``ara`` - Arabic

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "timestamp": "2024-01-01T12:00:00",
        "text": "Extracted text content",
        "confidence": 0.92,
        "language": "eng"
      }

   **Example Usage**:

   .. code-block:: python

      response = requests.post(
          'http://localhost:5000/vision/ocr',
          json={
              'image': f'data:image/jpeg;base64,{image_data}',
              'language': 'eng'
          }
      )
      text_result = response.json()
      print(f"Text: {text_result['text']}")

Face Recognition
~~~~~~~~~~~~~~~~

.. http:post:: /vision/face/identify

   Identify faces in images with confidence scoring and bounding boxes.

   **Request Format**:

   .. sourcecode:: http

      POST /vision/face/identify HTTP/1.1
      Host: localhost:5000
      Content-Type: application/json

      {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
      }

   **Parameters**:
   
   :json string image: Base64-encoded image data

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "timestamp": "2024-01-01T12:00:00",
        "faces": [
          {
            "face_id": "face_001",
            "name": "John Doe",
            "confidence": 0.89,
            "bbox": {
              "x": 120,
              "y": 80,
              "width": 150,
              "height": 180
            }
          }
        ],
        "count": 1
      }

Image Classification
~~~~~~~~~~~~~~~~~~~~

.. http:post:: /vision/classify

   Classify images into categories with confidence scores.

   **Request Format**:

   .. sourcecode:: http

      POST /vision/classify HTTP/1.1
      Host: localhost:5000
      Content-Type: application/json

      {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
        "top_k": 5
      }

   **Parameters**:
   
   :json string image: Base64-encoded image data
   :json int top_k: Number of top predictions to return (default: 5)

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "timestamp": "2024-01-01T12:00:00",
        "classifications": [
          {
            "class_name": "dog",
            "confidence": 0.92,
            "class_id": 243
          },
          {
            "class_name": "puppy",
            "confidence": 0.78,
            "class_id": 244
          }
        ],
        "processing_time": 0.145
      }

Camera API Endpoints
--------------------

Camera Discovery
~~~~~~~~~~~~~~~~

.. http:get:: /camera/discover

   Discover CP Plus cameras on the network.

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "timestamp": "2024-01-01T12:00:00",
        "cameras_found": [
          {
            "ip_address": "192.168.1.100",
            "name": "CP Plus Camera",
            "model": "CP-VAS-2TMPIR36",
            "capabilities": ["ptz", "ir", "motion"]
          }
        ],
        "count": 1
      }

Add Camera
~~~~~~~~~~

.. http:post:: /camera/add

   Add a new camera to the system.

   **Request Format**:

   .. sourcecode:: http

      POST /camera/add HTTP/1.1
      Host: localhost:5000
      Content-Type: application/json

      {
        "camera_id": "cam_001",
        "name": "Front Door Camera",
        "credentials": {
          "username": "admin",
          "password": "password123",
          "ip_address": "192.168.1.100",
          "port": 80,
          "rtsp_port": 554,
          "http_port": 80
        },
        "capabilities": {
          "has_360_view": true,
          "has_motion_detection": true,
          "has_motion_tracking": true,
          "has_privacy_mode": true,
          "has_ir_lights": true,
          "has_night_vision": true
        },
        "motion_detection_enabled": true,
        "ai_analysis_enabled": true
      }

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "message": "Camera added successfully",
        "camera_id": "cam_001",
        "status": "success"
      }

Camera Status
~~~~~~~~~~~~~

.. http:get:: /camera/(camera_id)/status

   Get current status and information for a specific camera.

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "camera_id": "cam_001",
        "name": "Front Door Camera",
        "status": "online",
        "connection_quality": "good",
        "current_position": {
          "pan": 45,
          "tilt": 20,
          "zoom": 1.5
        },
        "motion_detection": {
          "enabled": true,
          "sensitivity": 25,
          "last_motion": "2024-01-01T11:45:00"
        },
        "privacy_mode": false,
        "recording": false
      }

Camera Control
~~~~~~~~~~~~~~

.. http:post:: /camera/(camera_id)/control

   Control camera functions such as PTZ (Pan-Tilt-Zoom) and privacy mode.

   **Request Format**:

   .. sourcecode:: http

      POST /camera/cam_001/control HTTP/1.1
      Host: localhost:5000
      Content-Type: application/json

      {
        "action": "pan_right",
        "speed": 50
      }

   **Available Actions**:
   
   * ``pan_left`` / ``pan_right`` - Pan camera left/right
   * ``tilt_up`` / ``tilt_down`` - Tilt camera up/down
   * ``zoom_in`` / ``zoom_out`` - Zoom in/out
   * ``go_preset`` - Move to preset position
   * ``set_preset`` - Set current position as preset
   * ``privacy_on`` / ``privacy_off`` - Enable/disable privacy mode
   * ``ir_on`` / ``ir_off`` - Control IR lights

   **Parameters**:
   
   :json string action: Action to perform
   :json int speed: Speed of movement (1-100, optional)
   :json int preset_id: Preset position ID (for preset actions)

Privacy Schedule
~~~~~~~~~~~~~~~~

.. http:post:: /camera/(camera_id)/privacy/schedule

   Set privacy mode schedule for a camera.

   **Request Format**:

   .. sourcecode:: http

      POST /camera/cam_001/privacy/schedule HTTP/1.1
      Host: localhost:5000
      Content-Type: application/json

      {
        "schedule": {
          "monday": [
            {"start": "09:00", "end": "17:00"}
          ],
          "tuesday": [
            {"start": "09:00", "end": "17:00"}
          ]
        },
        "timezone": "UTC"
      }

List Cameras
~~~~~~~~~~~~

.. http:get:: /camera/list

   List all configured cameras with their status.

   **Response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "cameras": [
          {
            "camera_id": "cam_001",
            "name": "Front Door Camera",
            "status": "online",
            "ip_address": "192.168.1.100"
          }
        ],
        "count": 1,
        "timestamp": "2024-01-01T12:00:00"
      }

WebSocket API
-------------

Real-time Vision Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Connect to the WebSocket endpoint for real-time vision processing:

**Endpoint**: ``ws://localhost:5000/socket.io/``

**Connection**:

.. code-block:: javascript

   const socket = io('http://localhost:5000');
   
   socket.on('connect', () => {
       console.log('Connected to vision stream');
   });

**Start Streaming**:

.. code-block:: javascript

   socket.emit('start_stream', {
       type: 'camera',
       options: {
           enable_detection: true,
           enable_face_recognition: true,
           enable_classification: false
       }
   });

**Receive Frames**:

.. code-block:: javascript

   socket.on('vision_frame', (data) => {
       const { image, vision_data, timestamp } = data;
       
       // Process vision data
       console.log('Detections:', vision_data.detections);
       console.log('Faces:', vision_data.faces);
       
       // Update UI with processed frame
       updateVideoFrame(image);
   });

**Events**:

* ``connect`` - Connection established
* ``disconnect`` - Connection lost
* ``start_stream`` - Start vision processing stream
* ``stop_stream`` - Stop vision processing stream
* ``vision_frame`` - Receive processed frame data
* ``stream_status`` - Stream status updates

Python SDK
----------

Direct Python Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the components directly in your Python applications:

**Object Detection**:

.. code-block:: python

   from src.components.ai_vision.object_detection import detect_objects
   import cv2

   # Load image
   image = cv2.imread('image.jpg')
   
   # Detect objects
   detections = detect_objects(image, confidence_threshold=0.25)
   
   for detection in detections:
       print(f"{detection.class_name}: {detection.confidence:.3f}")

**OCR Processing**:

.. code-block:: python

   from src.components.ai_vision.ocr import OCRProcessor
   
   ocr = OCRProcessor()
   result = ocr.extract_text(image, language='eng')
   print(f"Text: {result.text}")
   print(f"Confidence: {result.confidence}")

**Face Recognition**:

.. code-block:: python

   from src.components.ai_vision.face_recognition import create_face_recognizer
   
   face_recognizer = create_face_recognizer()
   faces = face_recognizer.recognize_faces(image)
   
   for face in faces:
       print(f"Name: {face.name}, Confidence: {face.confidence}")

**Camera Management**:

.. code-block:: python

   from src.components.camera_integration.camera_manager import CameraManager
   from src.components.camera_integration.camera_manager import CameraConfig, CameraCredentials
   
   # Initialize camera manager
   camera_manager = CameraManager()
   
   # Discover cameras
   cameras = camera_manager.discover_cameras()
   
   # Add camera
   credentials = CameraCredentials(
       username="admin",
       password="password",
       ip_address="192.168.1.100"
   )
   
   config = CameraConfig(
       camera_id="cam_001",
       name="Front Camera",
       credentials=credentials
   )
   
   camera_manager.add_camera(config)

**Task Orchestration**:

.. code-block:: python

   from src.core.orchestration import Orchestrator
   import asyncio
   
   async def run_vision_task():
       orchestrator = Orchestrator()
       
       result = await orchestrator.handle_request(
           type='vision',
           task='object_detection',
           payload={
               'frame': image_array,
               'conf_threshold': 0.25
           }
       )
       
       if result['status'] == 'success':
           print(f"Detected {len(result['data'])} objects")
   
   # Run async task
   asyncio.run(run_vision_task())

Error Handling
--------------

All API endpoints return consistent error formats:

**Error Response Format**:

.. sourcecode:: http

   HTTP/1.1 400 Bad Request
   Content-Type: application/json

   {
     "error": "Detailed error message",
     "timestamp": "2024-01-01T12:00:00",
     "endpoint": "/vision/detect",
     "request_id": "req_12345"
   }

**Common Error Codes**:

* ``400`` - Bad Request (invalid parameters)
* ``404`` - Not Found (endpoint or resource not found)
* ``429`` - Too Many Requests (rate limiting)
* ``500`` - Internal Server Error
* ``503`` - Service Unavailable (model loading, etc.)

**Error Handling Example**:

.. code-block:: python

   try:
       response = requests.post(
           'http://localhost:5000/vision/detect',
           json={'image': image_data}
       )
       response.raise_for_status()
       result = response.json()
   except requests.exceptions.HTTPError as e:
       error_data = e.response.json()
       print(f"API Error: {error_data['error']}")
   except requests.exceptions.RequestException as e:
       print(f"Network Error: {e}")

Rate Limiting
-------------

To ensure optimal performance, the following rate limits apply:

* **Vision API**: 10 requests per second per client
* **Camera API**: 5 requests per second per client  
* **WebSocket**: 1 connection per client with 30 FPS limit

**Rate Limit Headers**:

.. sourcecode:: http

   X-RateLimit-Limit: 10
   X-RateLimit-Remaining: 9
   X-RateLimit-Reset: 1640995200

Best Practices
--------------

1. **Image Optimization**:
   
   * Resize large images before sending to reduce processing time
   * Use JPEG format for photos, PNG for graphics
   * Compress images appropriately (quality 80-90 for JPEG)

2. **Batch Processing**:
   
   * Use batch endpoints when processing multiple images
   * Implement client-side queuing for large volumes

3. **Error Handling**:
   
   * Always implement proper error handling
   * Use exponential backoff for retries
   * Log API responses for debugging

4. **Performance**:
   
   * Cache results when appropriate
   * Use WebSocket streaming for real-time applications
   * Monitor rate limits and adjust accordingly

5. **Security**:
   
   * Validate all inputs before sending to API
   * Use HTTPS in production
   * Implement proper authentication and authorization

For more detailed examples and advanced usage, see :doc:`examples/index`.

.. seealso::
   
   * :doc:`vision_guide` - Detailed vision capabilities guide
   * :doc:`camera_integration` - Camera setup and configuration
   * :doc:`examples/index` - Complete code examples
   * :doc:`troubleshooting` - Common issues and solutions
