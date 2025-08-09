Getting Started
===============

Welcome to Omni-Dev Agent! This guide will help you get up and running quickly with the platform's core capabilities.

Prerequisites
-------------

Before you begin, ensure you have:

* **Python 3.8 or higher** installed
* **Git** for version control
* **CUDA-capable GPU** (optional, for enhanced performance)
* **Network access** for downloading models

.. note::
   While a GPU is not required, it significantly improves performance for computer vision tasks.

Installation
------------

Quick Installation
~~~~~~~~~~~~~~~~~~

Clone the repository and install dependencies:

.. code-block:: bash

   git clone https://github.com/yourusername/omni-dev-agent.git
   cd omni-dev-agent
   pip install -r requirements.txt

Component-Specific Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install specific components based on your needs:

**Vision Components:**

.. code-block:: bash

   # Object Detection
   pip install -r src/components/ai_vision/requirements_object_detection.txt
   
   # OCR Processing
   pip install -r src/components/ai_vision/requirements_ocr.txt
   
   # Face Recognition
   pip install -r src/components/ai_vision/requirements_face_recognition.txt
   
   # Image Classification
   pip install -r src/components/ai_vision/requirements_image_classification.txt

**Camera Integration:**

.. code-block:: bash

   pip install -r src/requirements_camera_vision.txt

**Performance Optimization:**

.. code-block:: bash

   pip install -r src/requirements_performance_optimization.txt

Verify Installation
~~~~~~~~~~~~~~~~~~~

Test your installation:

.. code-block:: bash

   cd src
   python -c "import main; print('Installation successful!')"
   
   # Run basic tests
   python -m pytest tests/unit/ -v

First Steps
-----------

1. Start the Web Server
~~~~~~~~~~~~~~~~~~~~~~~

Launch the main application:

.. code-block:: bash

   cd src
   python main.py

The server will start on ``http://localhost:5000``. You should see:

.. code-block:: console

   Starting Omni-Dev Agent Vision API...
   * Serving Flask app "main" (lazy loading)
   * Environment: production
   * Debug mode: on
   * Running on all addresses.
   * Running on http://0.0.0.0:5000/

2. Test Basic Functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a new terminal and test the API:

.. code-block:: bash

   # Test server is running
   curl http://localhost:5000/
   # Output: "Omni-Dev Agent Vision API - Ready"

3. Access the Web Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open your browser and navigate to:

* **Main Interface:** http://localhost:5000/ui
* **Demo Page:** http://localhost:5000/demo

Basic Usage Examples
--------------------

Object Detection
~~~~~~~~~~~~~~~~

Test object detection with a sample image:

.. code-block:: python

   import requests
   import base64
   
   # Load and encode an image
   with open('sample_image.jpg', 'rb') as f:
       image_data = base64.b64encode(f.read()).decode()
   
   # Make detection request
   response = requests.post(
       'http://localhost:5000/vision/detect',
       json={
           'image': f'data:image/jpeg;base64,{image_data}',
           'confidence': 0.25
       }
   )
   
   result = response.json()
   print(f"Detected {result['count']} objects:")
   for detection in result['detections']:
       print(f"  - {detection['class_name']}: {detection['confidence']:.2f}")

OCR Text Extraction
~~~~~~~~~~~~~~~~~~~

Extract text from an image:

.. code-block:: python

   response = requests.post(
       'http://localhost:5000/vision/ocr',
       json={
           'image': f'data:image/jpeg;base64,{image_data}',
           'language': 'eng'
       }
   )
   
   result = response.json()
   print(f"Extracted text: {result['text']}")
   print(f"Confidence: {result['confidence']:.2f}")

Camera Discovery
~~~~~~~~~~~~~~~~

Discover cameras on your network:

.. code-block:: python

   # Discover cameras
   response = requests.get('http://localhost:5000/camera/discover')
   cameras = response.json()
   
   print(f"Found {cameras['count']} cameras:")
   for camera in cameras['cameras_found']:
       print(f"  - {camera['name']} at {camera['ip_address']}")

Using the Python SDK
~~~~~~~~~~~~~~~~~~~~

For more advanced usage, use the Python SDK directly:

.. code-block:: python

   from src.components.ai_vision.object_detection import detect_objects
   from src.components.ai_vision.ocr import OCRProcessor
   from src.components.camera_integration.camera_manager import CameraManager
   import cv2
   
   # Load an image
   image = cv2.imread('sample_image.jpg')
   
   # Object detection
   detections = detect_objects(image, confidence_threshold=0.25)
   print(f"Found {len(detections)} objects")
   
   # OCR processing
   ocr_processor = OCRProcessor()
   text_result = ocr_processor.extract_text(image)
   print(f"Extracted text: {text_result.text}")
   
   # Camera management
   camera_manager = CameraManager()
   cameras = camera_manager.discover_cameras()
   print(f"Available cameras: {cameras}")

Configuration
-------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

Create a ``.env`` file in the project root:

.. code-block:: bash

   # Flask Configuration
   FLASK_ENV=development
   SECRET_KEY=your-secret-key-here
   
   # Vision Configuration
   CONFIDENCE_THRESHOLD=0.25
   ENABLE_GPU=true
   MAX_CONCURRENT_TASKS=5
   
   # Camera Configuration
   CAMERA_STORAGE_PATH=./camera_data
   RTSP_TIMEOUT=30
   
   # Performance Configuration
   ENABLE_PROFILING=true
   LOG_LEVEL=INFO

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For advanced configuration, edit the YAML files in ``src/config/``:

**Vision Configuration** (``src/config/vision_config.yaml``):

.. code-block:: yaml

   vision:
     models:
       object_detection:
         model_name: "yolov8n"
         confidence_threshold: 0.25
         device: "cuda"  # or "cpu"
       
       ocr:
         language: "eng"
         oem: 3
         psm: 6
   
     performance:
       batch_size: 1
       max_concurrent_tasks: 5
       enable_caching: true

**Camera Configuration** (``src/config/camera_config.yaml``):

.. code-block:: yaml

   cameras:
     default_settings:
       rtsp_timeout: 30
       connection_retry: 3
       motion_sensitivity: 25
     
     discovery:
       scan_range: "192.168.1.0/24"
       timeout: 10

Next Steps
----------

Now that you have Omni-Dev Agent running, you can:

1. **Explore Vision Capabilities**: See :doc:`vision_guide` for detailed computer vision features
2. **Set Up Cameras**: Follow :doc:`camera_integration` to connect your cameras  
3. **Check Performance**: Use :doc:`performance_optimization` to optimize your setup
4. **View Examples**: Browse :doc:`examples/index` for more code examples
5. **Read API Docs**: Check :doc:`api_reference` for complete API documentation

Common Issues
-------------

Port Already in Use
~~~~~~~~~~~~~~~~~~~

If port 5000 is already in use:

.. code-block:: bash

   # Kill process using port 5000
   lsof -ti:5000 | xargs kill -9
   
   # Or run on a different port
   export FLASK_RUN_PORT=5001
   python main.py

GPU Not Detected
~~~~~~~~~~~~~~~~~

If CUDA GPU is not detected:

.. code-block:: bash

   # Check CUDA installation
   nvidia-smi
   
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Model Download Issues
~~~~~~~~~~~~~~~~~~~~~

If models fail to download:

.. code-block:: bash

   # Pre-download models
   python -c "
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   print('YOLOv8 model downloaded successfully')
   "

Memory Issues
~~~~~~~~~~~~~

If you encounter out-of-memory errors:

1. **Reduce batch size** in configuration
2. **Use smaller models** (e.g., yolov8n instead of yolov8l)
3. **Enable GPU** if available
4. **Close other applications** to free up memory

Getting Help
------------

If you need further assistance:

* Check the :doc:`troubleshooting` guide
* Search the `GitHub Issues <https://github.com/yourusername/omni-dev-agent/issues>`_
* Ask questions in `Discussions <https://github.com/yourusername/omni-dev-agent/discussions>`_
* Review the :doc:`faq` section

.. tip::
   Join our community Discord server for real-time help and discussions with other users!

What's Next?
------------

Continue your journey with these guides:

.. toctree::
   :maxdepth: 1
   
   vision_guide
   camera_integration
   performance_optimization
   examples/index
