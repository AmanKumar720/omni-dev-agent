# Face Detection & Recognition Module

A comprehensive face detection and recognition system using OpenCV DNN for detection, FaceNet-style embeddings for recognition, and SQLite for persistent identity storage.

## Features

### ✅ Face Detection
- **OpenCV DNN Face Detector**: Supports custom DNN models (TensorFlow .pb format)
- **Haar Cascade Fallback**: Built-in OpenCV cascade classifier for basic detection
- **Configurable Confidence**: Adjustable detection threshold
- **Bounding Box Output**: Precise face coordinates and confidence scores

### ✅ Face Recognition  
- **Face Embeddings**: FaceNet-style 128-dimensional feature vectors
- **Cosine Similarity**: Robust face matching using cosine distance
- **Multiple Embeddings**: Support for multiple face samples per identity
- **Unknown Detection**: Handles unrecognized faces gracefully

### ✅ Identity Management
- **SQLite Persistence**: Robust database storage in `faces.db`
- **Multiple Images**: Enroll identities with multiple face samples
- **Metadata Support**: Store additional identity information
- **CRUD Operations**: Full create, read, update, delete support

### ✅ REST API
- **Flask Web Server**: Complete HTTP API for integration
- **JSON & Form Support**: Multiple input formats
- **Base64 Images**: Direct image data transmission
- **File Uploads**: Standard multipart file handling

### ✅ Production Ready
- **Async Support**: Compatible with async frameworks
- **Error Handling**: Comprehensive error reporting
- **Logging**: Detailed operation logging
- **Testing**: Full test suite included

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_face_recognition.txt

# Basic usage
python example_face_recognition.py

# Start API server
python face_api.py --host 127.0.0.1 --port 5000
```

### Basic Usage

```python
from face_recognition import create_face_recognizer, recognize_faces, enroll_person
import cv2

# Create face recognizer
recognizer = create_face_recognizer(
    db_path="faces.db",
    detection_threshold=0.5,
    recognition_threshold=0.7
)

# Enroll a person
images = [cv2.imread("person1_1.jpg"), cv2.imread("person1_2.jpg")]
identity_id = enroll_person(
    name="John Doe", 
    face_images=images,
    metadata={"department": "Engineering", "employee_id": "EMP001"}
)

# Recognize faces in new image
test_image = cv2.imread("test.jpg")
results = recognize_faces(test_image)

for result in results:
    print(f"Name: {result.name}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Box: ({result.detection.bbox.x}, {result.detection.bbox.y})")
```

### API Usage

```python
import requests
import base64

# Health check
response = requests.get("http://127.0.0.1:5000/api/health")
print(response.json())

# Enroll identity via API
with open("face.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

enroll_data = {
    "name": "Jane Smith",
    "images": [image_b64],
    "metadata": {"department": "HR"}
}

response = requests.post("http://127.0.0.1:5000/api/enroll", json=enroll_data)
print(response.json())

# Recognize faces
recognize_data = {"image": image_b64}
response = requests.post("http://127.0.0.1:5000/api/recognize", json=recognize_data)
print(response.json())
```

## Architecture

### Core Components

```
face_recognition.py
├── FaceDetector       # OpenCV DNN face detection
├── FaceEmbedder      # Face feature extraction  
├── FaceDatabase      # SQLite persistence
├── FaceRecognizer    # Main recognition pipeline
└── FaceRecognitionTask # Async task wrapper
```

### Database Schema

```sql
-- Identity storage
CREATE TABLE identities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embeddings BLOB NOT NULL,      -- Pickled numpy arrays
    metadata TEXT,                 -- JSON metadata
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Detection logging
CREATE TABLE face_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id TEXT,
    confidence REAL,
    detection_data TEXT,           -- JSON detection info
    timestamp TIMESTAMP,
    FOREIGN KEY (identity_id) REFERENCES identities (id)
);
```

### Face Recognition Pipeline

```
Input Image → Face Detection → Face Cropping → Embedding Extraction → Database Matching → Result
```

1. **Face Detection**: Locate faces in image using OpenCV DNN or Haar cascades
2. **Face Cropping**: Extract face regions based on bounding boxes
3. **Embedding Extraction**: Generate 128D feature vectors for each face
4. **Database Matching**: Compare embeddings against enrolled identities using cosine similarity
5. **Result Generation**: Return matched identities with confidence scores

## API Reference

### Endpoints

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0.0"
}
```

#### `POST /api/enroll`
Enroll a new identity.

**JSON Request:**
```json
{
    "name": "John Doe",
    "images": ["base64_image1", "base64_image2"],
    "metadata": {"department": "Engineering", "employee_id": "EMP001"}
}
```

**Response:**
```json
{
    "message": "Identity enrolled successfully",
    "identity_id": "abc123...",
    "name": "John Doe", 
    "num_images": 2
}
```

#### `POST /api/recognize`
Recognize faces in image.

**JSON Request:**
```json
{
    "image": "base64_encoded_image"
}
```

**Response:**
```json
{
    "results": [
        {
            "face_id": "abc123...",
            "name": "John Doe",
            "confidence": 0.92,
            "bbox": {"x": 100, "y": 50, "width": 80, "height": 100},
            "detection_confidence": 0.98
        }
    ],
    "count": 1
}
```

#### `GET /api/identities`
List all enrolled identities.

**Response:**
```json
{
    "identities": [
        {
            "id": "abc123...",
            "name": "John Doe",
            "num_embeddings": 3,
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z"
        }
    ],
    "total": 1
}
```

#### `GET /api/identities/{identity_id}`
Get specific identity information.

#### `DELETE /api/identities/{identity_id}`
Delete an identity.

#### `POST /api/detect`
Detect faces without recognition.

## Configuration

### Face Recognition Parameters

```python
recognizer = create_face_recognizer(
    db_path="faces.db",               # Database path
    detection_threshold=0.5,          # Min face detection confidence
    recognition_threshold=0.7         # Min face recognition similarity
)
```

### API Server Parameters

```bash
python face_api.py \
    --host 127.0.0.1 \
    --port 5000 \
    --db-path faces.db \
    --detection-threshold 0.5 \
    --recognition-threshold 0.7 \
    --upload-folder temp_uploads \
    --debug
```

## Advanced Usage

### Custom Face Detector

```python
# Use custom DNN model
detector = FaceDetector(
    model_path="opencv_face_detector.pb",
    config_path="opencv_face_detector.pbtxt", 
    confidence_threshold=0.7
)
```

### Custom Face Embedder

```python
# Use custom embedding model
embedder = FaceEmbedder(model_path="facenet_model.pb")
```

### Async Task Integration

```python
from face_recognition import FaceRecognitionTask
from core import AIVisionAgent

# Create vision agent
agent = AIVisionAgent("face_agent", "Face Recognition Agent")

# Create face recognition task
recognizer = create_face_recognizer()
task = FaceRecognitionTask("face_task_001", recognizer)

# Register and execute
agent.register_task(task)
result = await agent.execute_task("face_task_001", image_data)
```

### Batch Processing

```python
# Process multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = []

for path in image_paths:
    result = recognize_faces(path)
    results.extend(result)

# Or using recognizer directly for better performance
recognizer = create_face_recognizer()
for path in image_paths:
    image = cv2.imread(path)
    result = recognizer.recognize_faces(image)
    results.extend(result)
```

## Production Notes

### Performance Optimization

1. **Real Face Models**: Replace demo feature extractor with production models:
   ```python
   # Use real FaceNet model
   embedder = FaceEmbedder("facenet_keras.h5")
   
   # Use OpenCV DNN face detector
   detector = FaceDetector(
       "opencv_face_detector_uint8.pb",
       "opencv_face_detector.pbtxt"
   )
   ```

2. **Database Connection Pooling**: Use proper database connections for high load
3. **Caching**: Implement embedding caching for frequently accessed identities
4. **Batch Processing**: Process multiple faces simultaneously
5. **GPU Acceleration**: Use CUDA-enabled OpenCV for faster processing

### Security Considerations

1. **API Authentication**: Add authentication to enrollment endpoints
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **Input Validation**: Validate all image inputs thoroughly  
4. **Database Encryption**: Encrypt sensitive identity data
5. **Access Logging**: Log all face recognition attempts

### Scalability

1. **Database**: Migrate to PostgreSQL/MySQL for production
2. **Distributed Processing**: Use message queues for async processing
3. **Load Balancing**: Deploy multiple API instances
4. **Model Serving**: Use dedicated model serving infrastructure
5. **Monitoring**: Implement comprehensive monitoring and alerting

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest test_face_recognition.py -v

# Run specific test class
python -m pytest test_face_recognition.py::TestFaceDatabase -v

# Run with coverage
pip install pytest-cov
python -m pytest test_face_recognition.py --cov=face_recognition --cov-report=html
```

### Example Test Output

```
test_face_recognition.py::TestBoundingBox::test_bounding_box_properties PASSED
test_face_recognition.py::TestFaceDatabase::test_database_initialization PASSED
test_face_recognition.py::TestFaceDatabase::test_add_and_get_identity PASSED
test_face_recognition.py::TestFaceDetector::test_detector_initialization PASSED
test_face_recognition.py::TestFaceEmbedder::test_extract_embedding PASSED
test_face_recognition.py::TestFaceRecognizer::test_recognizer_initialization PASSED
test_face_recognition.py::TestFaceRecognitionAPI::test_health_check PASSED
test_face_recognition.py::test_integration_workflow PASSED
```

## Troubleshooting

### Common Issues

**1. No faces detected:**
```python
# Lower detection threshold
detector = FaceDetector(confidence_threshold=0.3)

# Check image quality and face size
# Faces should be at least 50x50 pixels
```

**2. Poor recognition accuracy:**
```python
# Lower recognition threshold
recognizer = create_face_recognizer(recognition_threshold=0.5)

# Enroll with more diverse face samples
# Include different angles, lighting, expressions
```

**3. Database locked errors:**
```python
# Ensure proper database cleanup
try:
    # database operations
    pass
finally:
    conn.close()
```

**4. API server issues:**
```bash
# Check if port is available
netstat -an | grep 5000

# Increase request timeout
curl -m 30 http://127.0.0.1:5000/api/health
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# API debug mode
python face_api.py --debug
```

## Dependencies

See `requirements_face_recognition.txt` for full dependency list:

- **opencv-python**: Computer vision operations
- **numpy**: Numerical computing
- **scipy**: Scientific computing (cosine similarity)
- **Flask**: Web API framework  
- **Pillow**: Image processing
- **aiohttp/aiofiles**: Async support

### Optional Production Dependencies

```bash
# Deep learning models (production)
pip install tensorflow>=2.8.0
pip install onnxruntime>=1.12.0
pip install insightface>=0.6.0

# Enhanced database
pip install psycopg2-binary  # PostgreSQL
pip install sqlalchemy      # ORM

# Monitoring
pip install prometheus_client
pip install sentry-sdk
```

## License

This face recognition module is part of the AI Vision component system and follows the same licensing terms as the parent project.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/face-recognition-enhancement`)
3. Commit your changes (`git commit -am 'Add face alignment preprocessing'`)
4. Push to the branch (`git push origin feature/face-recognition-enhancement`)
5. Create a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone <repository>
cd src/components/ai_vision

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements_face_recognition.txt
pip install pytest pytest-cov black flake8

# Run tests before committing
python -m pytest test_face_recognition.py -v
```

## Support

For issues and questions:

1. Check this documentation
2. Review the example files (`example_face_recognition.py`)
3. Run the test suite to verify functionality
4. Check the logs for detailed error information
5. Create an issue with detailed reproduction steps
