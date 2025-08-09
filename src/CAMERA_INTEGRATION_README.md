# CP Plus Camera Integration with AI Vision

This comprehensive integration system connects your CP Plus CP-E305A Wi-Fi camera with the Omni-Dev Agent, providing intelligent monitoring, automated responses, and seamless OS integration.

## Features

### 🎥 Camera Integration
- **Automatic Network Discovery** - Scans your network to find CP Plus cameras
- **360° View Control** - Pan, tilt, zoom control via API
- **Privacy Mode** - Scheduled lens cover control
- **IR Night Vision** - Automatic or manual control
- **RTSP Streaming** - High-quality video streaming
- **Motion Detection** - Hardware and software-based detection

### 🤖 AI Vision Capabilities
- **Object Detection** - Real-time detection of people, vehicles, animals
- **Face Recognition** - Identify known individuals
- **OCR Text Extraction** - Read text from camera feed
- **Image Classification** - Categorize scenes and objects
- **Motion Tracking** - Smart motion analysis

### 🔔 Smart Notifications
- **Cross-platform Notifications** - Windows, macOS, Linux support
- **Text-to-Speech** - Voice announcements
- **Email Alerts** - Automated email notifications with images
- **Webhook Integration** - Connect to external services
- **Smart Home Integration** - Control lights, alarms, etc.

### 🏠 Home Automation
- **Time-based Rules** - Different actions for day/night
- **Presence Detection** - Welcome home notifications
- **Security Alerts** - Immediate notifications for unknown persons
- **Automated Recording** - Event-triggered video recording
- **Privacy Scheduling** - Automatic privacy mode during personal hours

## Quick Start

### 1. Installation

```bash
# Clone the repository and navigate to src directory
cd omni-dev-agent/src

# Install dependencies
pip install -r requirements_camera_vision.txt

# Run the setup wizard
python setup_camera_integration.py
```

### 2. Setup Wizard

The setup wizard will guide you through:
1. **Network Discovery** - Automatically find your CP Plus camera
2. **Credential Configuration** - Set up camera access
3. **AI Vision Setup** - Configure intelligent features
4. **Notification Setup** - Configure alerts and automation
5. **Testing** - Verify everything works

### 3. Start the API Server

```bash
python main.py
```

Your vision API server will be running at `http://localhost:5000`

## API Endpoints

### Vision Processing

#### Object Detection
```http
POST /vision/detect
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "confidence": 0.5
}
```

Response:
```json
{
    "timestamp": "2024-08-06T21:30:00Z",
    "detections": [
        {
            "class_name": "person",
            "confidence": 0.85,
            "bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 300}
        }
    ],
    "count": 1
}
```

#### Face Recognition
```http
POST /vision/face/identify
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

#### OCR Text Extraction
```http
POST /vision/ocr
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "language": "eng"
}
```

#### Image Classification
```http
POST /vision/classify
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "top_k": 5
}
```

### Camera Management

#### Discover Cameras
```http
GET /camera/discover
```

Response:
```json
{
    "timestamp": "2024-08-06T21:30:00Z",
    "cameras_found": [
        {
            "ip": "192.168.1.100",
            "port": 80,
            "manufacturer": "CP Plus",
            "model": "CP-E305A"
        }
    ],
    "count": 1
}
```

#### Add Camera
```http
POST /camera/add
Content-Type: application/json

{
    "camera_id": "cp_plus_main",
    "name": "Main Security Camera",
    "credentials": {
        "username": "admin",
        "password": "your_password",
        "ip_address": "192.168.1.100"
    },
    "capabilities": {
        "has_360_view": true,
        "has_motion_detection": true,
        "has_privacy_mode": true,
        "has_ir_lights": true
    },
    "motion_detection_enabled": true,
    "ai_analysis_enabled": true
}
```

#### Camera Control
```http
POST /camera/cp_plus_main/control
Content-Type: application/json

{
    "action": "ptz_control",
    "direction": "left",
    "speed": 5
}
```

Available actions:
- `enable_privacy_mode` / `disable_privacy_mode`
- `enable_motion_detection` / `disable_motion_detection`
- `set_ir_lights` (with `enabled: true/false`)
- `ptz_control` (with `direction: up/down/left/right/zoom_in/zoom_out/stop`)

#### Set Privacy Schedule
```http
POST /camera/cp_plus_main/privacy/schedule
Content-Type: application/json

{
    "privacy_hours": [22, 23, 0, 1, 2, 3, 4, 5, 6]
}
```

#### List Cameras
```http
GET /camera/list
```

#### Camera Status
```http
GET /camera/cp_plus_main/status
```

### WebSocket Streaming

Connect to `ws://localhost:5000` for real-time video streaming with AI analysis.

```javascript
const socket = io('http://localhost:5000');

// Start streaming
socket.emit('start_stream', {
    type: 'camera',
    options: {
        enable_detection: true,
        enable_face_recognition: true,
        enable_classification: false
    }
});

// Receive frames
socket.on('vision_frame', (data) => {
    console.log('Frame received:', data.timestamp);
    console.log('Vision data:', data.vision_data);
    // data.image contains base64 encoded frame
});
```

## Configuration

### Camera Configuration

The system automatically detects your camera's capabilities, but you can customize:

```json
{
    "camera_id": "cp_plus_main",
    "name": "Main Camera",
    "credentials": {
        "username": "admin",
        "password": "your_password",
        "ip_address": "192.168.1.100",
        "rtsp_port": 554,
        "http_port": 80
    },
    "capabilities": {
        "has_360_view": true,
        "has_motion_detection": true,
        "has_motion_tracking": true,
        "has_privacy_mode": true,
        "has_ir_lights": true,
        "has_night_vision": true,
        "max_resolution": [1920, 1080]
    }
}
```

### Automation Rules

Configure intelligent responses in `camera_config.json`:

```json
{
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password"
    },
    "automation_rules": [
        {
            "name": "Person Detection - Day",
            "event_type": "motion_detected",
            "conditions": {
                "time_range": [8, 18],
                "detected_objects": ["person"],
                "min_confidence": 0.7
            },
            "actions": [
                {
                    "type": "send_notification",
                    "title": "Person Detected",
                    "message": "Someone is at your door!",
                    "sound": true,
                    "speak": true
                },
                {
                    "type": "send_email",
                    "to_email": "alerts@yourdomain.com",
                    "subject": "Security Alert",
                    "body": "Person detected at {timestamp}"
                }
            ]
        },
        {
            "name": "Night Security",
            "event_type": "motion_detected",
            "conditions": {
                "time_range": [20, 6]
            },
            "actions": [
                {
                    "type": "control_lights",
                    "action": "flash",
                    "brightness": 100
                },
                {
                    "type": "sound_alarm",
                    "duration": 5
                }
            ]
        }
    ]
}
```

## Common Use Cases

### 1. Home Security System

```python
# Set up comprehensive security monitoring
camera_config = {
    "motion_detection_enabled": True,
    "ai_analysis_enabled": True
}

# Configure alerts for unknown people
automation_rule = {
    "name": "Unknown Person Alert",
    "event_type": "motion_detected", 
    "conditions": {
        "detected_objects": ["person"],
        "min_confidence": 0.8
    },
    "actions": [
        {"type": "send_notification", "title": "Security Alert"},
        {"type": "sound_alarm", "duration": 3},
        {"type": "record_video", "duration": 30}
    ]
}
```

### 2. Smart Doorbell

```python
# Welcome known family members
welcome_rule = {
    "name": "Family Welcome",
    "event_type": "face_recognized",
    "conditions": {
        "recognized_faces": ["John", "Jane", "Kids"]
    },
    "actions": [
        {
            "type": "send_notification",
            "title": "Welcome Home!",
            "message": "{recognized_faces} is home",
            "speak": True
        }
    ]
}
```

### 3. Package Delivery Monitoring

```python
# Monitor for delivery vehicles
delivery_rule = {
    "name": "Delivery Detection",
    "event_type": "motion_detected",
    "conditions": {
        "detected_objects": ["truck", "car", "person"],
        "time_range": [9, 17]
    },
    "actions": [
        {
            "type": "send_notification",
            "title": "Delivery Alert",
            "message": "Possible delivery detected"
        },
        {"type": "record_video", "duration": 60}
    ]
}
```

### 4. Privacy Mode Scheduling

```python
# Automatic privacy during personal hours
privacy_schedule = {
    "privacy_hours": [22, 23, 0, 1, 2, 3, 4, 5, 6]  # 10 PM - 6 AM
}

# Via API
requests.post('http://localhost:5000/camera/cp_plus_main/privacy/schedule', 
              json=privacy_schedule)
```

## Troubleshooting

### Camera Not Found

1. **Check Network Connection**
   ```bash
   ping 192.168.1.100  # Replace with your camera's IP
   ```

2. **Manual Discovery**
   ```python
   # Use camera's IP directly if auto-discovery fails
   camera_ip = "192.168.1.100"
   credentials = CameraCredentials(
       username="admin",
       password="admin", 
       ip_address=camera_ip
   )
   ```

3. **Common Default Credentials**
   - Username: `admin`, Password: `admin`
   - Username: `admin`, Password: `` (empty)
   - Username: `admin`, Password: `123456`

### Connection Issues

1. **Check Firewall**
   - Ensure ports 80, 554, 8080 are accessible
   - Add firewall exceptions if needed

2. **Network Configuration**
   ```bash
   # Check if camera is accessible
   curl -u admin:admin http://192.168.1.100/
   ```

3. **RTSP Stream Test**
   ```bash
   ffplay rtsp://admin:admin@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
   ```

### AI Vision Issues

1. **Model Loading**
   - First run will download AI models (may take time)
   - Ensure good internet connection

2. **Performance**
   - Reduce confidence thresholds for better detection
   - Use GPU acceleration if available

3. **Face Recognition Setup**
   ```python
   # Add known faces
   face_recognizer.enroll_identity("John", [face_image1, face_image2])
   ```

## Advanced Features

### Custom Automation Actions

Extend the automation system with custom actions:

```python
def custom_action(action_config, event_data):
    # Your custom logic here
    webhook_url = action_config.get('webhook_url')
    requests.post(webhook_url, json=event_data)

# Register custom action
automation_engine.actions['custom_webhook'] = custom_action
```

### Integration with Home Assistant

```python
# Send events to Home Assistant
ha_webhook_action = {
    "type": "send_webhook",
    "url": "http://homeassistant.local:8123/api/webhook/camera_event",
    "data": {
        "entity_id": "camera.cp_plus_main",
        "event_type": "{event_type}",
        "timestamp": "{timestamp}"
    }
}
```

### Cloud Recording Integration

```python
# Save recordings to cloud storage
def upload_to_cloud(file_path):
    # AWS S3 example
    import boto3
    s3 = boto3.client('s3')
    s3.upload_file(file_path, 'my-camera-bucket', 
                   f'recordings/{datetime.now().isoformat()}.mp4')
```

## API Client Examples

### Python Client

```python
import requests
import base64

# Object detection
def detect_objects(image_path):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    response = requests.post('http://localhost:5000/vision/detect', 
                           json={'image': f'data:image/jpeg;base64,{image_data}'})
    return response.json()

# Camera control
def control_camera(camera_id, action, **kwargs):
    payload = {'action': action, **kwargs}
    response = requests.post(f'http://localhost:5000/camera/{camera_id}/control',
                           json=payload)
    return response.json()

# Usage
results = detect_objects('test_image.jpg')
control_camera('cp_plus_main', 'ptz_control', direction='left', speed=3)
```

### JavaScript Client

```javascript
// Object detection
async function detectObjects(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await fetch('/vision/detect', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// WebSocket streaming
const socket = io();
socket.on('vision_frame', (data) => {
    const img = document.getElementById('stream');
    img.src = data.image;
    
    // Display detections
    console.log('Detections:', data.vision_data.detections);
});
```

## Security Considerations

1. **Camera Credentials**
   - Change default passwords immediately
   - Use strong, unique passwords
   - Enable camera firmware updates

2. **Network Security**
   - Use VPN for remote access
   - Segment camera network
   - Regular security audits

3. **Data Privacy**
   - Local processing (no cloud dependency)
   - Encrypted storage for face recognition data
   - Privacy mode scheduling

4. **API Security**
   - Use HTTPS in production
   - Implement authentication
   - Rate limiting

## Performance Optimization

1. **Hardware Requirements**
   - GPU acceleration recommended for AI processing
   - Minimum 8GB RAM for face recognition
   - SSD storage for video recording

2. **Configuration Tuning**
   ```python
   # Optimize for performance
   config = {
       'detection_interval': 0.5,  # Process every 0.5 seconds
       'confidence_threshold': 0.6,  # Higher threshold = fewer false positives
       'max_detections': 10  # Limit detections per frame
   }
   ```

3. **Resource Monitoring**
   - Monitor CPU/GPU usage
   - Adjust processing intervals based on load
   - Use frame skipping during high load

## Support

For issues and questions:

1. Check the logs in the `camera_data/logs` directory
2. Run the setup wizard again: `python setup_camera_integration.py`
3. Test individual components in isolation
4. Check camera manufacturer documentation for CP Plus CP-E305A

## License

This camera integration system is part of the Omni-Dev Agent project and follows the same licensing terms.
