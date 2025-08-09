

from flask import Flask, jsonify, Response
import sys
import os

# Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.camera_integration.camera_manager import CameraManager

app = Flask(__name__)

# Initialize the camera manager
camera_manager = CameraManager()

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')

@app.route('/api/widgets', methods=['GET'])
def get_widgets():
    """Return a list of available widgets."""
    # For now, we will hardcode the camera widget
    widgets = [
        {
            'id': 'camera_widget_1',
            'type': 'camera',
            'title': 'Live Camera Feed',
            'source': '/api/camera/stream/cp_plus_main' 
        }
    ]
    return jsonify(widgets)

def gen(camera_id):
    """Video streaming generator function."""
    
    # Get the video capture object from the camera manager
    video_capture = camera_manager.cameras.get(camera_id)
    if not video_capture:
        return

    while True:
        frame = video_capture.get_frame()
        if frame is None:
            continue
        
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route('/api/camera/stream/<camera_id>')
def camera_stream(camera_id):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

