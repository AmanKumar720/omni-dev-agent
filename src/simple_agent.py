from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import json
import threading
import os
import base64
from datetime import datetime
from typing import Dict, Any, Optional
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'omni-dev-agent-key'
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def hello_world():
    return "Omni-Dev Agent - Ready"

@app.route("/multimodal")
def multimodal_interface():
    """Serve the multimodal interface web UI"""
    return render_template('multimodal.html')

@app.route("/ui")
def agent_interface():
    """Serve the agent interface web UI"""
    return render_template('index.html')

@app.route("/demo")
def demo_page():
    """Serve the demo page"""
    return render_template('demo.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'active',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'components': [
            {'name': 'core', 'status': 'active'},
            {'name': 'web_interface', 'status': 'active'}
        ]
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('status', {'status': 'connected', 'message': 'Welcome to Omni-Dev Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('command')
def handle_command(data):
    """Handle command from client"""
    command = data.get('command', '')
    params = data.get('params', {})
    
    print(f"Received command: {command} with params: {params}")
    
    # Process command
    result = {
        'command': command,
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'result': f"Processed command: {command}"
    }
    
    emit('command_result', result)

@socketio.on('text_message')
def handle_text_message(data):
    """Handle text message from client"""
    message = data.get('message', '')
    print(f"Received text message: {message}")
    
    # Process the message and generate a response
    response = f"You said: {message}. This is a simulated response from the agent."
    
    # Send response back to client
    emit('message', {'message': response})

@socketio.on('create_plan')
def handle_create_plan(data):
    """Handle plan creation from client"""
    plan = data.get('plan', {})
    print(f"Received plan creation request: {plan}")
    
    # In a real implementation, you would save the plan to a database
    # For now, we'll just echo it back with a confirmation message
    emit('message', {'message': f"Plan '{plan.get('title')}' created successfully!"})

@socketio.on('update_plan')
def handle_update_plan(data):
    """Handle plan update from client"""
    plan = data.get('plan', {})
    print(f"Received plan update request: {plan}")
    
    # In a real implementation, you would update the plan in a database
    # For now, we'll just acknowledge the update
    emit('message', {'message': f"Plan '{plan.get('title')}' updated successfully!"})

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'documents'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'audio'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_document_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DOCUMENT_EXTENSIONS

@app.route('/api/image', methods=['POST'])
def process_image():
    """Process uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_image_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', filename)
        file.save(file_path)
        
        # In a real implementation, you would process the image here
        # For now, we'll just simulate an analysis response
        analysis = "I can see an image. It appears to contain some visual content."
        
        # Send the analysis to the client via Socket.IO
        image_url = f"/uploads/images/{filename}"
        socketio.emit('image_analysis', {
            'analysis': analysis,
            'image_url': image_url
        }, room=request.sid)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': file_path,
            'url': image_url
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/document', methods=['POST'])
def process_document():
    """Process uploaded document"""
    if 'document' not in request.files:
        return jsonify({'error': 'No document part'}), 400
        
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_document_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'documents', filename)
        file.save(file_path)
        
        # In a real implementation, you would process the document here
        # For now, we'll just simulate an analysis response
        analysis = f"I've analyzed the document '{filename}'. It appears to contain textual content."
        
        # Send the analysis to the client via Socket.IO
        document_info = {
            'name': filename,
            'type': filename.rsplit('.', 1)[1].lower(),
            'size': os.path.getsize(file_path)
        }
        
        socketio.emit('document_analysis', {
            'analysis': analysis,
            'document': document_info
        }, room=request.sid)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': file_path
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/voice', methods=['POST'])
def process_voice():
    """Process voice recording"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio part'}), 400
        
    file = request.files['audio']
    transcript = request.form.get('transcript', '')
    language = request.form.get('language', 'en-US')
    confidence = request.form.get('confidence', '0')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', filename)
    file.save(file_path)
    
    # Check if the transcript mentions Janhvi Kapoor
    if 'janhvi' in transcript.lower() or 'kapoor' in transcript.lower() or 'janhvi kapoor' in transcript.lower():
        response = "Janhvi Kapoor is an Indian actress who works in Hindi films. She is the daughter of film producer Boney Kapoor and the late legendary actress Sridevi. She made her acting debut in 2018 with the romantic drama film 'Dhadak' opposite Ishaan Khatter. Some of her notable films include 'Gunjan Saxena: The Kargil Girl', 'Roohi', 'Good Luck Jerry', and 'Mili'. She is known for her versatile acting skills and fashion sense."
    else:
        # Generate a simple response based on the transcript and language
        lang_code = language.split('-')[0] if '-' in language else language
        
        if lang_code == 'en':
            response = f"I heard: {transcript}"
        elif lang_code == 'hi':
            response = f"मैंने सुना: {transcript}"
        elif lang_code == 'ta':
            response = f"நான் உங்கள் செய்தியைப் பெற்றேன்: {transcript}"
        elif lang_code == 'te':
            response = f"నేను మీ సందేశాన్ని స్వీకరించాను: {transcript}"
        else:
            response = f"I heard ({lang_code}): {transcript}"
        
        # Add confidence information if available
        if confidence and confidence != '0':
            response += f" (Confidence: {confidence}%)"
    
    # Send the response to the client via Socket.IO
    socketio.emit('voice_response', {
        'message': response,
        'audio': None  # In a real implementation, this could be a base64-encoded audio response
    }, room=request.sid)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'path': file_path,
        'transcript': transcript,
        'response': response
    })

@app.route('/uploads/<path:folder>/<path:filename>')
def uploaded_file(folder, filename):
    """Serve uploaded files"""
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], folder), filename)

if __name__ == "__main__":
    print("Starting Omni-Dev Agent...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)