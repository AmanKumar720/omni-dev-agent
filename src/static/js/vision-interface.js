class VisionInterface {
    constructor() {
        this.socket = null;
        this.canvas = null;
        this.ctx = null;
        this.isStreaming = false;
        this.currentMode = 'live';
        
        // Performance tracking
        this.fps = 0;
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        this.processingTimes = [];
        
        // Statistics
        this.stats = {
            totalObjects: 0,
            totalFaces: 0,
            textBlocks: 0,
            avgProcessingTime: 0
        };
        
        // Current thresholds
        this.thresholds = {
            confidence: 0.5,
            face: 0.7,
            ocr: 0.6,
            topK: 5
        };
        
        this.init();
    }
    
    init() {
        this.initElements();
        this.initSocket();
        this.initEventListeners();
        this.initCanvas();
        this.initDragAndDrop();
    }
    
    initElements() {
        // Canvas
        this.canvas = document.getElementById('visionCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Controls
        this.startStreamBtn = document.getElementById('startStream');
        this.stopStreamBtn = document.getElementById('stopStream');
        this.processUploadBtn = document.getElementById('processUpload');
        this.imageUploadInput = document.getElementById('imageUpload');
        
        // Mode controls
        this.liveModeRadio = document.getElementById('liveMode');
        this.uploadModeRadio = document.getElementById('uploadMode');
        this.uploadSection = document.getElementById('uploadSection');
        this.cameraSection = document.getElementById('cameraSection');
        
        // Threshold controls
        this.confidenceThreshold = document.getElementById('confidenceThreshold');
        this.faceThreshold = document.getElementById('faceThreshold');
        this.ocrConfidence = document.getElementById('ocrConfidence');
        this.topKSlider = document.getElementById('topK');
        
        // Status elements
        this.connectionStatus = document.getElementById('connectionStatus');
        this.fpsCounter = document.getElementById('fpsCounter');
        this.processingStatus = document.getElementById('processingStatus');
        this.resolution = document.getElementById('resolution');
        
        // Analysis panels
        this.detectionsList = document.getElementById('detectionsList');
        this.facesList = document.getElementById('facesList');
        this.ocrResults = document.getElementById('ocrResults');
        
        // Statistics
        this.totalObjectsEl = document.getElementById('totalObjects');
        this.totalFacesEl = document.getElementById('totalFaces');
        this.textBlocksEl = document.getElementById('textBlocks');
        this.avgProcessingTimeEl = document.getElementById('avgProcessingTime');
        
        // Other controls
        this.saveSnapshotBtn = document.getElementById('saveSnapshot');
        this.fullscreenToggleBtn = document.getElementById('fullscreenToggle');
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }
    
    initSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.updateConnectionStatus(true);
            this.showToast('Connected', 'Successfully connected to vision server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            this.updateConnectionStatus(false);
            this.showToast('Disconnected', 'Lost connection to vision server', 'warning');
        });
        
        this.socket.on('vision_frame', (data) => {
            this.handleVisionFrame(data);
        });
        
        this.socket.on('stream_status', (data) => {
            console.log('Stream status:', data);
        });
        
        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this.showToast('Error', `Connection error: ${error}`, 'error');
        });
    }
    
    initEventListeners() {
        // Mode switching
        this.liveModeRadio.addEventListener('change', () => this.switchMode('live'));
        this.uploadModeRadio.addEventListener('change', () => this.switchMode('upload'));
        
        // Stream controls
        this.startStreamBtn.addEventListener('click', () => this.startStream());
        this.stopStreamBtn.addEventListener('click', () => this.stopStream());
        
        // Upload processing
        this.processUploadBtn.addEventListener('click', () => this.processUploadedImage());
        this.imageUploadInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Threshold controls
        this.confidenceThreshold.addEventListener('input', (e) => this.updateThreshold('confidence', e.target.value));
        this.faceThreshold.addEventListener('input', (e) => this.updateThreshold('face', e.target.value));
        this.ocrConfidence.addEventListener('input', (e) => this.updateThreshold('ocr', e.target.value));
        this.topKSlider.addEventListener('input', (e) => this.updateThreshold('topK', e.target.value));
        
        // Utility controls
        this.saveSnapshotBtn.addEventListener('click', () => this.saveSnapshot());
        this.fullscreenToggleBtn.addEventListener('click', () => this.toggleFullscreen());
        
        // Processing option checkboxes
        document.getElementById('enableDetection').addEventListener('change', () => this.updateProcessingOptions());
        document.getElementById('enableFaceRecognition').addEventListener('change', () => this.updateProcessingOptions());
        document.getElementById('enableOCR').addEventListener('change', () => this.updateProcessingOptions());
        document.getElementById('enableClassification').addEventListener('change', () => this.updateProcessingOptions());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }
    
    initCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Initial canvas setup
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Vision Interface Ready', this.canvas.width / 2, this.canvas.height / 2);
    }
    
    initDragAndDrop() {
        const displayContainer = document.getElementById('displayContainer');
        
        displayContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            displayContainer.classList.add('drag-over');
        });
        
        displayContainer.addEventListener('dragleave', (e) => {
            e.preventDefault();
            displayContainer.classList.remove('drag-over');
        });
        
        displayContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            displayContainer.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.processDroppedFile(files[0]);
            }
        });
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const containerRect = container.getBoundingClientRect();
        
        // Set canvas size to container size
        this.canvas.width = containerRect.width;
        this.canvas.height = Math.max(400, containerRect.height);
        
        this.updateResolutionDisplay();
    }
    
    updateConnectionStatus(connected) {
        const icon = this.connectionStatus.querySelector('i');
        if (connected) {
            icon.className = 'fas fa-circle text-success';
            this.connectionStatus.innerHTML = '<i class="fas fa-circle text-success"></i> Connected';
        } else {
            icon.className = 'fas fa-circle text-danger';
            this.connectionStatus.innerHTML = '<i class="fas fa-circle text-danger"></i> Disconnected';
        }
    }
    
    switchMode(mode) {
        this.currentMode = mode;
        
        if (mode === 'live') {
            this.uploadSection.style.display = 'none';
            this.cameraSection.style.display = 'block';
        } else {
            this.uploadSection.style.display = 'block';
            this.cameraSection.style.display = 'none';
            this.stopStream();
        }
    }
    
    startStream() {
        if (this.isStreaming) return;
        
        const options = this.getProcessingOptions();
        
        this.socket.emit('start_stream', {
            type: 'camera',
            options: options
        });
        
        this.isStreaming = true;
        this.startStreamBtn.disabled = true;
        this.stopStreamBtn.disabled = false;
        this.updateProcessingStatus('Streaming');
        this.hideLoadingOverlay();
        
        // Start FPS calculation
        this.startFPSCounter();
    }
    
    stopStream() {
        if (!this.isStreaming) return;
        
        this.socket.emit('stop_stream');
        this.isStreaming = false;
        this.startStreamBtn.disabled = false;
        this.stopStreamBtn.disabled = true;
        this.updateProcessingStatus('Ready');
        
        this.stopFPSCounter();
        this.clearCanvas();
    }
    
    getProcessingOptions() {
        return {
            enable_detection: document.getElementById('enableDetection').checked,
            enable_face_recognition: document.getElementById('enableFaceRecognition').checked,
            enable_ocr: document.getElementById('enableOCR').checked,
            enable_classification: document.getElementById('enableClassification').checked,
            confidence_threshold: this.thresholds.confidence,
            face_threshold: this.thresholds.face,
            ocr_confidence: this.thresholds.ocr,
            top_k: this.thresholds.topK,
            language: document.getElementById('ocrLanguage').value
        };
    }
    
    updateProcessingOptions() {
        if (this.isStreaming) {
            const options = this.getProcessingOptions();
            this.socket.emit('update_options', options);
        }
    }
    
    updateThreshold(type, value) {
        this.thresholds[type] = parseFloat(value);
        
        // Update display value
        const displayMap = {
            'confidence': 'confidenceValue',
            'face': 'faceValue',
            'ocr': 'ocrValue',
            'topK': 'topKValue'
        };
        
        const displayEl = document.getElementById(displayMap[type]);
        if (displayEl) {
            displayEl.textContent = type === 'topK' ? value : parseFloat(value).toFixed(2);
        }
        
        // Update processing options if streaming
        this.updateProcessingOptions();
    }
    
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            this.processUploadBtn.disabled = false;
        }
    }
    
    processUploadedImage() {
        const file = this.imageUploadInput.files[0];
        if (!file) return;
        
        this.processDroppedFile(file);
    }
    
    processDroppedFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Display image on canvas
                this.drawImageToCanvas(img);
                
                // Process image with all enabled options
                this.processStaticImage(e.target.result);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    drawImageToCanvas(img) {
        // Clear canvas
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Calculate scaled dimensions
        const scale = Math.min(this.canvas.width / img.width, this.canvas.height / img.height);
        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;
        
        const x = (this.canvas.width - scaledWidth) / 2;
        const y = (this.canvas.height - scaledHeight) / 2;
        
        // Draw image
        this.ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
        
        this.updateResolutionDisplay();
    }
    
    async processStaticImage(imageData) {
        this.updateProcessingStatus('Processing');
        
        const options = this.getProcessingOptions();
        const base64Data = imageData.split(',')[1];
        
        try {
            // Process with different endpoints based on enabled options
            const results = {};
            
            if (options.enable_detection) {
                results.detections = await this.callAPI('/vision/detect', {
                    image: base64Data,
                    confidence: options.confidence_threshold
                });
            }
            
            if (options.enable_face_recognition) {
                results.faces = await this.callAPI('/vision/face/identify', {
                    image: base64Data
                });
            }
            
            if (options.enable_ocr) {
                results.ocr = await this.callAPI('/vision/ocr', {
                    image: base64Data,
                    language: options.language
                });
            }
            
            if (options.enable_classification) {
                results.classification = await this.callAPI('/vision/classify', {
                    image: base64Data,
                    top_k: options.top_k
                });
            }
            
            // Update UI with results
            this.updateAnalysisResults(results);
            
        } catch (error) {
            console.error('Processing error:', error);
            this.showToast('Error', `Processing failed: ${error.message}`, 'error');
        } finally {
            this.updateProcessingStatus('Ready');
        }
    }
    
    async callAPI(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`API call failed: ${response.status}`);
        }
        
        return await response.json();
    }
    
    handleVisionFrame(data) {
        // Update FPS
        this.updateFPS();
        
        // Draw frame
        const img = new Image();
        img.onload = () => {
            this.drawImageToCanvas(img);
            
            // Draw vision overlays
            if (data.vision_data) {
                this.drawVisionOverlays(data.vision_data);
                this.updateAnalysisResults(data.vision_data);
            }
        };
        img.src = data.image;
        
        this.updateProcessingStatus('Streaming');
    }
    
    drawVisionOverlays(visionData) {
        // Draw object detections
        if (visionData.detections) {
            this.drawDetections(visionData.detections);
        }
        
        // Draw face detections
        if (visionData.faces) {
            this.drawFaces(visionData.faces);
        }
    }
    
    drawDetections(detections) {
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;
        this.ctx.font = '14px Arial';
        this.ctx.fillStyle = '#00ff00';
        
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Draw bounding box
            this.ctx.strokeRect(x1, y1, width, height);
            
            // Draw label
            const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
            const textWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = '#00ff00';
            this.ctx.fillRect(x1, y1 - 20, textWidth + 8, 18);
            this.ctx.fillStyle = '#000';
            this.ctx.fillText(label, x1 + 4, y1 - 6);
        });
    }
    
    drawFaces(faces) {
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 2;
        this.ctx.font = '14px Arial';
        
        faces.forEach(face => {
            const [x, y, width, height] = face.bbox;
            
            // Draw bounding box
            this.ctx.strokeRect(x, y, width, height);
            
            // Draw label
            const name = face.name || 'Unknown';
            const label = `${name} ${(face.confidence * 100).toFixed(1)}%`;
            const textWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = '#ff0000';
            this.ctx.fillRect(x, y - 20, textWidth + 8, 18);
            this.ctx.fillStyle = '#fff';
            this.ctx.fillText(label, x + 4, y - 6);
        });
    }
    
    updateAnalysisResults(data) {
        // Update object detections
        if (data.detections) {
            this.updateDetectionsList(data.detections);
            this.stats.totalObjects += data.detections.length;
        }
        
        // Update face recognition
        if (data.faces) {
            this.updateFacesList(data.faces);
            this.stats.totalFaces += data.faces.length;
        }
        
        // Update OCR results
        if (data.ocr || data.text) {
            this.updateOCRResults(data.ocr || { text: data.text });
            this.stats.textBlocks++;
        }
        
        // Update statistics
        this.updateStatistics();
    }
    
    updateDetectionsList(detections) {
        if (detections.length === 0) {
            this.detectionsList.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-search fa-2x mb-2"></i>
                    <p>No objects detected</p>
                </div>
            `;
            return;
        }
        
        this.detectionsList.innerHTML = detections.map(detection => {
            const confidenceClass = this.getConfidenceClass(detection.confidence);
            return `
                <div class="detection-item">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <strong>${detection.class_name}</strong>
                        <span class="detection-confidence ${confidenceClass}">
                            ${(detection.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="detection-bbox">
                        BBox: [${detection.bbox.map(x => Math.round(x)).join(', ')}]
                    </div>
                </div>
            `;
        }).join('');
    }
    
    updateFacesList(faces) {
        if (faces.length === 0) {
            this.facesList.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-user fa-2x mb-2"></i>
                    <p>No faces detected</p>
                </div>
            `;
            return;
        }
        
        this.facesList.innerHTML = faces.map(face => {
            const confidenceClass = this.getConfidenceClass(face.confidence);
            const name = face.name || 'Unknown';
            return `
                <div class="face-item">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <strong>${name}</strong>
                        <span class="face-confidence ${confidenceClass}">
                            ${(face.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="face-bbox">
                        BBox: [${face.bbox.map(x => Math.round(x)).join(', ')}]
                    </div>
                </div>
            `;
        }).join('');
    }
    
    updateOCRResults(ocr) {
        if (!ocr.text || ocr.text.trim() === '') {
            this.ocrResults.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-font fa-2x mb-2"></i>
                    <p>No text detected</p>
                </div>
            `;
            return;
        }
        
        const confidenceClass = this.getConfidenceClass(ocr.confidence || 0);
        this.ocrResults.innerHTML = `
            <div class="ocr-item">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <strong>Detected Text</strong>
                    ${ocr.confidence ? `<span class="ocr-confidence ${confidenceClass}">
                        ${(ocr.confidence * 100).toFixed(1)}%
                    </span>` : ''}
                </div>
                <div class="ocr-text">${ocr.text}</div>
                ${ocr.language ? `<small class="text-muted">Language: ${ocr.language}</small>` : ''}
            </div>
        `;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.75) return 'confidence-high';
        if (confidence >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    }
    
    updateStatistics() {
        this.totalObjectsEl.textContent = this.stats.totalObjects;
        this.totalFacesEl.textContent = this.stats.totalFaces;
        this.textBlocksEl.textContent = this.stats.textBlocks;
        
        if (this.processingTimes.length > 0) {
            const avg = this.processingTimes.reduce((a, b) => a + b, 0) / this.processingTimes.length;
            this.avgProcessingTimeEl.textContent = `${Math.round(avg)}ms`;
        }
    }
    
    startFPSCounter() {
        this.frameCount = 0;
        this.lastFrameTime = Date.now();
        
        this.fpsInterval = setInterval(() => {
            const now = Date.now();
            const elapsed = now - this.lastFrameTime;
            this.fps = Math.round((this.frameCount * 1000) / elapsed);
            this.fpsCounter.textContent = this.fps;
            this.frameCount = 0;
            this.lastFrameTime = now;
        }, 1000);
    }
    
    stopFPSCounter() {
        if (this.fpsInterval) {
            clearInterval(this.fpsInterval);
            this.fpsInterval = null;
        }
        this.fpsCounter.textContent = '0';
    }
    
    updateFPS() {
        this.frameCount++;
    }
    
    updateProcessingStatus(status) {
        this.processingStatus.textContent = status;
        this.processingStatus.className = '';
        
        switch (status.toLowerCase()) {
            case 'ready':
                this.processingStatus.classList.add('status-ready');
                break;
            case 'processing':
            case 'streaming':
                this.processingStatus.classList.add('status-processing');
                break;
            case 'error':
                this.processingStatus.classList.add('status-error');
                break;
        }
    }
    
    updateResolutionDisplay() {
        this.resolution.textContent = `${this.canvas.width}x${this.canvas.height}`;
    }
    
    saveSnapshot() {
        const link = document.createElement('a');
        link.download = `vision-snapshot-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
        link.href = this.canvas.toDataURL();
        link.click();
    }
    
    toggleFullscreen() {
        const container = document.getElementById('displayContainer');
        
        if (!document.fullscreenElement) {
            container.requestFullscreen().then(() => {
                container.classList.add('fullscreen');
                this.resizeCanvas();
                this.fullscreenToggleBtn.innerHTML = '<i class="fas fa-compress me-1"></i>Exit Fullscreen';
            });
        } else {
            document.exitFullscreen().then(() => {
                container.classList.remove('fullscreen');
                this.resizeCanvas();
                this.fullscreenToggleBtn.innerHTML = '<i class="fas fa-expand me-1"></i>Fullscreen';
            });
        }
    }
    
    clearCanvas() {
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Vision Interface Ready', this.canvas.width / 2, this.canvas.height / 2);
    }
    
    hideLoadingOverlay() {
        this.loadingOverlay.style.display = 'none';
    }
    
    showLoadingOverlay() {
        this.loadingOverlay.style.display = 'block';
    }
    
    handleKeyboard(event) {
        // Keyboard shortcuts
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 's':
                    event.preventDefault();
                    this.saveSnapshot();
                    break;
                case 'f':
                    event.preventDefault();
                    this.toggleFullscreen();
                    break;
            }
        }
        
        // Space to start/stop stream
        if (event.code === 'Space' && this.currentMode === 'live') {
            event.preventDefault();
            if (this.isStreaming) {
                this.stopStream();
            } else {
                this.startStream();
            }
        }
    }
    
    showToast(title, message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toastId = 'toast-' + Date.now();
        
        const iconMap = {
            success: 'fa-check-circle text-success',
            error: 'fa-exclamation-circle text-danger',
            warning: 'fa-exclamation-triangle text-warning',
            info: 'fa-info-circle text-info'
        };
        
        const toastHTML = `
            <div class="toast" id="${toastId}" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header">
                    <i class="fas ${iconMap[type]} me-2"></i>
                    <strong class="me-auto">${title}</strong>
                    <small class="text-muted">now</small>
                    <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHTML);
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: type === 'error' ? 8000 : 4000
        });
        
        toast.show();
        
        // Remove toast element after it's hidden
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }
}

// Initialize the vision interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.visionInterface = new VisionInterface();
});
