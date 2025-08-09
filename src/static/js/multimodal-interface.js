class MultimodalInterface {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.voiceModal = null;
        this.planModal = null;
        
        this.init();
    }
    
    init() {
        this.initElements();
        this.initSocket();
        this.initEventListeners();
        this.initModals();
    }
    
    initElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageForm = document.getElementById('messageForm');
        this.messageInput = document.getElementById('messageInput');
        
        // Buttons
        this.sendBtn = document.getElementById('sendBtn');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.imageBtn = document.getElementById('imageBtn');
        this.documentBtn = document.getElementById('documentBtn');
        this.planBtn = document.getElementById('planBtn');
        
        // File inputs
        this.imageInput = document.getElementById('imageInput');
        this.documentInput = document.getElementById('documentInput');
        
        // Voice modal elements
        this.recordingIndicator = document.getElementById('recordingIndicator');
        this.recordingStatus = document.getElementById('recordingStatus');
        this.transcriptContainer = document.getElementById('transcriptContainer');
        this.transcript = document.getElementById('transcript');
        this.stopRecordingBtn = document.getElementById('stopRecordingBtn');
        this.sendVoiceBtn = document.getElementById('sendVoiceBtn');
        
        // Plan modal elements
        this.planTitle = document.getElementById('planTitle');
        this.planDescription = document.getElementById('planDescription');
        this.planItems = document.getElementById('planItems');
        this.addPlanItemBtn = document.getElementById('addPlanItemBtn');
        this.createPlanBtn = document.getElementById('createPlanBtn');
        
        // Status elements
        this.connectionStatus = document.getElementById('connectionStatus');
    }
    
    initSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('message', (data) => {
            this.addMessage(data.message, 'agent');
        });
        
        this.socket.on('voice_response', (data) => {
            if (data.audio) {
                this.playAudio(data.audio);
            }
            if (data.message) {
                this.addMessage(data.message, 'agent');
            }
        });
        
        this.socket.on('image_analysis', (data) => {
            this.addMessage(data.analysis, 'agent', data.image_url);
        });
        
        this.socket.on('document_analysis', (data) => {
            this.addMessage(data.analysis, 'agent', null, data.document);
        });
        
        this.socket.on('plan_created', (data) => {
            this.addPlanToChat(data.plan);
        });
    }
    
    initEventListeners() {
        // Send message form
        this.messageForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendTextMessage();
        });
        
        // Voice button
        this.voiceBtn.addEventListener('click', () => {
            console.log('Voice button clicked');
            // Show the voice modal first
            if (this.voiceModal) {
                this.voiceModal.show();
                // Start recording after modal is shown
                setTimeout(() => this.startVoiceRecording(), 500);
            } else {
                console.error('Voice modal not initialized');
                alert('Voice modal not initialized. Please refresh the page.');
            }
        });
        
        // Stop recording button
        this.stopRecordingBtn.addEventListener('click', () => {
            this.stopVoiceRecording();
        });
        
        // Send voice button
        this.sendVoiceBtn.addEventListener('click', () => {
            this.sendVoiceMessage();
        });
        
        // Image button
        this.imageBtn.addEventListener('click', () => {
            this.imageInput.click();
        });
        
        // Image input change
        this.imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0]);
            }
        });
        
        // Document button
        this.documentBtn.addEventListener('click', () => {
            this.documentInput.click();
        });
        
        // Document input change
        this.documentInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleDocumentUpload(e.target.files[0]);
            }
        });
        
        // Plan button
        this.planBtn.addEventListener('click', () => {
            this.planModal.show();
        });
        
        // Add plan item button
        this.addPlanItemBtn.addEventListener('click', () => {
            this.addPlanItemInput();
        });
        
        // Create plan button
        this.createPlanBtn.addEventListener('click', () => {
            this.createPlan();
        });
        
        // Remove plan item buttons (delegated)
        this.planItems.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-item') || e.target.parentElement.classList.contains('remove-item')) {
                const button = e.target.classList.contains('remove-item') ? e.target : e.target.parentElement;
                const itemInput = button.closest('.plan-item-input');
                if (this.planItems.children.length > 1) {
                    itemInput.remove();
                }
            }
        });
    }
    
    initModals() {
        // Make sure Bootstrap is loaded before initializing modals
        if (typeof bootstrap !== 'undefined') {
            const voiceModalEl = document.getElementById('voiceModal');
            const planModalEl = document.getElementById('planModal');
            
            if (voiceModalEl) {
                this.voiceModal = new bootstrap.Modal(voiceModalEl);
                // Reset modals when hidden
                voiceModalEl.addEventListener('hidden.bs.modal', () => {
                    this.resetVoiceModal();
                });
            } else {
                console.error('Voice modal element not found');
            }
            
            if (planModalEl) {
                this.planModal = new bootstrap.Modal(planModalEl);
                planModalEl.addEventListener('hidden.bs.modal', () => {
                    this.resetPlanModal();
                });
            } else {
                console.error('Plan modal element not found');
            }
        } else {
            console.error('Bootstrap not loaded. Modal functionality will not work.');
            setTimeout(() => this.initModals(), 1000); // Try again after 1 second
        }
    }
    
    updateConnectionStatus(connected) {
        const statusIcon = this.connectionStatus.querySelector('i');
        if (connected) {
            statusIcon.className = 'fas fa-circle text-success';
            this.connectionStatus.innerHTML = `<i class="fas fa-circle text-success"></i> Connected`;
        } else {
            statusIcon.className = 'fas fa-circle text-danger';
            this.connectionStatus.innerHTML = `<i class="fas fa-circle text-danger"></i> Disconnected`;
        }
    }
    
    addMessage(content, sender, mediaUrl = null, document = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        let messageContent = `<div>${content}</div>`;
        
        // Add media preview if provided
        if (mediaUrl) {
            messageContent = `<div>${content}</div><img src="${mediaUrl}" class="media-preview" alt="Media">`;
        }
        
        // Add document preview if provided
        if (document) {
            const docIcon = this.getDocumentIcon(document.type);
            messageContent = `
                <div>${content}</div>
                <div class="document-preview">
                    <div class="document-icon"><i class="${docIcon}"></i></div>
                    <div class="document-info">
                        <p class="document-name">${document.name}</p>
                        <p class="document-size">${this.formatFileSize(document.size)}</p>
                    </div>
                </div>
            `;
        }
        
        // Add timestamp
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        messageContent += `<div class="message-time">${timeString}</div>`;
        
        messageDiv.innerHTML = messageContent;
        this.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    sendTextMessage() {
        const message = this.messageInput.value.trim();
        if (message) {
            this.addMessage(message, 'user');
            this.socket.emit('text_message', { message });
            this.messageInput.value = '';
        }
    }
    
    startVoiceRecording() {
        // First show the modal so the user sees what's happening
        this.voiceModal.show();
        
        // Check if MediaRecorder is supported
        if (!window.MediaRecorder) {
            console.error('MediaRecorder not supported in this browser');
            this.recordingStatus.textContent = 'Recording not supported in this browser';
            return;
        }
        
        // Check if getUserMedia is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('getUserMedia not supported in this browser');
            this.recordingStatus.textContent = 'Microphone access not supported in this browser';
            return;
        }
        
        // Request microphone access
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                try {
                    this.mediaRecorder = new MediaRecorder(stream);
                    this.audioChunks = [];
                    
                    this.mediaRecorder.ondataavailable = (e) => {
                        this.audioChunks.push(e.data);
                    };
                    
                    this.mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        
                        // Use speech recognition to get transcript
                        this.getTranscript(audioBlob);
                    };
                    
                    this.mediaRecorder.onerror = (event) => {
                        console.error('MediaRecorder error:', event.error);
                        this.recordingStatus.textContent = 'Recording error occurred';
                    };
                    
                    this.mediaRecorder.start();
                    this.isRecording = true;
                    this.recordingStatus.textContent = 'Recording...';
                    this.stopRecordingBtn.style.display = 'block';
                } catch (err) {
                    console.error('Error creating MediaRecorder:', err);
                    this.recordingStatus.textContent = 'Could not start recording';
                    // Stop all tracks to release the microphone
                    stream.getTracks().forEach(track => track.stop());
                }
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                this.recordingStatus.textContent = 'Microphone access denied or error';
                
                // Show more specific error messages
                if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                    this.recordingStatus.textContent = 'Microphone permission denied. Please allow microphone access.';
                } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                    this.recordingStatus.textContent = 'No microphone found. Please connect a microphone.';
                }
            });
    }
    
    stopVoiceRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.recordingStatus.textContent = 'Processing...';
            this.stopRecordingBtn.style.display = 'none';
        }
    }
    
    getTranscript(audioBlob) {
        // Check if Web Speech API is supported
        if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
            console.error('Speech recognition not supported in this browser');
            this.transcript.textContent = 'Speech recognition not supported in this browser. Please try Chrome, Edge, or Safari.';
            this.transcriptContainer.style.display = 'block';
            this.recordingStatus.textContent = 'Transcription failed';
            this.sendVoiceBtn.style.display = 'block';
            return;
        }
        
        // Use the Web Speech API for real-time transcription with multilingual support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        // Get the selected language from the language selector
        const languageSelector = document.getElementById('languageSelector');
        recognition.lang = languageSelector ? languageSelector.value : 'en-US';
        
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.continuous = false;
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.transcript.textContent = transcript;
            this.transcriptContainer.style.display = 'block';
            this.recordingStatus.textContent = 'Ready to send';
            this.sendVoiceBtn.style.display = 'block';
            
            // Show confidence level
            const confidence = Math.round(event.results[0][0].confidence * 100);
            document.getElementById('transcriptConfidence').textContent = `Confidence: ${confidence}%`;
            document.getElementById('transcriptConfidence').style.display = 'block';
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            // Fallback to sending the audio without transcript
            this.transcript.textContent = 'Could not transcribe audio. Please try again or type your message.';
            this.transcriptContainer.style.display = 'block';
            this.recordingStatus.textContent = 'Transcription failed';
            this.sendVoiceBtn.style.display = 'block';
            document.getElementById('transcriptConfidence').style.display = 'none';
        };
        
        // Start recognition with the recorded audio
        recognition.start();
    }
    
    sendVoiceMessage() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const transcript = this.transcript.textContent || 'Voice message (no transcript available)';
        
        // Get the selected language
        const languageSelector = document.getElementById('languageSelector');
        const language = languageSelector ? languageSelector.value : 'en-US';
        
        // Get the confidence level
        const confidenceElement = document.getElementById('transcriptConfidence');
        const confidenceText = confidenceElement ? confidenceElement.textContent : '';
        const confidence = confidenceText.match(/\d+/) ? confidenceText.match(/\d+/)[0] : '0';
        
        // Create a FormData object to send the audio file
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('transcript', transcript);
        formData.append('language', language);
        formData.append('confidence', confidence);
        
        // Add the message to the chat
        this.addMessage(transcript, 'user');
        
        // Update status
        this.recordingStatus.textContent = 'Sending...';
        
        // Send the audio to the server
        fetch('/api/voice', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Voice message sent successfully:', data);
            this.voiceModal.hide();
        })
        .catch(error => {
            console.error('Error sending voice message:', error);
            this.recordingStatus.textContent = 'Error sending message';
            // Still allow closing the modal
            setTimeout(() => {
                alert('Error sending voice message: ' + error.message);
            }, 100);
        });
    }
    
    resetVoiceModal() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
        }
        
        this.audioChunks = [];
        this.transcriptContainer.style.display = 'none';
        this.transcript.textContent = '';
        document.getElementById('transcriptConfidence').textContent = '';
        document.getElementById('transcriptConfidence').style.display = 'none';
        this.recordingStatus.textContent = 'Recording...';
        this.stopRecordingBtn.style.display = 'block';
        this.sendVoiceBtn.style.display = 'none';
        
        // Stop all audio tracks
        if (this.mediaRecorder && this.mediaRecorder.stream) {
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    
    handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const imageUrl = e.target.result;
            this.addMessage('Analyzing image...', 'user', imageUrl);
            
            // Create a FormData object to send the image file
            const formData = new FormData();
            formData.append('image', file);
            
            // Send the image to the server
            fetch('/api/image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Image uploaded successfully:', data);
            })
            .catch(error => {
                console.error('Error uploading image:', error);
                this.addMessage('Error analyzing image. Please try again.', 'agent');
            });
        };
        
        reader.readAsDataURL(file);
    }
    
    handleDocumentUpload(file) {
        const validTypes = ['.pdf', '.doc', '.docx', '.txt'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(fileExt)) {
            alert('Please select a valid document file (PDF, DOC, DOCX, or TXT).');
            return;
        }
        
        // Add document message to chat
        const document = {
            name: file.name,
            type: fileExt.substring(1), // Remove the dot
            size: file.size
        };
        
        this.addMessage('Analyzing document...', 'user', null, document);
        
        // Create a FormData object to send the document file
        const formData = new FormData();
        formData.append('document', file);
        
        // Send the document to the server
        fetch('/api/document', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Document uploaded successfully:', data);
        })
        .catch(error => {
            console.error('Error uploading document:', error);
            this.addMessage('Error analyzing document. Please try again.', 'agent');
        });
    }
    
    addPlanItemInput() {
        const itemInput = document.createElement('div');
        itemInput.className = 'plan-item-input mb-2 d-flex';
        itemInput.innerHTML = `
            <input type="text" class="form-control me-2" placeholder="Enter plan item">
            <button type="button" class="btn btn-outline-danger btn-sm remove-item"><i class="fas fa-times"></i></button>
        `;
        this.planItems.appendChild(itemInput);
    }
    
    createPlan() {
        const title = this.planTitle.value.trim();
        const description = this.planDescription.value.trim();
        
        if (!title) {
            alert('Please enter a plan title.');
            return;
        }
        
        const items = [];
        const itemInputs = this.planItems.querySelectorAll('input');
        
        itemInputs.forEach(input => {
            const itemText = input.value.trim();
            if (itemText) {
                items.push({
                    text: itemText,
                    completed: false
                });
            }
        });
        
        if (items.length === 0) {
            alert('Please add at least one plan item.');
            return;
        }
        
        const plan = {
            title,
            description,
            items,
            created_at: new Date().toISOString()
        };
        
        // Send plan to server
        this.socket.emit('create_plan', { plan });
        
        // Add plan to chat
        this.addPlanToChat(plan);
        
        // Close modal
        this.planModal.hide();
    }
    
    addPlanToChat(plan) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message agent-message';
        
        let itemsHtml = '';
        plan.items.forEach(item => {
            const completedClass = item.completed ? 'completed' : '';
            const checkedAttr = item.completed ? 'checked' : '';
            itemsHtml += `
                <div class="plan-item ${completedClass}">
                    <div class="form-check">
                        <input class="form-check-input plan-item-checkbox" type="checkbox" ${checkedAttr}>
                        <label class="form-check-label">${item.text}</label>
                    </div>
                </div>
            `;
        });
        
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div>Here's your plan:</div>
            <div class="card mt-2 mb-2">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">${plan.title}</h5>
                </div>
                <div class="card-body">
                    ${plan.description ? `<p>${plan.description}</p>` : ''}
                    <div class="plan-items mt-3">
                        ${itemsHtml}
                    </div>
                </div>
            </div>
            <div class="message-time">${timeString}</div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        
        // Add event listeners to checkboxes
        const checkboxes = messageDiv.querySelectorAll('.plan-item-checkbox');
        checkboxes.forEach((checkbox, index) => {
            checkbox.addEventListener('change', () => {
                const planItem = checkbox.closest('.plan-item');
                if (checkbox.checked) {
                    planItem.classList.add('completed');
                    plan.items[index].completed = true;
                } else {
                    planItem.classList.remove('completed');
                    plan.items[index].completed = false;
                }
                
                // Update plan on server
                this.socket.emit('update_plan', { plan });
            });
        });
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    resetPlanModal() {
        this.planTitle.value = '';
        this.planDescription.value = '';
        
        // Remove all plan items except the first one
        while (this.planItems.children.length > 1) {
            this.planItems.removeChild(this.planItems.lastChild);
        }
        
        // Clear the first input
        const firstInput = this.planItems.querySelector('input');
        if (firstInput) {
            firstInput.value = '';
        }
    }
    
    playAudio(audioData) {
        // In a real implementation, this would play the audio response
        // For demo purposes, we'll just log it
        console.log('Playing audio response');
    }
    
    getDocumentIcon(type) {
        switch (type.toLowerCase()) {
            case 'pdf':
                return 'fas fa-file-pdf';
            case 'doc':
            case 'docx':
                return 'fas fa-file-word';
            case 'txt':
                return 'fas fa-file-alt';
            default:
                return 'fas fa-file';
        }
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the interface when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.multimodalInterface = new MultimodalInterface();
});