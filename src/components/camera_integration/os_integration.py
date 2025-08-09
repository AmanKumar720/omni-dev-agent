# components/camera_integration/os_integration.py

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import requests
import threading

# Platform-specific imports
if sys.platform == "win32":
    import win10toast
    import winsound
    try:
        import pyttsx3
    except ImportError:
        pyttsx3 = None
elif sys.platform == "darwin":
    import pync
elif sys.platform.startswith("linux"):
    import notify2

logger = logging.getLogger(__name__)

class NotificationManager:
    """Cross-platform notification system for camera events"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.NotificationManager")
        self.tts_engine = None
        self._setup_notifications()
        self._setup_tts()
    
    def _setup_notifications(self):
        """Setup platform-specific notifications"""
        try:
            if sys.platform == "win32":
                self.toaster = win10toast.ToastNotifier()
            elif sys.platform.startswith("linux"):
                notify2.init("Camera System")
        except Exception as e:
            self.logger.warning(f"Failed to setup notifications: {e}")
    
    def _setup_tts(self):
        """Setup text-to-speech engine"""
        try:
            if pyttsx3:
                self.tts_engine = pyttsx3.init()
                # Configure voice settings
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.8)
        except Exception as e:
            self.logger.warning(f"Failed to setup TTS: {e}")
    
    def send_notification(self, title: str, message: str, image_path: Optional[str] = None, 
                         priority: str = "normal", sound: bool = True, speak: bool = False):
        """Send cross-platform notification"""
        try:
            if sys.platform == "win32":
                self._send_windows_notification(title, message, image_path, sound)
            elif sys.platform == "darwin":
                self._send_macos_notification(title, message, sound)
            elif sys.platform.startswith("linux"):
                self._send_linux_notification(title, message, image_path)
            
            if speak and self.tts_engine:
                self._speak_notification(f"{title}. {message}")
                
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def _send_windows_notification(self, title: str, message: str, image_path: Optional[str], sound: bool):
        """Send Windows toast notification"""
        try:
            # Play sound if requested
            if sound:
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
            
            # Send toast notification
            self.toaster.show_toast(
                title=title,
                msg=message,
                icon_path=image_path if image_path and os.path.exists(image_path) else None,
                duration=10,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Windows notification failed: {e}")
    
    def _send_macos_notification(self, title: str, message: str, sound: bool):
        """Send macOS notification"""
        try:
            pync.notify(
                message,
                title=title,
                sound="Glass" if sound else None
            )
        except Exception as e:
            self.logger.error(f"macOS notification failed: {e}")
    
    def _send_linux_notification(self, title: str, message: str, image_path: Optional[str]):
        """Send Linux desktop notification"""
        try:
            notification = notify2.Notification(title, message)
            if image_path and os.path.exists(image_path):
                notification.set_icon_from_pixbuf(image_path)
            notification.show()
        except Exception as e:
            self.logger.error(f"Linux notification failed: {e}")
    
    def _speak_notification(self, text: str):
        """Speak notification using TTS"""
        try:
            if self.tts_engine:
                def speak():
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
                # Run TTS in separate thread to avoid blocking
                thread = threading.Thread(target=speak, daemon=True)
                thread.start()
        except Exception as e:
            self.logger.error(f"TTS failed: {e}")

class EmailNotifier:
    """Email notification system for camera alerts"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.logger = logging.getLogger(f"{__name__}.EmailNotifier")
    
    def send_email(self, to_email: str, subject: str, body: str, 
                   attachments: Optional[List[str]] = None):
        """Send email notification with optional attachments"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                                img = MIMEImage(f.read())
                                img.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
                                msg.attach(img)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent to {to_email}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

class WebhookNotifier:
    """Webhook notification system for integration with external services"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WebhookNotifier")
    
    def send_webhook(self, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
        """Send webhook notification"""
        try:
            if headers is None:
                headers = {'Content-Type': 'application/json'}
            
            response = requests.post(url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook sent to {url}")
            
        except Exception as e:
            self.logger.error(f"Webhook failed: {e}")

class AutomationEngine:
    """Automation engine for camera-triggered actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AutomationEngine")
        self.rules: List[Dict[str, Any]] = []
        self.actions: Dict[str, Callable] = {
            'send_notification': self._action_send_notification,
            'send_email': self._action_send_email,
            'send_webhook': self._action_send_webhook,
            'run_command': self._action_run_command,
            'control_lights': self._action_control_lights,
            'sound_alarm': self._action_sound_alarm,
            'record_video': self._action_record_video
        }
        
        self.notification_manager = NotificationManager()
        self.email_notifier = None
        self.webhook_notifier = WebhookNotifier()
    
    def setup_email_notifier(self, smtp_server: str, smtp_port: int, username: str, password: str):
        """Setup email notifications"""
        self.email_notifier = EmailNotifier(smtp_server, smtp_port, username, password)
    
    def add_automation_rule(self, rule: Dict[str, Any]):
        """Add automation rule"""
        self.rules.append(rule)
        self.logger.info(f"Added automation rule: {rule.get('name', 'Unnamed')}")
    
    def process_event(self, event_type: str, event_data: Dict[str, Any]):
        """Process camera event and trigger automation rules"""
        try:
            for rule in self.rules:
                if self._rule_matches(rule, event_type, event_data):
                    self.logger.info(f"Triggering rule: {rule.get('name', 'Unnamed')}")
                    self._execute_actions(rule.get('actions', []), event_data)
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
    
    def _rule_matches(self, rule: Dict[str, Any], event_type: str, event_data: Dict[str, Any]) -> bool:
        """Check if rule conditions match the event"""
        try:
            # Check event type
            if rule.get('event_type') and rule['event_type'] != event_type:
                return False
            
            # Check conditions
            conditions = rule.get('conditions', {})
            
            # Time-based conditions
            if 'time_range' in conditions:
                current_hour = datetime.now().hour
                start_hour, end_hour = conditions['time_range']
                if not (start_hour <= current_hour <= end_hour):
                    return False
            
            # Camera-specific conditions
            if 'camera_id' in conditions:
                if event_data.get('camera_id') != conditions['camera_id']:
                    return False
            
            # Object detection conditions
            if 'detected_objects' in conditions:
                required_objects = conditions['detected_objects']
                detected_objects = [obj['class_name'] for obj in event_data.get('objects_detected', [])]
                if not any(obj in detected_objects for obj in required_objects):
                    return False
            
            # Confidence threshold conditions
            if 'min_confidence' in conditions:
                max_confidence = max([obj['confidence'] for obj in event_data.get('objects_detected', [])], default=0)
                if max_confidence < conditions['min_confidence']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule conditions: {e}")
            return False
    
    def _execute_actions(self, actions: List[Dict[str, Any]], event_data: Dict[str, Any]):
        """Execute automation actions"""
        for action in actions:
            try:
                action_type = action.get('type')
                if action_type in self.actions:
                    self.actions[action_type](action, event_data)
                else:
                    self.logger.warning(f"Unknown action type: {action_type}")
            except Exception as e:
                self.logger.error(f"Error executing action {action.get('type')}: {e}")
    
    def _action_send_notification(self, action: Dict[str, Any], event_data: Dict[str, Any]):
        """Send desktop notification"""
        title = action.get('title', 'Camera Alert')
        message = action.get('message', f"Motion detected on {event_data.get('camera_id')}")
        sound = action.get('sound', True)
        speak = action.get('speak', False)
        
        # Format message with event data
        message = message.format(**event_data)
        
        self.notification_manager.send_notification(
            title=title,
            message=message,
            image_path=event_data.get('image_path'),
            sound=sound,
            speak=speak
        )
    
    def _action_send_email(self, action: Dict[str, Any], event_data: Dict[str, Any]):
        """Send email notification"""
        if not self.email_notifier:
            self.logger.warning("Email notifier not configured")
            return
        
        to_email = action.get('to_email')
        subject = action.get('subject', 'Camera Alert')
        body = action.get('body', f"Motion detected on {event_data.get('camera_id')}")
        
        # Format with event data
        subject = subject.format(**event_data)
        body = body.format(**event_data)
        
        attachments = [event_data.get('image_path')] if event_data.get('image_path') else None
        
        self.email_notifier.send_email(to_email, subject, body, attachments)
    
    def _action_send_webhook(self, action: Dict[str, Any], event_data: Dict[str, Any]):
        """Send webhook notification"""
        url = action.get('url')
        data = action.get('data', {})
        
        # Merge event data
        webhook_data = {**data, **event_data}
        
        self.webhook_notifier.send_webhook(url, webhook_data)
    
    def _action_run_command(self, action: Dict[str, Any], event_data: Dict[str, Any]):
        """Run system command"""
        command = action.get('command')
        if not command:
            return
        
        # Format command with event data
        command = command.format(**event_data)
        
        try:
            subprocess.run(command, shell=True, check=True)
            self.logger.info(f"Command executed: {command}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
    
    def _action_control_lights(self, action: Dict[str, Any], event_data: Dict[str, Any]):
        """Control smart lights (placeholder for integration)"""
        # This would integrate with smart home systems like Home Assistant, Philips Hue, etc.
        light_action = action.get('action', 'turn_on')  # turn_on, turn_off, flash
        brightness = action.get('brightness', 100)
        color = action.get('color', 'white')
        
        self.logger.info(f"Light control: {light_action}, brightness: {brightness}, color: {color}")
        
        # TODO: Integrate with actual smart light systems
        # Examples:
        # - Send HTTP request to Home Assistant
        # - Use Philips Hue API
        # - Control MQTT-based lights
    
    def _action_sound_alarm(self, action: Dict[str, Any], event_data: Dict[str, Any]):
        """Sound alarm/siren"""
        duration = action.get('duration', 5)  # seconds
        
        if sys.platform == "win32":
            try:
                # Play system alarm sound
                for _ in range(duration):
                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
                    threading.Event().wait(1)
            except Exception as e:
                self.logger.error(f"Failed to sound alarm: {e}")
    
    def _action_record_video(self, action: Dict[str, Any], event_data: Dict[str, Any]):
        """Start video recording"""
        duration = action.get('duration', 30)  # seconds
        camera_id = event_data.get('camera_id')
        
        # This would trigger the camera manager to start recording
        self.logger.info(f"Starting video recording for camera {camera_id} for {duration} seconds")
        
        # TODO: Integrate with camera manager to start recording

class OSIntegration:
    """Main OS integration class combining all notification and automation features"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.OSIntegration")
        self.notification_manager = NotificationManager()
        self.automation_engine = AutomationEngine()
        
        # Load configuration
        self.config_path = Path("camera_config.json")
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Setup email if configured
                email_config = config.get('email')
                if email_config:
                    self.automation_engine.setup_email_notifier(
                        smtp_server=email_config['smtp_server'],
                        smtp_port=email_config['smtp_port'],
                        username=email_config['username'],
                        password=email_config['password']
                    )
                
                # Load automation rules
                rules = config.get('automation_rules', [])
                for rule in rules:
                    self.automation_engine.add_automation_rule(rule)
                
                self.logger.info("Configuration loaded successfully")
        
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}")
    
    def save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def handle_motion_event(self, motion_event):
        """Handle motion detection event"""
        event_data = {
            'event_type': 'motion_detected',
            'camera_id': motion_event.camera_id,
            'timestamp': motion_event.timestamp.isoformat(),
            'confidence': motion_event.confidence,
            'bbox': motion_event.bbox,
            'objects_detected': motion_event.objects_detected,
            'image_path': getattr(motion_event, 'image_path', None)
        }
        
        self.automation_engine.process_event('motion_detected', event_data)
    
    def handle_face_recognition_event(self, camera_id: str, recognized_faces: List[Dict[str, Any]], image_path: str):
        """Handle face recognition event"""
        event_data = {
            'event_type': 'face_recognized',
            'camera_id': camera_id,
            'timestamp': datetime.now().isoformat(),
            'recognized_faces': recognized_faces,
            'image_path': image_path
        }
        
        self.automation_engine.process_event('face_recognized', event_data)
    
    def send_test_notification(self, title: str = "Test Notification", message: str = "Camera system is working"):
        """Send test notification"""
        self.notification_manager.send_notification(title, message, sound=True, speak=True)

# Example configuration for automation rules
EXAMPLE_AUTOMATION_CONFIG = {
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password"
    },
    "automation_rules": [
        {
            "name": "Motion Alert - Daytime",
            "event_type": "motion_detected",
            "conditions": {
                "time_range": [8, 18],  # 8 AM to 6 PM
                "detected_objects": ["person"],
                "min_confidence": 0.7
            },
            "actions": [
                {
                    "type": "send_notification",
                    "title": "Person Detected",
                    "message": "Person detected on {camera_id} at {timestamp}",
                    "sound": True,
                    "speak": True
                },
                {
                    "type": "send_email",
                    "to_email": "alerts@yourdomain.com",
                    "subject": "Security Alert - Person Detected",
                    "body": "A person was detected on camera {camera_id} at {timestamp}"
                }
            ]
        },
        {
            "name": "Motion Alert - Nighttime",
            "event_type": "motion_detected",
            "conditions": {
                "time_range": [19, 7]  # 7 PM to 7 AM (next day)
            },
            "actions": [
                {
                    "type": "send_notification",
                    "title": "Night Motion Alert",
                    "message": "Motion detected at night on {camera_id}",
                    "sound": True,
                    "speak": True
                },
                {
                    "type": "control_lights",
                    "action": "flash",
                    "brightness": 100
                },
                {
                    "type": "sound_alarm",
                    "duration": 3
                }
            ]
        },
        {
            "name": "Known Person Recognition",
            "event_type": "face_recognized",
            "conditions": {
                "recognized_faces": ["John", "Jane"]  # Known people
            },
            "actions": [
                {
                    "type": "send_notification",
                    "title": "Welcome Home",
                    "message": "Welcome home! {recognized_faces} detected",
                    "sound": False,
                    "speak": True
                }
            ]
        }
    ]
}

if __name__ == "__main__":
    # Test the OS integration
    logging.basicConfig(level=logging.INFO)
    
    os_integration = OSIntegration()
    
    # Send test notification
    os_integration.send_test_notification()
    
    # Save example configuration
    os_integration.save_configuration(EXAMPLE_AUTOMATION_CONFIG)
