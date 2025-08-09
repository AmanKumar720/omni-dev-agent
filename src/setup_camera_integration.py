#!/usr/bin/env python3
"""
CP Plus Camera Integration Setup Script

This script helps you discover, configure, and integrate your CP Plus CP-E305A camera
with the Omni-Dev Agent system, including AI vision capabilities and OS notifications.
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from components.camera_integration.camera_manager import CameraManager, CameraConfig, CameraCredentials, CameraCapabilities
    from components.camera_integration.os_integration import OSIntegration, EXAMPLE_AUTOMATION_CONFIG
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this from the src directory and all dependencies are installed.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraSetupWizard:
    """Interactive setup wizard for camera integration"""
    
    def __init__(self):
        self.camera_manager = CameraManager()
        self.os_integration = OSIntegration()
        self.config = {}
        
    def run_setup(self):
        """Run the complete setup wizard"""
        print("\n" + "="*60)
        print("   CP Plus Camera Integration Setup Wizard")
        print("="*60)
        print()
        print("This wizard will help you:")
        print("• Discover your CP Plus CP-E305A camera on the network")
        print("• Configure camera credentials and settings")
        print("• Set up AI vision capabilities")
        print("• Configure automated notifications and responses")
        print("• Test the complete integration")
        print()
        
        try:
            # Step 1: Network Discovery
            print("Step 1: Discovering cameras on your network...")
            cameras = self.discover_cameras()
            
            # Step 2: Camera Configuration
            print("\nStep 2: Camera configuration...")
            camera_config = self.configure_camera(cameras)
            
            # Step 3: AI Vision Setup
            print("\nStep 3: AI Vision setup...")
            self.setup_ai_vision()
            
            # Step 4: Notifications and Automation
            print("\nStep 4: Notifications and automation setup...")
            self.setup_notifications()
            
            # Step 5: Add camera to system
            print("\nStep 5: Adding camera to system...")
            if self.camera_manager.add_camera(camera_config):
                print("✅ Camera added successfully!")
            else:
                print("❌ Failed to add camera. Please check the configuration.")
                return False
            
            # Step 6: Test the integration
            print("\nStep 6: Testing the integration...")
            self.test_integration()
            
            print("\n" + "="*60)
            print("   Setup Complete!")
            print("="*60)
            print("\nYour CP Plus camera is now integrated with the Omni-Dev Agent system.")
            print("The following features are now active:")
            print("• Real-time motion detection with AI analysis")
            print("• Face recognition and identification")
            print("• Automated notifications and alerts")
            print("• Scheduled privacy mode")
            print("• 360° view control via API")
            print("• Integration with OS notifications")
            print()
            print("API Endpoints:")
            print("• http://localhost:5000/camera/discover - Discover cameras")
            print("• http://localhost:5000/camera/list - List configured cameras")
            print("• http://localhost:5000/vision/detect - Object detection")
            print("• http://localhost:5000/vision/face/identify - Face recognition")
            print("• WebSocket: ws://localhost:5000 - Live video streaming")
            print()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user.")
            return False
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            print(f"\n❌ Setup failed: {e}")
            return False
    
    def discover_cameras(self) -> List[Dict[str, Any]]:
        """Discover cameras on the network"""
        print("🔍 Scanning network for CP Plus cameras...")
        print("This may take a minute...")
        
        cameras = self.camera_manager.discover_cameras()
        
        if not cameras:
            print("❌ No cameras found on the network.")
            print("\nTroubleshooting tips:")
            print("• Ensure your camera is powered on and connected to Wi-Fi")
            print("• Check that your computer and camera are on the same network")
            print("• Verify the camera's IP address manually if known")
            print()
            
            manual_ip = input("Enter camera IP address manually (or press Enter to exit): ").strip()
            if manual_ip:
                cameras = [{'ip': manual_ip, 'port': 80, 'manufacturer': 'CP Plus', 'model': 'CP-E305A'}]
            else:
                sys.exit(1)
        else:
            print(f"✅ Found {len(cameras)} potential camera(s):")
            for i, camera in enumerate(cameras):
                print(f"  {i+1}. {camera['ip']}:{camera['port']} - {camera.get('manufacturer', 'Unknown')} {camera.get('model', '')}")
        
        return cameras
    
    def configure_camera(self, cameras: List[Dict[str, Any]]) -> CameraConfig:
        """Configure camera credentials and settings"""
        
        # Select camera if multiple found
        if len(cameras) > 1:
            while True:
                try:
                    choice = int(input(f"\nSelect camera (1-{len(cameras)}): ")) - 1
                    if 0 <= choice < len(cameras):
                        selected_camera = cameras[choice]
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            selected_camera = cameras[0]
        
        print(f"\nSelected camera: {selected_camera['ip']}")
        
        # Get credentials
        print("\n📝 Camera credentials:")
        print("For CP Plus cameras, common default credentials are:")
        print("  Username: admin, Password: admin")
        print("  Username: admin, Password: (empty)")
        print("  Username: admin, Password: 123456")
        print()
        
        username = input("Enter username (default: admin): ").strip() or "admin"
        password = input("Enter password (default: admin): ").strip() or "admin"
        
        # Create credentials
        credentials = CameraCredentials(
            username=username,
            password=password,
            ip_address=selected_camera['ip'],
            port=selected_camera.get('port', 80)
        )
        
        # Test credentials
        print(f"\n🔄 Testing connection to {credentials.ip_address}...")
        try:
            from components.camera_integration.camera_manager import CPPlusCameraController
            controller = CPPlusCameraController(credentials)
            device_info = controller.get_device_info()
            
            if device_info.get('status') == 'success':
                print("✅ Connection successful!")
            else:
                print("❌ Connection failed. Please check credentials and network connectivity.")
                return None
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            retry = input("Continue anyway? (y/N): ").strip().lower()
            if retry != 'y':
                return None
        
        # Camera capabilities for CP-E305A
        capabilities = CameraCapabilities(
            has_360_view=True,
            has_motion_detection=True,
            has_motion_tracking=True,
            has_privacy_mode=True,
            has_ir_lights=True,
            has_night_vision=True,
            max_resolution=(1920, 1080),
            supported_formats=['H264', 'MJPEG']
        )
        
        # Create camera config
        camera_id = input("Enter camera ID (default: cp_plus_main): ").strip() or "cp_plus_main"
        camera_name = input("Enter camera name (default: CP Plus Main Camera): ").strip() or "CP Plus Main Camera"
        
        config = CameraConfig(
            camera_id=camera_id,
            name=camera_name,
            credentials=credentials,
            capabilities=capabilities,
            motion_detection_enabled=True,
            ai_analysis_enabled=True
        )
        
        return config
    
    def setup_ai_vision(self):
        """Setup AI vision capabilities"""
        print("\n🤖 AI Vision capabilities:")
        print("The following AI features will be enabled:")
        print("• Object detection (people, vehicles, animals)")
        print("• Face recognition and identification")
        print("• Motion tracking with smart alerts")
        print("• Real-time analysis of camera feed")
        print()
        
        enable_face_recognition = input("Enable face recognition? (Y/n): ").strip().lower() != 'n'
        if enable_face_recognition:
            print("💡 To use face recognition, you can add known faces through the API:")
            print("  POST /vision/face/enroll - Add new face")
            print("  GET /vision/face/identities - List enrolled faces")
    
    def setup_notifications(self):
        """Setup notifications and automation"""
        print("\n🔔 Notification and automation setup:")
        print("Configure how you want to be notified of camera events...")
        print()
        
        # Desktop notifications
        enable_desktop = input("Enable desktop notifications? (Y/n): ").strip().lower() != 'n'
        enable_tts = input("Enable text-to-speech announcements? (y/N): ").strip().lower() == 'y'
        
        # Email notifications
        enable_email = input("Enable email notifications? (y/N): ").strip().lower() == 'y'
        email_config = {}
        
        if enable_email:
            print("\n📧 Email configuration:")
            email_config = {
                'smtp_server': input("SMTP server (e.g., smtp.gmail.com): ").strip(),
                'smtp_port': int(input("SMTP port (default: 587): ").strip() or "587"),
                'username': input("Email username: ").strip(),
                'password': input("Email password/app password: ").strip(),
                'to_email': input("Alert recipient email: ").strip()
            }
        
        # Privacy schedule
        print("\n🔒 Privacy mode schedule:")
        enable_privacy_schedule = input("Enable privacy mode schedule? (y/N): ").strip().lower() == 'y'
        
        privacy_hours = []
        if enable_privacy_schedule:
            print("Enter hours for privacy mode (camera will be disabled):")
            start_hour = int(input("Start hour (0-23, e.g., 22 for 10 PM): ").strip())
            end_hour = int(input("End hour (0-23, e.g., 6 for 6 AM): ").strip())
            
            if start_hour > end_hour:
                # Spans midnight
                privacy_hours = list(range(start_hour, 24)) + list(range(0, end_hour + 1))
            else:
                privacy_hours = list(range(start_hour, end_hour + 1))
        
        # Create automation configuration
        automation_config = {
            'email': email_config if enable_email else None,
            'automation_rules': []
        }
        
        # Add basic motion detection rule
        motion_rule = {
            'name': 'Motion Detection Alert',
            'event_type': 'motion_detected',
            'conditions': {
                'detected_objects': ['person'],
                'min_confidence': 0.6
            },
            'actions': []
        }
        
        if enable_desktop:
            motion_rule['actions'].append({
                'type': 'send_notification',
                'title': 'Security Alert',
                'message': 'Person detected on {camera_id} at {timestamp}',
                'sound': True,
                'speak': enable_tts
            })
        
        if enable_email and email_config.get('to_email'):
            motion_rule['actions'].append({
                'type': 'send_email',
                'to_email': email_config['to_email'],
                'subject': 'Security Alert - Motion Detected',
                'body': 'Motion detected on camera {camera_id} at {timestamp}'
            })
        
        automation_config['automation_rules'].append(motion_rule)
        
        # Save configuration
        self.os_integration.save_configuration(automation_config)
        
        # Set privacy schedule if enabled
        if privacy_hours:
            self.privacy_hours = privacy_hours
        
        print("✅ Notification and automation configured!")
    
    def test_integration(self):
        """Test the complete integration"""
        print("\n🧪 Testing integration...")
        
        # Test desktop notification
        try:
            self.os_integration.send_test_notification(
                "Camera Integration Test",
                "Your CP Plus camera integration is working!"
            )
            print("✅ Desktop notification test successful")
        except Exception as e:
            print(f"⚠️  Desktop notification test failed: {e}")
        
        # Test camera connection
        try:
            cameras = list(self.camera_manager.cameras.keys())
            if cameras:
                camera_id = cameras[0]
                status = self.camera_manager.get_camera_status(camera_id)
                print(f"✅ Camera status: {status.get('status', 'unknown')}")
            else:
                print("⚠️  No cameras configured for testing")
        except Exception as e:
            print(f"⚠️  Camera test failed: {e}")
        
        print("\n🚀 Integration testing complete!")

def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_camera_vision.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please install dependencies manually:")
        print(f"  pip install -r requirements_camera_vision.txt")
        return False

def main():
    """Main setup function"""
    print("CP Plus Camera Integration Setup")
    print("================================")
    
    # Check if running from correct directory
    if not Path("components/camera_integration").exists():
        print("❌ Please run this script from the src directory")
        sys.exit(1)
    
    # Ask about dependency installation
    install_deps = input("Install required dependencies? (Y/n): ").strip().lower() != 'n'
    if install_deps:
        if not install_dependencies():
            print("Please install dependencies before continuing.")
            return
    
    print()
    
    # Run setup wizard
    wizard = CameraSetupWizard()
    success = wizard.run_setup()
    
    if success:
        print("Setup completed successfully!")
        print("You can now start the vision API server with:")
        print("  python main.py")
    else:
        print("Setup was not completed successfully.")
        print("Please check the logs and try again.")

if __name__ == "__main__":
    main()
