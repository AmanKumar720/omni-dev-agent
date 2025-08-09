import tkinter as tk
from tkinter import ttk
# from components.device_manager import DeviceManager, StreamStatus, FrameData
import traceback
import cv2
from PIL import Image, ImageTk
import threading
import sounddevice as sd
import numpy as np
from ultralytics import YOLO

class DesktopAgent(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Omni-Dev Agent - Vision Mode")
        self.geometry("1280x800")

        # self.device_manager = DeviceManager()
        self.stream_id = None
        self.is_streaming = False
        self.audio_stream = None
        self.is_recording = False
        self.vision_mode = tk.BooleanVar(value=False)

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Using the nano model for performance

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for controls
        left_frame = ttk.Frame(main_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Right frame for video and audio
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Control frame
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # Refresh button
        self.refresh_button = ttk.Button(control_frame, text="Refresh Devices", command=self.refresh_devices)
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 5))

        # Device list (Treeview)
        self.device_tree = ttk.Treeview(left_frame, columns=("type", "name", "index"), show="headings")
        self.device_tree.heading("type", text="Type")
        self.device_tree.heading("name", text="Name/IP")
        self.device_tree.heading("index", text="Index")
        self.device_tree.column("index", width=50, stretch=tk.NO)
        self.device_tree.pack(fill=tk.BOTH, expand=True, pady=5)

        # Camera control frame
        camera_control_frame = ttk.LabelFrame(left_frame, text="Camera")
        camera_control_frame.pack(fill=tk.X, pady=5)

        self.start_camera_button = ttk.Button(camera_control_frame, text="Start Camera", command=self.start_camera)
        self.start_camera_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_camera_button = ttk.Button(camera_control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_camera_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.vision_mode_check = ttk.Checkbutton(camera_control_frame, text="Vision Mode", variable=self.vision_mode)
        self.vision_mode_check.pack(side=tk.LEFT, padx=5, pady=5)

        # Audio control frame
        audio_control_frame = ttk.LabelFrame(left_frame, text="Microphone")
        audio_control_frame.pack(fill=tk.X, pady=5)

        self.start_mic_button = ttk.Button(audio_control_frame, text="Start Mic", command=self.start_microphone)
        self.start_mic_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_mic_button = ttk.Button(audio_control_frame, text="Stop Mic", command=self.stop_microphone, state=tk.DISABLED)
        self.stop_mic_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Video display
        self.video_label = ttk.Label(right_frame, text="Camera feed will appear here", anchor=tk.CENTER, relief=tk.SUNKEN)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Audio visualization
        self.audio_canvas = tk.Canvas(right_frame, height=50, bg='black')
        self.audio_canvas.pack(fill=tk.X, pady=5)
        self.volume_bar = self.audio_canvas.create_rectangle(0, 0, 0, 50, fill='green')

        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.refresh_devices()

    def refresh_devices(self):
        self.status_bar.config(text="Refreshing devices...")
        self.device_tree.delete(*self.device_tree.get_children())
        try:
            devices = self.device_manager.unified_device_list()
            for device in devices:
                device_type = device.get('type', 'Unknown')
                device_name = device.get('name', device.get('ip', 'Unknown'))
                device_index = device.get('index', '')
                self.device_tree.insert("", tk.END, values=(device_type, device_name, device_index))
            self.status_bar.config(text=f"Found {len(devices)} devices.")
        except Exception as e:
            self.status_bar.config(text="Error refreshing devices.")
            print(traceback.format_exc())

    def start_camera(self):
        selected_item = self.device_tree.focus()
        if not selected_item:
            self.status_bar.config(text="Please select a camera from the list.")
            return

        item = self.device_tree.item(selected_item)
        if item['values'][0] != 'camera':
            self.status_bar.config(text="Please select a camera device.")
            return

        camera_index = item['values'][2]

        self.stream_id = self.device_manager.get_camera_stream(camera_index, (640, 480), 30)
        if self.stream_id:
            self.device_manager.subscribe_to_stream(self.stream_id, self.frame_callback, self.status_callback)
            self.is_streaming = True
            self.start_camera_button.config(state=tk.DISABLED)
            self.stop_camera_button.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Started camera stream: {self.stream_id}")
        else:
            self.status_bar.config(text="Failed to start camera stream.")

    def stop_camera(self):
        if self.stream_id and self.is_streaming:
            self.device_manager.stop_camera_stream(self.stream_id)
            self.is_streaming = False
            self.stream_id = None
            self.start_camera_button.config(state=tk.NORMAL)
            self.stop_camera_button.config(state=tk.DISABLED)
            self.video_label.config(image='')
            self.video_label.photo = None
            self.status_bar.config(text="Camera stream stopped.")

    def frame_callback(self, frame_data: FrameData):
        frame = frame_data.frame
        if self.vision_mode.get():
            results = self.model(frame, verbose=False)
            frame = results[0].plot()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk)
        self.video_label.photo = imgtk

    def status_callback(self, status: StreamStatus, message: str):
        self.status_bar.config(text=f"Stream Status: {status.value} - {message}")

    def start_microphone(self):
        selected_item = self.device_tree.focus()
        if not selected_item:
            self.status_bar.config(text="Please select a microphone from the list.")
            return

        item = self.device_tree.item(selected_item)
        if item['values'][0] != 'input':
            self.status_bar.config(text="Please select a microphone device.")
            return

        mic_index = item['values'][2]

        try:
            self.audio_stream = sd.InputStream(device=mic_index, channels=1, callback=self.audio_callback)
            self.audio_stream.start()
            self.is_recording = True
            self.start_mic_button.config(state=tk.DISABLED)
            self.stop_mic_button.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Started microphone: {item['values'][1]}")
        except Exception as e:
            self.status_bar.config(text="Failed to start microphone.")
            print(traceback.format_exc())

    def stop_microphone(self):
        if self.audio_stream and self.is_recording:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.is_recording = False
            self.audio_stream = None
            self.start_mic_button.config(state=tk.NORMAL)
            self.stop_mic_button.config(state=tk.DISABLED)
            self.audio_canvas.coords(self.volume_bar, 0, 0, 0, 50)
            self.status_bar.config(text="Microphone stopped.")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        volume_norm = np.linalg.norm(indata) * 10
        bar_width = min(int(volume_norm), self.audio_canvas.winfo_width())
        self.audio_canvas.coords(self.volume_bar, 0, 0, bar_width, 50)

    def on_closing(self):
        if self.is_streaming:
            self.stop_camera()
        if self.is_recording:
            self.stop_microphone()
        self.destroy()

if __name__ == "__main__":
    app = DesktopAgent()
    app.mainloop()
