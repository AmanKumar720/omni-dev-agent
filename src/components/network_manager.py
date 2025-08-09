
import nmap
import socket
import cv2
import time

class NetworkManager:
    """
    A component for discovering and identifying devices on the local network.
    """

    def __init__(self):
        try:
            self.nm = nmap.PortScanner()
        except nmap.PortScannerError:
            raise Exception("Nmap not found. Please install it on your system: https://nmap.org/download.html")

    def get_local_ip(self):
        """
        Gets the local IP address of the machine.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def discover_devices(self, network_range=None):
        """
        Discovers all devices on the local network.

        :param network_range: The network range to scan (e.g., '192.168.1.0/24').
                              If None, it will be automatically determined.
        :return: A list of dictionaries, where each dictionary represents a device.
        """
        if not network_range:
            local_ip = self.get_local_ip()
            network_range = f"{'.'.join(local_ip.split('.')[:-1])}.0/24"

        self.nm.scan(hosts=network_range, arguments='-sn')
        hosts_list = [(x, self.nm[x]['status']['state']) for x in self.nm.all_hosts()]
        return [{"ip": host, "status": status} for host, status in hosts_list]

    def scan_device(self, ip_address):
        """
        Scans a single device for open ports and services.

        :param ip_address: The IP address of the device to scan.
        :return: A dictionary containing the scan results.
        """
        self.nm.scan(hosts=ip_address, arguments='-sV')
        return self.nm[ip_address]

    def find_cameras(self, network_range=None):
        """
        Finds potential cameras on the network by scanning for the RTSP port.

        :param network_range: The network range to scan (e.g., '192.168.1.0/24').
        :return: A list of IP addresses that are likely cameras.
        """
        if not network_range:
            local_ip = self.get_local_ip()
            network_range = f"{'.'.join(local_ip.split('.')[:-1])}.0/24"

        self.nm.scan(hosts=network_range, arguments='-p 554 --open')
        camera_ips = self.nm.all_hosts()
        return camera_ips

    def record_camera(self, camera_ip, duration_hours=4, output_file='recording.avi'):
        """
        Records video from a camera for a specified duration.

        :param camera_ip: The IP address of the camera.
        :param duration_hours: The duration of the recording in hours.
        :param output_file: The name of the output video file.
        """
        rtsp_url = f"rtsp://{camera_ip}:554/"
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            raise Exception(f"Could not open camera stream at {rtsp_url}")

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
