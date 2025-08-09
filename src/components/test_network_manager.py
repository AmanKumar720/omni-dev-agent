
import unittest
from unittest.mock import patch, MagicMock
from src.components.network_manager import NetworkManager
import cv2

class TestNetworkManager(unittest.TestCase):

    @patch('src.components.network_manager.nmap.PortScanner')
    @patch('src.components.network_manager.cv2.VideoCapture')
    @patch('src.components.network_manager.cv2.VideoWriter')
    def test_record_camera(self, mock_video_writer, mock_video_capture, mock_port_scanner):
        # Arrange
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.get.side_effect = [24, 640, 480]  # fps, width, height
        mock_capture_instance.read.side_effect = [(True, 'frame1'), (True, 'frame2'), (False, None)]
        mock_video_capture.return_value = mock_capture_instance

        mock_writer_instance = MagicMock()
        mock_video_writer.return_value = mock_writer_instance

        network_manager = NetworkManager()

        # Act
        network_manager.record_camera('192.168.1.100', duration_hours=0.001)

        # Assert
        mock_video_capture.assert_called_with('rtsp://192.168.1.100:554/')
        mock_video_writer.assert_called_with('recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, (640, 480))
        self.assertEqual(mock_writer_instance.write.call_count, 2)
        mock_capture_instance.release.assert_called_once()
        mock_writer_instance.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()
