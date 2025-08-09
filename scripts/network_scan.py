
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.network_manager import NetworkManager

def main():
    """
    Scans the local network and prints a report of all discovered devices.
    """
    try:
        nm = NetworkManager()
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Discovering devices on the network...")
    devices = nm.discover_devices()

    if not devices:
        print("No devices found.")
        return

    print(f"Found {len(devices)} devices:")
    for device in devices:
        print(f"- IP: {device['ip']}")
        print("  Scanning for open ports and services...")
        try:
            scan_results = nm.scan_device(device['ip'])
            if 'tcp' in scan_results:
                for port, port_info in scan_results['tcp'].items():
                    print(f"    - Port {port}: {port_info['name']} ({port_info['product']} {port_info['version']})")
            if 'osmatch' in scan_results and scan_results['osmatch']:
                print(f"  OS: {scan_results['osmatch'][0]['name']}")
        except Exception as e:
            print(f"  Error scanning device: {e}")

if __name__ == "__main__":
    main()
