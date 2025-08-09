#!/usr/bin/env python3
"""
Performance Optimization Setup Script

This script sets up the performance optimization system with GPU acceleration,
CUDA support, and all necessary dependencies.

Usage:
    python setup_performance_optimization.py [--install-cuda] [--test] [--benchmark]
"""

import subprocess
import sys
import os
import platform
import logging
import argparse
from pathlib import Path
import json
import importlib.util

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def check_package_installed(package_name):
    """Check if a package is installed"""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except ImportError:
        return False

def run_command(command, description=None, check=True):
    """Run a system command"""
    if description:
        logger.info(f"{description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.debug(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        if e.stderr:
            logger.error(f"Command failed: {e.stderr}")
        if check:
            raise
        return e

def detect_system_info():
    """Detect system information"""
    info = {
        'platform': platform.system(),
        'architecture': platform.architecture()[0],
        'python_version': sys.version,
        'cuda_available': False,
        'gpu_detected': False
    }
    
    # Check for NVIDIA GPU
    try:
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            info['gpu_detected'] = True
            logger.info("NVIDIA GPU detected")
    except:
        logger.info("No NVIDIA GPU detected or nvidia-smi not available")
    
    # Check CUDA availability
    if check_package_installed('torch'):
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                logger.info(f"CUDA available: {info['cuda_version']}, GPUs: {info['gpu_count']}")
        except:
            pass
    
    return info

def install_base_requirements():
    """Install base requirements"""
    logger.info("Installing base requirements...")
    
    # Install basic packages first
    base_packages = [
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'Pillow>=9.5.0',
        'psutil>=5.9.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0'
    ]
    
    for package in base_packages:
        run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")

def install_pytorch(cuda_version=None):
    """Install PyTorch with optional CUDA support"""
    logger.info("Installing PyTorch...")
    
    if cuda_version:
        if cuda_version in ["11.8", "118"]:
            index_url = "https://download.pytorch.org/whl/cu118"
            logger.info("Installing PyTorch with CUDA 11.8 support")
        elif cuda_version in ["12.1", "121"]:
            index_url = "https://download.pytorch.org/whl/cu121"
            logger.info("Installing PyTorch with CUDA 12.1 support")
        else:
            logger.warning(f"Unknown CUDA version {cuda_version}, installing CPU version")
            index_url = None
    else:
        logger.info("Installing PyTorch CPU version")
        index_url = None
    
    pytorch_packages = "torch torchvision torchaudio"
    
    if index_url:
        command = f"{sys.executable} -m pip install {pytorch_packages} --index-url {index_url}"
    else:
        command = f"{sys.executable} -m pip install {pytorch_packages}"
    
    run_command(command, "Installing PyTorch")

def install_gpu_utilities():
    """Install GPU monitoring utilities"""
    logger.info("Installing GPU utilities...")
    
    gpu_packages = [
        'GPUtil>=1.4.0',
        'nvidia-ml-py>=12.535.77'
    ]
    
    for package in gpu_packages:
        try:
            run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to install {package}, continuing...")

def install_performance_packages():
    """Install performance optimization packages"""
    logger.info("Installing performance optimization packages...")
    
    packages = [
        'asyncio-throttle>=1.0.2',
        'aiofiles>=23.1.0',
        'onnxruntime>=1.15.0',
        'transformers>=4.30.0',
        'ultralytics>=8.0.0',
        'pyyaml>=6.0',
        'colorama>=0.4.6'
    ]
    
    for package in packages:
        try:
            run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {package}: {e}")

def install_optional_packages():
    """Install optional high-performance packages"""
    logger.info("Installing optional packages...")
    
    optional_packages = [
        'memory-profiler>=0.60.0',
        'line-profiler>=4.0.0',
        'pytest>=7.3.0',
        'pytest-asyncio>=0.21.0',
        'pytest-benchmark>=4.0.0'
    ]
    
    for package in optional_packages:
        try:
            run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to install optional package {package}")

def test_installation():
    """Test the installation"""
    logger.info("Testing installation...")
    
    # Test basic imports
    test_imports = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('psutil', 'psutil'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_passed = True
    
    for module_name, display_name in test_imports:
        try:
            __import__(module_name)
            logger.info(f"✓ {display_name} imported successfully")
        except ImportError as e:
            logger.error(f"✗ Failed to import {display_name}: {e}")
            all_passed = False
    
    # Test PyTorch CUDA
    try:
        import torch
        logger.info(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.version.cuda}")
            logger.info(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {gpu_name}")
        else:
            logger.info("ℹ CUDA not available, will use CPU")
    except Exception as e:
        logger.error(f"✗ PyTorch test failed: {e}")
        all_passed = False
    
    # Test performance optimizer
    try:
        from components.ai_vision.performance_optimizer import PerformanceOptimizer
        optimizer = PerformanceOptimizer()
        logger.info("✓ Performance optimizer loaded successfully")
        
        # Test device detection
        device = optimizer.device_manager.detect_optimal_device()
        logger.info(f"✓ Optimal device detected: {device}")
        
    except ImportError as e:
        logger.error(f"✗ Failed to import performance optimizer: {e}")
        all_passed = False
    except Exception as e:
        logger.error(f"✗ Performance optimizer test failed: {e}")
        all_passed = False
    
    return all_passed

def run_quick_benchmark():
    """Run a quick benchmark test"""
    logger.info("Running quick benchmark test...")
    
    try:
        # Import benchmark script
        benchmark_script = Path(__file__).parent / "benchmark_performance_optimization.py"
        
        if benchmark_script.exists():
            command = f"{sys.executable} {benchmark_script} --quick --scenarios baseline_cpu optimized_cpu"
            result = run_command(command, "Running quick benchmark")
            logger.info("✓ Quick benchmark completed successfully")
            return True
        else:
            logger.warning("Benchmark script not found, skipping benchmark test")
            return False
    except Exception as e:
        logger.error(f"✗ Benchmark test failed: {e}")
        return False

def create_config_files():
    """Create example configuration files"""
    logger.info("Creating configuration files...")
    
    # Performance config example
    perf_config = {
        "device": "AUTO",
        "precision": "AUTO",
        "optimization_level": "AGGRESSIVE",
        "batch_size": 8,
        "max_queue_size": 100,
        "async_processing": True,
        "enable_profiling": True
    }
    
    config_file = Path("performance_config.json")
    with open(config_file, 'w') as f:
        json.dump(perf_config, f, indent=2)
    
    logger.info(f"✓ Created performance config: {config_file}")
    
    # Benchmark config example
    benchmark_config = {
        "test_scenarios": [
            {
                "name": "baseline_cpu",
                "device": "CPU",
                "precision": "FULL",
                "optimization_level": "NONE",
                "batch_size": 1,
                "async_processing": False
            },
            {
                "name": "optimized_auto",
                "device": "AUTO",
                "precision": "AUTO",
                "optimization_level": "AGGRESSIVE",
                "batch_size": 8,
                "async_processing": True
            }
        ],
        "test_parameters": {
            "num_frames": 50,
            "frame_resolution": [640, 640],
            "measurement_iterations": 3
        }
    }
    
    benchmark_config_file = Path("benchmark_config.json")
    with open(benchmark_config_file, 'w') as f:
        json.dump(benchmark_config, f, indent=2)
    
    logger.info(f"✓ Created benchmark config: {benchmark_config_file}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Performance Optimization Setup")
    parser.add_argument('--install-cuda', type=str, help='Install CUDA version (11.8 or 12.1)')
    parser.add_argument('--test', action='store_true', help='Run installation tests')
    parser.add_argument('--benchmark', action='store_true', help='Run quick benchmark')
    parser.add_argument('--skip-install', action='store_true', help='Skip package installation')
    parser.add_argument('--config-only', action='store_true', help='Only create config files')
    
    args = parser.parse_args()
    
    logger.info("Performance Optimization Setup Starting...")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Detect system info
    system_info = detect_system_info()
    logger.info(f"System: {system_info['platform']} {system_info['architecture']}")
    
    if args.config_only:
        create_config_files()
        logger.info("Configuration files created successfully!")
        return 0
    
    if not args.skip_install:
        try:
            # Install packages
            install_base_requirements()
            
            # Install PyTorch
            install_pytorch(args.install_cuda)
            
            # Install GPU utilities
            if system_info['gpu_detected'] or args.install_cuda:
                install_gpu_utilities()
            
            # Install performance packages
            install_performance_packages()
            
            # Install optional packages
            install_optional_packages()
            
            logger.info("✓ All packages installed successfully!")
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return 1
    
    # Create config files
    create_config_files()
    
    # Run tests
    if args.test or not args.skip_install:
        logger.info("\n" + "="*50)
        logger.info("RUNNING INSTALLATION TESTS")
        logger.info("="*50)
        
        if test_installation():
            logger.info("✓ All tests passed!")
        else:
            logger.warning("⚠ Some tests failed, but installation may still work")
    
    # Run benchmark
    if args.benchmark:
        logger.info("\n" + "="*50)
        logger.info("RUNNING QUICK BENCHMARK")
        logger.info("="*50)
        
        if run_quick_benchmark():
            logger.info("✓ Benchmark completed successfully!")
        else:
            logger.warning("⚠ Benchmark failed or was skipped")
    
    # Final instructions
    logger.info("\n" + "="*50)
    logger.info("SETUP COMPLETE!")
    logger.info("="*50)
    
    print("""
Next steps:
1. Test the performance optimizer:
   python example_performance_optimization_integration.py

2. Run comprehensive benchmarks:
   python benchmark_performance_optimization.py

3. Customize configuration:
   Edit performance_config.json for your needs

4. For CUDA setup help:
   python setup_performance_optimization.py --install-cuda 11.8

5. For testing only:
   python setup_performance_optimization.py --test --skip-install
    """)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
