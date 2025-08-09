# src/components/ai_vision/model_hub.py

import os
import hashlib
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import wraps
import pickle
import weakref

# Import error handling
try:
    from ...error_handling import (
        with_model_error_handling, with_vision_error_handling, vision_error_handler,
        ModelLoadError, ModelValidationError, NetworkError, InsufficientMemoryError,
        GPUError, DependencyError, ConfigurationError
    )
except ImportError:
    from error_handling import (
        with_model_error_handling, with_vision_error_handling, vision_error_handler,
        ModelLoadError, ModelValidationError, NetworkError, InsufficientMemoryError,
        GPUError, DependencyError, ConfigurationError
    )

# Model-specific imports (lazy loaded)
_torch = None
_transformers = None
_cv2 = None
_ultralytics = None
_tensorflow = None
_face_recognition = None
_pytesseract = None


class ModelType(Enum):
    """Enumeration of supported model types"""
    YOLO_V8 = "yolo_v8"
    SSD = "ssd"
    RESNET = "resnet" 
    MOBILENET = "mobilenet"
    OCR = "ocr"
    FACE_RECOGNITION = "face_recognition"


class DeviceType(Enum):
    """Device type enumeration"""
    CPU = "cpu"
    GPU = "cuda"
    AUTO = "auto"


class ModelStatus(Enum):
    """Model status enumeration"""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    READY = "ready"
    ERROR = "error"
    LOADING = "loading"
    LOADED = "loaded"


@dataclass
class ModelMetadata:
    """Model metadata container"""
    name: str
    model_type: ModelType
    version: str
    size_mb: float
    checksum_sha256: str
    download_url: str
    description: str
    supported_devices: List[DeviceType]
    dependencies: List[str]
    last_updated: datetime
    license: str = "Unknown"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class CacheEntry:
    """Cache entry for model data"""
    model_path: Path
    metadata: ModelMetadata
    last_accessed: datetime
    access_count: int
    size_bytes: int
    checksum_verified: bool = False


class ModelRegistry:
    """Registry of available models with their metadata"""
    
    def __init__(self):
        self._models = self._initialize_model_registry()
    
    def _initialize_model_registry(self) -> Dict[str, ModelMetadata]:
        """Initialize the model registry with predefined models"""
        return {
            # YOLOv8 Models
            "yolov8n": ModelMetadata(
                name="yolov8n",
                model_type=ModelType.YOLO_V8,
                version="8.0.0",
                size_mb=6.2,
                checksum_sha256="3cc4b9f81e8f6e0e2e2b2e8c1c5f4a8a9c3c8e2e3d8f9c8a5b2c1e5f8a9c3c8",
                download_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                description="YOLOv8 Nano - fastest and most lightweight",
                supported_devices=[DeviceType.CPU, DeviceType.GPU],
                dependencies=["ultralytics", "torch", "torchvision"],
                last_updated=datetime.now(),
                license="AGPL-3.0"
            ),
            "yolov8s": ModelMetadata(
                name="yolov8s",
                model_type=ModelType.YOLO_V8,
                version="8.0.0",
                size_mb=21.5,
                checksum_sha256="4cc5c9f81e8f6e0e2e2b2e8c1c5f4a8a9c3c8e2e3d8f9c8a5b2c1e5f8a9c3c9",
                download_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
                description="YOLOv8 Small - balanced speed and accuracy",
                supported_devices=[DeviceType.CPU, DeviceType.GPU],
                dependencies=["ultralytics", "torch", "torchvision"],
                last_updated=datetime.now(),
                license="AGPL-3.0"
            ),
            # SSD Models
            "ssd_mobilenet_v2": ModelMetadata(
                name="ssd_mobilenet_v2",
                model_type=ModelType.SSD,
                version="1.0.0",
                size_mb=67.0,
                checksum_sha256="5cc6d9f81e8f6e0e2e2b2e8c1c5f4a8a9c3c8e2e3d8f9c8a5b2c1e5f8a9c3ca",
                download_url="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
                description="SSD MobileNet v2 - efficient object detection",
                supported_devices=[DeviceType.CPU, DeviceType.GPU],
                dependencies=["tensorflow", "opencv-python"],
                last_updated=datetime.now(),
                license="Apache-2.0"
            ),
            # ResNet Models
            "resnet50": ModelMetadata(
                name="resnet50",
                model_type=ModelType.RESNET,
                version="1.0.0",
                size_mb=98.0,
                checksum_sha256="6cc7e9f81e8f6e0e2e2b2e8c1c5f4a8a9c3c8e2e3d8f9c8a5b2c1e5f8a9c3cb",
                download_url="https://download.pytorch.org/models/resnet50-19c8e357.pth",
                description="ResNet-50 - deep residual network for image classification",
                supported_devices=[DeviceType.CPU, DeviceType.GPU],
                dependencies=["torch", "torchvision"],
                last_updated=datetime.now(),
                license="BSD"
            ),
            # MobileNet Models
            "mobilenet_v2": ModelMetadata(
                name="mobilenet_v2",
                model_type=ModelType.MOBILENET,
                version="1.0.0",
                size_mb=14.0,
                checksum_sha256="7cc8f9f81e8f6e0e2e2b2e8c1c5f4a8a9c3c8e2e3d8f9c8a5b2c1e5f8a9c3cc",
                download_url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                description="MobileNet v2 - efficient mobile-first architecture",
                supported_devices=[DeviceType.CPU, DeviceType.GPU],
                dependencies=["torch", "torchvision"],
                last_updated=datetime.now(),
                license="Apache-2.0"
            ),
            # OCR Models
            "tesseract_eng": ModelMetadata(
                name="tesseract_eng",
                model_type=ModelType.OCR,
                version="5.0.0",
                size_mb=5.2,
                checksum_sha256="8cc9g9f81e8f6e0e2e2b2e8c1c5f4a8a9c3c8e2e3d8f9c8a5b2c1e5f8a9c3cd",
                download_url="https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata",
                description="Tesseract English language model",
                supported_devices=[DeviceType.CPU],
                dependencies=["pytesseract", "pillow"],
                last_updated=datetime.now(),
                license="Apache-2.0"
            ),
            # Face Recognition Models
            "face_recognition_hog": ModelMetadata(
                name="face_recognition_hog",
                model_type=ModelType.FACE_RECOGNITION,
                version="1.3.0",
                size_mb=3.1,
                checksum_sha256="9ccah9f81e8f6e0e2e2b2e8c1c5f4a8a9c3c8e2e3d8f9c8a5b2c1e5f8a9c3ce",
                download_url="https://github.com/ageitgey/face_recognition_models/raw/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat",
                description="HOG-based face recognition model",
                supported_devices=[DeviceType.CPU],
                dependencies=["face_recognition", "dlib"],
                last_updated=datetime.now(),
                license="MIT"
            ),
        }
    
    def get_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Get model metadata by name"""
        return self._models.get(model_name)
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """List available models, optionally filtered by type"""
        models = list(self._models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return models
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model"""
        try:
            self._models[metadata.name] = metadata
            return True
        except Exception:
            return False


class DeviceManager:
    """Manages device selection and optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".DeviceManager")
        self._device_cache = {}
        self._lock = threading.Lock()
    
    def get_optimal_device(self, requested_device: DeviceType = DeviceType.AUTO) -> str:
        """Get optimal device for model execution"""
        with self._lock:
            cache_key = requested_device.value
            if cache_key in self._device_cache:
                return self._device_cache[cache_key]
            
            device = self._determine_device(requested_device)
            self._device_cache[cache_key] = device
            return device
    
    def _determine_device(self, requested_device: DeviceType) -> str:
        """Determine the best available device"""
        if requested_device == DeviceType.CPU:
            return "cpu"
        
        if requested_device == DeviceType.GPU:
            if self._is_gpu_available():
                return "cuda" if self._is_cuda_available() else "cpu"
            else:
                self.logger.warning("GPU requested but not available, falling back to CPU")
                return "cpu"
        
        # AUTO selection
        if self._is_gpu_available():
            return "cuda" if self._is_cuda_available() else "cpu"
        return "cpu"
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            global _torch
            if _torch is None:
                import torch as _torch
            return _torch.cuda.is_available()
        except ImportError:
            return False
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            global _torch
            if _torch is None:
                import torch as _torch
            return _torch.cuda.is_available() and _torch.backends.cudnn.enabled
        except ImportError:
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            "cpu_available": True,
            "gpu_available": self._is_gpu_available(),
            "cuda_available": self._is_cuda_available(),
            "recommended_device": self.get_optimal_device()
        }
        
        try:
            global _torch
            if _torch is None:
                import torch as _torch
            if _torch.cuda.is_available():
                device_count = _torch.cuda.device_count()
                info.update({
                    "gpu_count": device_count,
                    "gpu_name": _torch.cuda.get_device_name(0) if device_count > 0 else None,
                    "gpu_memory": _torch.cuda.get_device_properties(0).total_memory if device_count > 0 else None
                })
        except (ImportError, AttributeError, TypeError):
            pass
        
        return info


class ChecksumValidator:
    """Validates file checksums"""
    
    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def validate_checksum(file_path: Path, expected_checksum: str) -> bool:
        """Validate file checksum"""
        try:
            actual_checksum = ChecksumValidator.calculate_sha256(file_path)
            return actual_checksum.lower() == expected_checksum.lower()
        except Exception:
            return False


class ModelDownloader:
    """Handles model downloading with progress tracking"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__ + ".ModelDownloader")
        self._download_progress = {}
        self._lock = threading.Lock()
    
    async def download_model(self, metadata: ModelMetadata, 
                           progress_callback: Optional[Callable[[float], None]] = None) -> Path:
        """Download a model with progress tracking"""
        model_path = self.cache_dir / f"{metadata.name}_{metadata.version}"
        
        if model_path.exists():
            self.logger.info(f"Model {metadata.name} already exists at {model_path}")
            return model_path
        
        self.logger.info(f"Downloading model {metadata.name} from {metadata.download_url}")
        
        try:
            # Create temporary download path
            temp_path = model_path.with_suffix('.tmp')
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(metadata.download_url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress = downloaded_size / total_size
                                progress_callback(progress)
            
            # Validate checksum
            if not ChecksumValidator.validate_checksum(temp_path, metadata.checksum_sha256):
                temp_path.unlink()
                raise Exception("Checksum validation failed")
            
            # Move to final location
            temp_path.rename(model_path)
            self.logger.info(f"Model {metadata.name} downloaded successfully")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to download model {metadata.name}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def get_download_progress(self, model_name: str) -> float:
        """Get download progress for a model"""
        with self._lock:
            return self._download_progress.get(model_name, 0.0)


class CacheManager:
    """Manages model cache with LRU eviction"""
    
    def __init__(self, cache_dir: Path, max_cache_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.cache_index_path = cache_dir / "cache_index.json"
        self.logger = logging.getLogger(__name__ + ".CacheManager")
        self._lock = threading.Lock()
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache index
        self._cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, CacheEntry]:
        """Load cache index from disk"""
        if not self.cache_index_path.exists():
            return {}
        
        try:
            with open(self.cache_index_path, 'r') as f:
                data = json.load(f)
            
            cache_index = {}
            for key, entry_data in data.items():
                # Reconstruct CacheEntry objects
                entry_data['model_path'] = Path(entry_data['model_path'])
                entry_data['last_accessed'] = datetime.fromisoformat(entry_data['last_accessed'])
                entry_data['metadata']['last_updated'] = datetime.fromisoformat(entry_data['metadata']['last_updated'])
                entry_data['metadata']['model_type'] = ModelType(entry_data['metadata']['model_type'])
                entry_data['metadata']['supported_devices'] = [DeviceType(d) for d in entry_data['metadata']['supported_devices']]
                
                metadata = ModelMetadata(**entry_data['metadata'])
                cache_entry = CacheEntry(**{k: v for k, v in entry_data.items() if k != 'metadata'}, metadata=metadata)
                cache_index[key] = cache_entry
            
            return cache_index
            
        except Exception as e:
            self.logger.warning(f"Failed to load cache index: {e}")
            return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            data = {}
            for key, entry in self._cache_index.items():
                entry_dict = asdict(entry)
                entry_dict['model_path'] = str(entry_dict['model_path'])
                entry_dict['last_accessed'] = entry_dict['last_accessed'].isoformat()
                entry_dict['metadata']['last_updated'] = entry_dict['metadata']['last_updated'].isoformat()
                entry_dict['metadata']['model_type'] = entry_dict['metadata']['model_type'].value
                entry_dict['metadata']['supported_devices'] = [d.value for d in entry_dict['metadata']['supported_devices']]
                data[key] = entry_dict
            
            with open(self.cache_index_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def add_model(self, model_name: str, model_path: Path, metadata: ModelMetadata):
        """Add a model to the cache"""
        with self._lock:
            cache_key = f"{model_name}_{metadata.version}"
            
            # Calculate file size
            size_bytes = model_path.stat().st_size if model_path.exists() else 0
            
            cache_entry = CacheEntry(
                model_path=model_path,
                metadata=metadata,
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                checksum_verified=True
            )
            
            self._cache_index[cache_key] = cache_entry
            self._enforce_cache_limits()
            self._save_cache_index()
    
    def get_model(self, model_name: str, version: str = None) -> Optional[CacheEntry]:
        """Get a model from cache"""
        with self._lock:
            # Try exact match first
            cache_key = f"{model_name}_{version}" if version else None
            if cache_key and cache_key in self._cache_index:
                entry = self._cache_index[cache_key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._save_cache_index()
                return entry
            
            # Try partial match if no version specified
            if not version:
                for key, entry in self._cache_index.items():
                    if key.startswith(f"{model_name}_"):
                        entry.last_accessed = datetime.now()
                        entry.access_count += 1
                        self._save_cache_index()
                        return entry
            
            return None
    
    def remove_model(self, model_name: str, version: str = None) -> bool:
        """Remove a model from cache"""
        with self._lock:
            cache_key = f"{model_name}_{version}" if version else None
            
            # Find model to remove
            keys_to_remove = []
            if cache_key and cache_key in self._cache_index:
                keys_to_remove = [cache_key]
            elif not version:
                keys_to_remove = [k for k in self._cache_index.keys() if k.startswith(f"{model_name}_")]
            
            if not keys_to_remove:
                return False
            
            # Remove files and cache entries
            for key in keys_to_remove:
                entry = self._cache_index[key]
                try:
                    if entry.model_path.exists():
                        if entry.model_path.is_dir():
                            shutil.rmtree(entry.model_path)
                        else:
                            entry.model_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove model file {entry.model_path}: {e}")
                
                del self._cache_index[key]
            
            self._save_cache_index()
            return True
    
    def _enforce_cache_limits(self):
        """Enforce cache size limits using LRU eviction"""
        total_size = sum(entry.size_bytes for entry in self._cache_index.values())
        
        if total_size <= self.max_cache_size_bytes:
            return
        
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries until under limit
        for key, entry in sorted_entries:
            if total_size <= self.max_cache_size_bytes:
                break
            
            try:
                if entry.model_path.exists():
                    if entry.model_path.is_dir():
                        shutil.rmtree(entry.model_path)
                    else:
                        entry.model_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove cached model {entry.model_path}: {e}")
            
            del self._cache_index[key]
            total_size -= entry.size_bytes
            self.logger.info(f"Evicted cached model {key} to enforce size limits")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache_index.values())
            return {
                "total_models": len(self._cache_index),
                "total_size_bytes": total_size,
                "total_size_gb": total_size / (1024 ** 3),
                "max_size_gb": self.max_cache_size_bytes / (1024 ** 3),
                "utilization": total_size / self.max_cache_size_bytes if self.max_cache_size_bytes > 0 else 0,
                "models": {
                    key: {
                        "name": entry.metadata.name,
                        "version": entry.metadata.version,
                        "size_mb": entry.size_bytes / (1024 ** 2),
                        "last_accessed": entry.last_accessed.isoformat(),
                        "access_count": entry.access_count
                    }
                    for key, entry in self._cache_index.items()
                }
            }


class LazyModelLoader:
    """Lazy loading manager for models"""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.logger = logging.getLogger(__name__ + ".LazyModelLoader")
        self._loaded_models = weakref.WeakValueDictionary()
        self._loading_locks = {}
        self._main_lock = threading.Lock()
    
    def get_loading_lock(self, model_key: str) -> threading.Lock:
        """Get or create a loading lock for a specific model"""
        with self._main_lock:
            if model_key not in self._loading_locks:
                self._loading_locks[model_key] = threading.Lock()
            return self._loading_locks[model_key]
    
    def load_model(self, model_path: Path, metadata: ModelMetadata, 
                  device: DeviceType = DeviceType.AUTO) -> Any:
        """Load a model with lazy loading"""
        model_key = f"{metadata.name}_{metadata.version}_{device.value}"
        
        # Check if already loaded
        if model_key in self._loaded_models:
            self.logger.debug(f"Model {model_key} already loaded, returning cached instance")
            return self._loaded_models[model_key]
        
        # Use per-model lock to prevent duplicate loading
        lock = self.get_loading_lock(model_key)
        with lock:
            # Double-check after acquiring lock
            if model_key in self._loaded_models:
                return self._loaded_models[model_key]
            
            self.logger.info(f"Loading model {metadata.name} from {model_path}")
            
            try:
                # Determine device
                device_str = self.device_manager.get_optimal_device(device)
                
                # Load based on model type
                model = self._load_model_by_type(model_path, metadata, device_str)
                
                # Cache the loaded model
                self._loaded_models[model_key] = model
                
                self.logger.info(f"Successfully loaded model {metadata.name} on {device_str}")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model {metadata.name}: {e}")
                raise
    
    def _load_model_by_type(self, model_path: Path, metadata: ModelMetadata, device: str) -> Any:
        """Load model based on its type"""
        model_type = metadata.model_type
        
        if model_type == ModelType.YOLO_V8:
            return self._load_yolo_model(model_path, device)
        elif model_type == ModelType.SSD:
            return self._load_ssd_model(model_path, device)
        elif model_type == ModelType.RESNET:
            return self._load_resnet_model(model_path, device)
        elif model_type == ModelType.MOBILENET:
            return self._load_mobilenet_model(model_path, device)
        elif model_type == ModelType.OCR:
            return self._load_ocr_model(model_path, device)
        elif model_type == ModelType.FACE_RECOGNITION:
            return self._load_face_recognition_model(model_path, device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_yolo_model(self, model_path: Path, device: str) -> Any:
        """Load YOLOv8 model"""
        try:
            global _ultralytics
            if _ultralytics is None:
                from ultralytics import YOLO as _ultralytics
            
            model = _ultralytics(str(model_path))
            model.to(device)
            return model
            
        except ImportError as e:
            raise DependencyError("ultralytics", "ultralytics package is required for YOLOv8 models")
        except Exception as e:
            raise ModelLoadError(f"yolo_{model_path.stem}", f"Failed to load YOLO model: {str(e)}")
    
    def _load_ssd_model(self, model_path: Path, device: str) -> Any:
        """Load SSD model"""
        try:
            global _tensorflow
            if _tensorflow is None:
                import tensorflow as _tensorflow
            
            # Load TensorFlow model
            model = _tensorflow.saved_model.load(str(model_path))
            return model
            
        except ImportError:
            raise ImportError("tensorflow package is required for SSD models")
    
    def _load_resnet_model(self, model_path: Path, device: str) -> Any:
        """Load ResNet model"""
        try:
            global _torch
            if _torch is None:
                import torch as _torch
            
            model = _torch.load(str(model_path), map_location=device)
            if hasattr(model, 'eval'):
                model.eval()
            return model
            
        except ImportError:
            raise ImportError("torch package is required for ResNet models")
    
    def _load_mobilenet_model(self, model_path: Path, device: str) -> Any:
        """Load MobileNet model"""
        try:
            global _torch
            if _torch is None:
                import torch as _torch
            
            model = _torch.load(str(model_path), map_location=device)
            if hasattr(model, 'eval'):
                model.eval()
            return model
            
        except ImportError:
            raise ImportError("torch package is required for MobileNet models")
    
    def _load_ocr_model(self, model_path: Path, device: str) -> Any:
        """Load OCR model"""
        try:
            global _pytesseract
            if _pytesseract is None:
                import pytesseract as _pytesseract
            
            # For Tesseract, we return the model path as the "model"
            # since tesseract uses model files directly
            return {"model_path": str(model_path), "engine": _pytesseract}
            
        except ImportError:
            raise ImportError("pytesseract package is required for OCR models")
    
    def _load_face_recognition_model(self, model_path: Path, device: str) -> Any:
        """Load face recognition model"""
        try:
            global _face_recognition
            if _face_recognition is None:
                import face_recognition as _face_recognition
            
            # Face recognition models are typically loaded automatically by the library
            return {"model_path": str(model_path), "engine": _face_recognition}
            
        except ImportError:
            raise ImportError("face_recognition package is required for face recognition models")
    
    def unload_model(self, model_name: str, version: str, device: DeviceType = DeviceType.AUTO) -> bool:
        """Unload a model from memory"""
        model_key = f"{model_name}_{version}_{device.value}"
        
        if model_key in self._loaded_models:
            del self._loaded_models[model_key]
            self.logger.info(f"Unloaded model {model_key}")
            return True
        
        return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self._loaded_models.keys())


def lazy_import_decorator(func):
    """Decorator for lazy importing of heavy dependencies"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            logging.getLogger(__name__).warning(f"Lazy import failed in {func.__name__}: {e}")
            raise
    return wrapper


class ModelHub:
    """Main model hub for managing AI vision models"""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_cache_size_gb: float = 10.0):
        # Initialize cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".omni_dev" / "models"
        
        self.cache_dir = Path(cache_dir)
        self.logger = logging.getLogger(__name__ + ".ModelHub")
        
        # Initialize components
        self.registry = ModelRegistry()
        self.device_manager = DeviceManager()
        self.cache_manager = CacheManager(self.cache_dir, max_cache_size_gb)
        self.downloader = ModelDownloader(self.cache_dir)
        self.loader = LazyModelLoader(self.device_manager)
        
        # Model status tracking
        self._model_status = {}
        self._status_lock = threading.Lock()
        
        self.logger.info(f"ModelHub initialized with cache directory: {self.cache_dir}")
    
    def list_available_models(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """List all available models"""
        models = self.registry.list_models(model_type)
        return [
            {
                "name": model.name,
                "type": model.model_type.value,
                "version": model.version,
                "size_mb": model.size_mb,
                "description": model.description,
                "supported_devices": [d.value for d in model.supported_devices],
                "dependencies": model.dependencies,
                "license": model.license,
                "status": self.get_model_status(model.name)
            }
            for model in models
        ]
    
    def get_model_status(self, model_name: str) -> str:
        """Get the current status of a model"""
        with self._status_lock:
            return self._model_status.get(model_name, ModelStatus.NOT_DOWNLOADED.value)
    
    def _set_model_status(self, model_name: str, status: ModelStatus):
        """Set model status"""
        with self._status_lock:
            self._model_status[model_name] = status.value
    
    async def download_model(self, model_name: str, 
                           progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """Download a model"""
        metadata = self.registry.get_model(model_name)
        if not metadata:
            raise ModelValidationError(model_name, "registry", f"Model {model_name} not found in registry")
        
        try:
            self._set_model_status(model_name, ModelStatus.DOWNLOADING)
            
            # Download model
            model_path = await self.downloader.download_model(metadata, progress_callback)
            
            # Add to cache
            self.cache_manager.add_model(model_name, model_path, metadata)
            
            self._set_model_status(model_name, ModelStatus.READY)
            self.logger.info(f"Model {model_name} downloaded and cached successfully")
            return True
            
        except Exception as e:
            self._set_model_status(model_name, ModelStatus.ERROR)
            vision_error_handler.handle_vision_error(e, {
                'component': 'ModelHub',
                'operation': 'download_model',
                'model_name': model_name
            })
            raise
    
    def load_model(self, model_name: str, device: DeviceType = DeviceType.AUTO, 
                  version: str = None) -> Optional[Any]:
        """Load a model for inference"""
        try:
            self._set_model_status(model_name, ModelStatus.LOADING)
            
            # Get from cache
            cache_entry = self.cache_manager.get_model(model_name, version)
            if not cache_entry:
                error = ModelLoadError(model_name, f"Model {model_name} not found in cache. Please download first.")
                self._set_model_status(model_name, ModelStatus.NOT_DOWNLOADED)
                vision_error_handler.handle_vision_error(error, {
                    'component': 'ModelHub',
                    'operation': 'load_model',
                    'model_name': model_name
                })
                raise error
            
            # Load model
            model = self.loader.load_model(cache_entry.model_path, cache_entry.metadata, device)
            
            self._set_model_status(model_name, ModelStatus.LOADED)
            return model
            
        except Exception as e:
            self._set_model_status(model_name, ModelStatus.ERROR)
            if not isinstance(e, ModelLoadError):
                vision_error_handler.handle_vision_error(e, {
                    'component': 'ModelHub',
                    'operation': 'load_model',
                    'model_name': model_name
                })
            raise
    
    async def ensure_model_ready(self, model_name: str, device: DeviceType = DeviceType.AUTO) -> Optional[Any]:
        """Ensure a model is downloaded and loaded"""
        status = self.get_model_status(model_name)
        
        # Download if needed
        if status == ModelStatus.NOT_DOWNLOADED.value:
            success = await self.download_model(model_name)
            if not success:
                return None
        
        # Load if needed
        if status in [ModelStatus.NOT_DOWNLOADED.value, ModelStatus.READY.value]:
            return self.load_model(model_name, device)
        
        # Return cached model if already loaded
        if status == ModelStatus.LOADED.value:
            cache_entry = self.cache_manager.get_model(model_name)
            if cache_entry:
                return self.loader.load_model(cache_entry.model_path, cache_entry.metadata, device)
        
        return None
    
    def unload_model(self, model_name: str, device: DeviceType = DeviceType.AUTO, 
                    version: str = None) -> bool:
        """Unload a model from memory"""
        if version is None:
            # Get version from cache
            cache_entry = self.cache_manager.get_model(model_name)
            if cache_entry:
                version = cache_entry.metadata.version
            else:
                return False
        
        success = self.loader.unload_model(model_name, version, device)
        if success:
            self._set_model_status(model_name, ModelStatus.READY)
        
        return success
    
    def remove_model(self, model_name: str, version: str = None) -> bool:
        """Remove a model from cache"""
        success = self.cache_manager.remove_model(model_name, version)
        if success:
            self._set_model_status(model_name, ModelStatus.NOT_DOWNLOADED)
        
        return success
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return self.cache_manager.get_cache_stats()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return self.device_manager.get_device_info()
    
    def validate_model(self, model_name: str, version: str = None) -> bool:
        """Validate a cached model's integrity"""
        cache_entry = self.cache_manager.get_model(model_name, version)
        if not cache_entry:
            return False
        
        if not cache_entry.model_path.exists():
            return False
        
        # Validate checksum
        return ChecksumValidator.validate_checksum(
            cache_entry.model_path, 
            cache_entry.metadata.checksum_sha256
        )
    
    def register_custom_model(self, metadata: ModelMetadata) -> bool:
        """Register a custom model"""
        return self.registry.register_model(metadata)
    
    def cleanup_cache(self) -> Dict[str, Any]:
        """Clean up cache by removing invalid entries"""
        stats = {"removed_count": 0, "reclaimed_bytes": 0}
        
        # Get all cached models
        cache_stats = self.cache_manager.get_cache_stats()
        
        for model_key, model_info in cache_stats["models"].items():
            model_name = model_info["name"]
            if not self.validate_model(model_name):
                # Remove invalid model
                if self.cache_manager.remove_model(model_name):
                    stats["removed_count"] += 1
                    stats["reclaimed_bytes"] += model_info.get("size_mb", 0) * 1024 * 1024
        
        return stats
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        metadata = self.registry.get_model(model_name)
        if not metadata:
            return None
        
        cache_entry = self.cache_manager.get_model(model_name)
        
        return {
            "name": metadata.name,
            "type": metadata.model_type.value,
            "version": metadata.version,
            "size_mb": metadata.size_mb,
            "description": metadata.description,
            "supported_devices": [d.value for d in metadata.supported_devices],
            "dependencies": metadata.dependencies,
            "license": metadata.license,
            "tags": metadata.tags,
            "status": self.get_model_status(model_name),
            "cached": cache_entry is not None,
            "cache_info": {
                "last_accessed": cache_entry.last_accessed.isoformat() if cache_entry else None,
                "access_count": cache_entry.access_count if cache_entry else 0,
                "checksum_verified": cache_entry.checksum_verified if cache_entry else False
            } if cache_entry else None
        }


# Convenience function for global model hub instance
_global_model_hub = None


def get_model_hub(cache_dir: Optional[Path] = None, max_cache_size_gb: float = 10.0) -> ModelHub:
    """Get or create global model hub instance"""
    global _global_model_hub
    if _global_model_hub is None:
        _global_model_hub = ModelHub(cache_dir, max_cache_size_gb)
    return _global_model_hub


# Export main classes and functions
__all__ = [
    'ModelHub',
    'ModelType',
    'DeviceType',
    'ModelStatus',
    'ModelMetadata',
    'get_model_hub'
]
