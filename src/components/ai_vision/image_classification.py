# src/components/ai_vision/image_classification.py

"""
Image Classification Module using TorchVision Pretrained CNNs

This module provides image classification capabilities using pretrained models with support for:
- ResNet50 and MobileNetV3 architectures
- Top-k predictions with probabilities
- Fine-tuning hooks for custom datasets
- Batch image processing
- GPU acceleration when available
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass
import time
import numpy as np

# Core imports - handle both relative and absolute imports
try:
    from .core import VisionTask, VisionResult, TaskStatus
    from .model_hub import ModelHub, get_model_hub, ModelType, DeviceType
except ImportError:
    # Fallback for standalone execution
    try:
        from core import VisionTask, VisionResult, TaskStatus
        from model_hub import ModelHub, get_model_hub, ModelType, DeviceType
    except ImportError:
        # Create minimal fallback classes for standalone testing
        from typing import Any, Optional
        from abc import ABC, abstractmethod
        from enum import Enum
        
        class TaskStatus(Enum):
            PENDING = "pending"
            IN_PROGRESS = "in_progress" 
            COMPLETED = "completed"
            FAILED = "failed"
        
        class VisionResult:
            def __init__(self, task_id: str, status: TaskStatus, data: Any, confidence: float, 
                        metadata: Optional[Dict[str, Any]] = None, error_message: Optional[str] = None):
                self.task_id = task_id
                self.status = status
                self.data = data
                self.confidence = confidence
                self.metadata = metadata
                self.error_message = error_message
        
        class VisionTask(ABC):
            def __init__(self, task_id: str, task_type: str, **kwargs):
                self.task_id = task_id
                self.task_type = task_type
                self.status = TaskStatus.PENDING
                self.result = None
                self.metadata = kwargs
                import logging
                self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            
            @abstractmethod
            async def execute(self, input_data: Any) -> VisionResult:
                pass
            
            @abstractmethod
            def validate_input(self, input_data: Any) -> bool:
                pass
        
        class DeviceType(Enum):
            CPU = "cpu"
            GPU = "cuda"
            AUTO = "auto"
        
        class ModelType(Enum):
            RESNET50 = "resnet50"
            MOBILENET_V3 = "mobilenet_v3"
        
        class ModelHub:
            def __init__(self, cache_dir=None, max_cache_size_gb=10.0):
                pass
            
            async def ensure_model_ready(self, model_name: str, device_type=None):
                return None
        
        def get_model_hub():
            return ModelHub()

# Lazy imports for heavy dependencies
_torch = None
_torchvision = None
_transforms = None
_PIL = None


def _lazy_import_torch():
    """Lazy import PyTorch"""
    global _torch
    if _torch is None:
        try:
            import torch as _torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for image classification. "
                "Install with: pip install torch torchvision"
            )
    return _torch


def _lazy_import_torchvision():
    """Lazy import TorchVision"""
    global _torchvision, _transforms
    if _torchvision is None:
        try:
            import torchvision as _torchvision
            from torchvision import transforms as _transforms
        except ImportError:
            raise ImportError(
                "TorchVision is required for image classification. "
                "Install with: pip install torchvision"
            )
    return _torchvision, _transforms


def _lazy_import_pil():
    """Lazy import PIL"""
    global _PIL
    if _PIL is None:
        try:
            from PIL import Image as _PIL
        except ImportError:
            raise ImportError(
                "Pillow is required for image processing. "
                "Install with: pip install Pillow"
            )
    return _PIL


class ClassificationModel(Enum):
    """Supported classification model architectures"""
    RESNET50 = "resnet50"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"


@dataclass
class ClassificationResult:
    """Container for a single classification prediction"""
    class_id: int
    class_name: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence
        }


@dataclass
class TopKResult:
    """Container for top-k classification results"""
    predictions: List[ClassificationResult]
    processing_time: float
    k: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'predictions': [pred.to_dict() for pred in self.predictions],
            'processing_time': self.processing_time,
            'k': self.k
        }


@dataclass
class BatchClassificationResult:
    """Container for batch classification results"""
    results: List[TopKResult]  # List of top-k results per image
    processing_time: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'results': [result.to_dict() for result in self.results],
            'processing_time': self.processing_time,
            'batch_size': self.batch_size
        }


class FineTuningHook:
    """Base class for fine-tuning hooks"""
    
    def on_training_start(self, model, optimizer, train_loader, val_loader=None):
        """Called at the start of training"""
        pass
    
    def on_epoch_start(self, epoch: int, model, optimizer):
        """Called at the start of each epoch"""
        pass
    
    def on_batch_start(self, batch_idx: int, batch, model):
        """Called at the start of each batch"""
        pass
    
    def on_batch_end(self, batch_idx: int, batch, outputs, loss, model):
        """Called at the end of each batch"""
        pass
    
    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: Optional[float], model):
        """Called at the end of each epoch"""
        pass
    
    def on_training_end(self, model, final_metrics: Dict[str, float]):
        """Called at the end of training"""
        pass


class LoggingHook(FineTuningHook):
    """Hook that logs training progress"""
    
    def __init__(self, log_every_n_batches: int = 100):
        self.log_every_n_batches = log_every_n_batches
        self.logger = logging.getLogger(__name__ + ".LoggingHook")
    
    def on_training_start(self, model, optimizer, train_loader, val_loader=None):
        try:
            self.logger.info(f"Starting training with {len(train_loader)} batches")
            if val_loader:
                self.logger.info(f"Validation set has {len(val_loader)} batches")
        except (TypeError, AttributeError):
            # Handle cases where train_loader might be a Mock or invalid object
            self.logger.info("Starting training")
    
    def on_epoch_start(self, epoch: int, model, optimizer):
        self.logger.info(f"Epoch {epoch} started")
    
    def on_batch_end(self, batch_idx: int, batch, outputs, loss, model):
        if batch_idx % self.log_every_n_batches == 0:
            self.logger.info(f"Batch {batch_idx}, Loss: {loss:.4f}")
    
    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: Optional[float], model):
        if val_loss is not None:
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")


class CheckpointHook(FineTuningHook):
    """Hook that saves model checkpoints"""
    
    def __init__(self, checkpoint_dir: Union[str, Path], save_every_n_epochs: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.logger = logging.getLogger(__name__ + ".CheckpointHook")
    
    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: Optional[float], model):
        if epoch % self.save_every_n_epochs == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch = _lazy_import_torch()
            torch.save(model.state_dict(), checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def on_training_end(self, model, final_metrics: Dict[str, float]):
        checkpoint_path = self.checkpoint_dir / "final_model.pth"
        torch = _lazy_import_torch()
        torch.save(model.state_dict(), checkpoint_path)
        self.logger.info(f"Saved final model: {checkpoint_path}")


class ImageClassificationTask(VisionTask):
    """Vision task implementation for image classification"""
    
    def __init__(self, task_id: str, model_name: str = "resnet50", **kwargs):
        super().__init__(task_id, "image_classification", **kwargs)
        self.model_name = model_name
        self.model = None
        self.model_hub = get_model_hub()
        
    async def execute(self, input_data: Any) -> VisionResult:
        """Execute image classification task"""
        try:
            if not self.model:
                await self._ensure_model_loaded()
            
            # Handle different input types
            if isinstance(input_data, dict):
                image = input_data.get('image')
                k = input_data.get('k', 5)
                batch_images = input_data.get('batch_images')
                
                if batch_images is not None:
                    # Batch processing
                    results = classify_images_batch(batch_images, self.model, k)
                    return VisionResult(
                        task_id=self.task_id,
                        status=TaskStatus.COMPLETED,
                        data=results.to_dict(),
                        confidence=1.0,
                        metadata={'batch_size': len(batch_images), 'k': k}
                    )
                else:
                    # Single image processing
                    result = classify_image(image, self.model, k)
                    return VisionResult(
                        task_id=self.task_id,
                        status=TaskStatus.COMPLETED,
                        data=result.to_dict(),
                        confidence=result.predictions[0].confidence if result.predictions else 0.0,
                        metadata={'k': k}
                    )
            else:
                # Direct image input
                result = classify_image(input_data, self.model)
                return VisionResult(
                    task_id=self.task_id,
                    status=TaskStatus.COMPLETED,
                    data=result.to_dict(),
                    confidence=result.predictions[0].confidence if result.predictions else 0.0,
                    metadata={'k': 5}
                )
        
        except Exception as e:
            self.logger.error(f"Classification task failed: {e}")
            return VisionResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                data=None,
                confidence=0.0,
                error_message=str(e)
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        try:
            if isinstance(input_data, dict):
                has_image = 'image' in input_data
                has_batch = 'batch_images' in input_data
                return has_image or has_batch
            else:
                # Assume it's an image (numpy array, PIL Image, etc.)
                return input_data is not None
        except Exception:
            return False
        
        return True
    
    async def _ensure_model_loaded(self):
        """Ensure the model is loaded"""
        try:
            self.model = await self.model_hub.ensure_model_ready(self.model_name)
            if self.model is None:
                raise RuntimeError(f"Failed to load model {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise


class ImageClassifier:
    """Main image classification class using TorchVision pretrained models"""
    
    def __init__(self, 
                 model_name: str = "resnet50", 
                 device: str = "auto", 
                 model_hub: Optional[ModelHub] = None,
                 num_classes: int = 1000):
        """
        Initialize image classifier
        
        Args:
            model_name: Model architecture ('resnet50', 'mobilenet_v3_large', 'mobilenet_v3_small')
            device: Device to run on ('cpu', 'cuda', 'auto')
            model_hub: Optional ModelHub instance
            num_classes: Number of output classes (default: 1000 for ImageNet)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device_type = DeviceType.AUTO if device == "auto" else DeviceType.GPU if device == "cuda" else DeviceType.CPU
        self.model_hub = model_hub or get_model_hub()
        self.model = None
        self.device = None
        self.transform = None
        self._model_lock = threading.Lock()
        self.logger = logging.getLogger(__name__ + ".ImageClassifier")
        
        # Load ImageNet class names (doesn't require dependencies)
        self._load_class_names()
    
    def _init_transforms(self):
        """Initialize image preprocessing transforms"""
        _, transforms = _lazy_import_torchvision()
        
        if self.model_name == "resnet50":
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif "mobilenet" in self.model_name:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Default transform
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_class_names(self):
        """Load ImageNet class names"""
        # ImageNet class names (simplified subset for demo)
        base_class_names = [
            'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
            'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch',
            'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie',
            'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl'
            # ... truncated for brevity - in a real implementation, load all 1000 classes
        ]
        
        if self.num_classes <= len(base_class_names):
            # Use only the first num_classes names
            self.class_names = base_class_names[:self.num_classes]
        else:
            # Use all base names and extend with generic class names
            self.class_names = base_class_names.copy()
            while len(self.class_names) < self.num_classes:
                self.class_names.append(f"class_{len(self.class_names)}")
    
    async def ensure_model_ready(self) -> bool:
        """Ensure model is downloaded and loaded"""
        try:
            with self._model_lock:
                if self.model is None:
                    self.logger.info(f"Loading {self.model_name} model")
                    await self._load_model()
                    self._setup_device()
                    self.logger.info(f"Model {self.model_name} loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to ensure model ready: {e}")
            return False
    
    async def _load_model(self):
        """Load the pretrained model"""
        torch = _lazy_import_torch()
        torchvision, _ = _lazy_import_torchvision()
        
        if self.model_name == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
        elif self.model_name == "mobilenet_v3_large":
            self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        elif self.model_name == "mobilenet_v3_small":
            self.model = torchvision.models.mobilenet_v3_small(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Modify final layer if needed
        if self.num_classes != 1000:
            if "resnet" in self.model_name:
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
            elif "mobilenet" in self.model_name:
                self.model.classifier[-1] = torch.nn.Linear(
                    self.model.classifier[-1].in_features, self.num_classes
                )
        
        self.model.eval()
    
    def _setup_device(self):
        """Setup computing device"""
        torch = _lazy_import_torch()
        
        if self.device_type == DeviceType.AUTO:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device_type == DeviceType.GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
    
    def _preprocess_image(self, image) -> Any:
        """Preprocess image for model input"""
        PIL = _lazy_import_pil()
        
        # Initialize transforms if not already done
        if self.transform is None:
            self._init_transforms()
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = PIL.fromarray(image)
        elif not hasattr(image, 'mode'):  # Not a PIL Image
            raise ValueError("Image must be PIL Image or numpy array")
        
        # Apply transforms
        return self.transform(image)
    
    def classify_image(self, image, k: int = 5) -> TopKResult:
        """
        Classify a single image with top-k predictions
        
        Args:
            image: Input image (PIL Image or numpy array)
            k: Number of top predictions to return
            
        Returns:
            TopKResult with top-k predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call ensure_model_ready() first.")
        
        torch = _lazy_import_torch()
        start_time = time.time()
        
        try:
            # Preprocess image
            input_tensor = self._preprocess_image(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top-k predictions
                top_k_probs, top_k_indices = torch.topk(probabilities, k)
                
                # Convert to results
                predictions = []
                for i in range(k):
                    class_id = top_k_indices[i].item()
                    confidence = top_k_probs[i].item()
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    predictions.append(ClassificationResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence
                    ))
            
            processing_time = time.time() - start_time
            
            return TopKResult(
                predictions=predictions,
                processing_time=processing_time,
                k=k
            )
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            raise
    
    def classify_images_batch(self, images: List[Any], k: int = 5) -> BatchClassificationResult:
        """
        Classify a batch of images
        
        Args:
            images: List of input images (PIL Images or numpy arrays)
            k: Number of top predictions to return per image
            
        Returns:
            BatchClassificationResult with results for each image
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call ensure_model_ready() first.")
        
        torch = _lazy_import_torch()
        start_time = time.time()
        batch_results = []
        
        try:
            # Preprocess all images
            input_tensors = []
            for image in images:
                input_tensor = self._preprocess_image(image)
                input_tensors.append(input_tensor)
            
            # Create batch
            input_batch = torch.stack(input_tensors).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Process each image's results
                for i, probs in enumerate(probabilities):
                    top_k_probs, top_k_indices = torch.topk(probs, k)
                    
                    predictions = []
                    for j in range(k):
                        class_id = top_k_indices[j].item()
                        confidence = top_k_probs[j].item()
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        predictions.append(ClassificationResult(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence
                        ))
                    
                    batch_results.append(TopKResult(
                        predictions=predictions,
                        processing_time=0.0,  # Individual timing not calculated in batch
                        k=k
                    ))
            
            processing_time = time.time() - start_time
            
            return BatchClassificationResult(
                results=batch_results,
                processing_time=processing_time,
                batch_size=len(images)
            )
            
        except Exception as e:
            self.logger.error(f"Batch classification failed: {e}")
            raise
    
    def fine_tune(self, 
                  train_loader, 
                  val_loader=None, 
                  epochs: int = 10,
                  learning_rate: float = 0.001,
                  hooks: Optional[List[FineTuningHook]] = None,
                  optimizer=None,
                  criterion=None) -> Dict[str, float]:
        """
        Fine-tune the model on custom data
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            hooks: List of training hooks for callbacks
            optimizer: Custom optimizer (if None, uses Adam)
            criterion: Loss criterion (if None, uses CrossEntropyLoss)
            
        Returns:
            Dictionary with final training metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call ensure_model_ready() first.")
        
        torch = _lazy_import_torch()
        
        # Setup optimizer and criterion
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize hooks
        hooks = hooks or [LoggingHook()]
        
        # Call training start hooks
        for hook in hooks:
            hook.on_training_start(self.model, optimizer, train_loader, val_loader)
        
        # Training loop
        final_metrics = {}
        
        for epoch in range(epochs):
            # Call epoch start hooks
            for hook in hooks:
                hook.on_epoch_start(epoch, self.model, optimizer)
            
            # Training phase
            self.model.train()
            total_train_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Call batch start hooks
                for hook in hooks:
                    hook.on_batch_start(batch_idx, (inputs, targets), self.model)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Call batch end hooks
                for hook in hooks:
                    hook.on_batch_end(batch_idx, (inputs, targets), outputs, loss.item(), self.model)
            
            train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        total_val_loss += loss.item()
                
                val_loss = total_val_loss / len(val_loader)
            
            # Call epoch end hooks
            for hook in hooks:
                hook.on_epoch_end(epoch, train_loss, val_loss, self.model)
            
            final_metrics[f'epoch_{epoch}_train_loss'] = train_loss
            if val_loss is not None:
                final_metrics[f'epoch_{epoch}_val_loss'] = val_loss
        
        # Set final metrics
        final_metrics['final_train_loss'] = train_loss
        if val_loss is not None:
            final_metrics['final_val_loss'] = val_loss
        
        # Call training end hooks
        for hook in hooks:
            hook.on_training_end(self.model, final_metrics)
        
        return final_metrics
    
    def save_model(self, path: Union[str, Path]):
        """Save the model state"""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        torch = _lazy_import_torch()
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load model state"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call ensure_model_ready() first.")
        
        torch = _lazy_import_torch()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info(f"Model loaded from {path}")


# Convenience functions
async def create_classifier(model_name: str = "resnet50", device: str = "auto") -> ImageClassifier:
    """
    Create and initialize an image classifier
    
    Args:
        model_name: Model architecture name
        device: Device to run on
        
    Returns:
        Initialized ImageClassifier instance
    """
    classifier = ImageClassifier(model_name=model_name, device=device)
    success = await classifier.ensure_model_ready()
    if not success:
        raise RuntimeError(f"Failed to initialize classifier with model {model_name}")
    return classifier


def classify_image(image, model: Any = None, k: int = 5) -> TopKResult:
    """
    Convenience function for single image classification
    
    Args:
        image: Input image (PIL Image or numpy array)
        model: Optional pre-loaded model (if None, uses a default classifier)
        k: Number of top predictions to return
        
    Returns:
        TopKResult with top-k predictions
    """
    if model is None:
        raise ValueError("Model must be provided or use ImageClassifier class")
    
    # Create temporary classifier with the model
    classifier = ImageClassifier()
    classifier.model = model
    
    return classifier.classify_image(image, k)


def classify_images_batch(images: List[Any], model: Any = None, k: int = 5) -> BatchClassificationResult:
    """
    Convenience function for batch image classification
    
    Args:
        images: List of input images (PIL Images or numpy arrays)
        model: Optional pre-loaded model (if None, uses a default classifier)
        k: Number of top predictions to return per image
        
    Returns:
        BatchClassificationResult with results for each image
    """
    if model is None:
        raise ValueError("Model must be provided or use ImageClassifier class")
    
    # Create temporary classifier with the model
    classifier = ImageClassifier()
    classifier.model = model
    
    return classifier.classify_images_batch(images, k)


# Export main classes and functions
__all__ = [
    'ImageClassifier',
    'ImageClassificationTask', 
    'ClassificationResult',
    'TopKResult',
    'BatchClassificationResult',
    'FineTuningHook',
    'LoggingHook',
    'CheckpointHook',
    'ClassificationModel',
    'classify_image',
    'classify_images_batch',
    'create_classifier'
]
