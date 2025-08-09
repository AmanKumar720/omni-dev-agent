# components/ai_vision/performance_optimizer.py

import torch
import torch.nn as nn
import numpy as np
import cv2
import asyncio
import threading
import queue
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import deque
import psutil
import GPUtil
import warnings
from pathlib import Path
import json

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Device type enumeration for computation"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon GPU
    AUTO = "auto"

class PrecisionType(Enum):
    """Precision type for inference"""
    FULL = "float32"
    HALF = "float16"
    MIXED = "mixed"
    AUTO = "auto"

class OptimizationLevel(Enum):
    """Optimization level settings"""
    NONE = 0
    BASIC = 1
    AGGRESSIVE = 2
    MAXIMUM = 3

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    device: DeviceType = DeviceType.AUTO
    precision: PrecisionType = PrecisionType.AUTO
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    batch_size: int = 4
    max_queue_size: int = 100
    max_workers: int = 4
    enable_tensorrt: bool = False
    enable_onnx: bool = False
    memory_fraction: float = 0.8
    async_processing: bool = True
    enable_profiling: bool = False

@dataclass
class FrameBatch:
    """Container for batched video frames"""
    frames: List[np.ndarray]
    metadata: List[Dict[str, Any]]
    timestamps: List[float]
    batch_id: str
    priority: int = 0

@dataclass
class ProcessingResult:
    """Result from batch processing"""
    batch_id: str
    results: List[Dict[str, Any]]
    processing_time: float
    device_used: str
    precision_used: str
    error: Optional[str] = None

class DeviceManager:
    """Manages device selection and configuration"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DeviceManager")
        self._device_info = None
        self._optimal_device = None
        
    def detect_optimal_device(self) -> str:
        """Detect the optimal device for computation"""
        if self._optimal_device:
            return self._optimal_device
            
        device_candidates = []
        
        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_devices = []
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'device': f'cuda:{i}',
                    'name': torch.cuda.get_device_name(i),
                    'memory': torch.cuda.get_device_properties(i).total_memory,
                    'compute_capability': torch.cuda.get_device_properties(i).major,
                    'score': 0
                }
                
                # Score based on memory and compute capability
                device_info['score'] = (
                    device_info['memory'] / (1024**3) * 10 +  # GB * 10
                    device_info['compute_capability'] * 20     # Compute capability * 20
                )
                
                cuda_devices.append(device_info)
                
            if cuda_devices:
                # Sort by score and select best
                cuda_devices.sort(key=lambda x: x['score'], reverse=True)
                device_candidates.extend(cuda_devices)
        
        # Check MPS (Apple Silicon) availability
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_candidates.append({
                'device': 'mps',
                'name': 'Apple Silicon GPU',
                'memory': psutil.virtual_memory().total,
                'score': 50  # Good baseline score for MPS
            })
        
        # CPU fallback
        cpu_cores = mp.cpu_count()
        cpu_memory = psutil.virtual_memory().total
        device_candidates.append({
            'device': 'cpu',
            'name': f'CPU ({cpu_cores} cores)',
            'memory': cpu_memory,
            'score': cpu_cores * 2  # Simple CPU scoring
        })
        
        # Select optimal device based on config
        if self.config.device == DeviceType.AUTO:
            # Auto-select best available device
            best_device = max(device_candidates, key=lambda x: x['score'])
            self._optimal_device = best_device['device']
        elif self.config.device == DeviceType.CUDA:
            cuda_devices = [d for d in device_candidates if 'cuda' in d['device']]
            if cuda_devices:
                self._optimal_device = cuda_devices[0]['device']
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                self._optimal_device = 'cpu'
        elif self.config.device == DeviceType.MPS:
            mps_devices = [d for d in device_candidates if d['device'] == 'mps']
            if mps_devices:
                self._optimal_device = 'mps'
            else:
                self.logger.warning("MPS requested but not available, falling back to CPU")
                self._optimal_device = 'cpu'
        else:
            self._optimal_device = 'cpu'
        
        self._device_info = next(
            d for d in device_candidates if d['device'] == self._optimal_device
        )
        
        self.logger.info(f"Selected device: {self._optimal_device} - {self._device_info['name']}")
        return self._optimal_device
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device"""
        if not self._device_info:
            self.detect_optimal_device()
        return self._device_info
    
    def configure_device_memory(self, device: str) -> None:
        """Configure device memory settings"""
        if 'cuda' in device:
            try:
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction,
                    device=torch.device(device)
                )
                
                # Enable memory management
                torch.cuda.empty_cache()
                
                self.logger.info(f"Configured CUDA memory fraction: {self.config.memory_fraction}")
            except Exception as e:
                self.logger.warning(f"Failed to configure CUDA memory: {e}")

class PrecisionManager:
    """Manages precision settings for models"""
    
    def __init__(self, config: PerformanceConfig, device: str):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.PrecisionManager")
        self._optimal_precision = None
    
    def detect_optimal_precision(self) -> str:
        """Detect optimal precision for the current device"""
        if self._optimal_precision:
            return self._optimal_precision
            
        if self.config.precision == PrecisionType.AUTO:
            if 'cuda' in self.device:
                # Check if device supports half precision
                device_props = torch.cuda.get_device_properties(self.device)
                if device_props.major >= 6:  # Pascal architecture or newer
                    self._optimal_precision = "float16"
                else:
                    self._optimal_precision = "float32"
            elif self.device == 'mps':
                self._optimal_precision = "float16"  # MPS supports half precision
            else:
                self._optimal_precision = "float32"  # CPU typically uses float32
        else:
            self._optimal_precision = self.config.precision.value
        
        self.logger.info(f"Selected precision: {self._optimal_precision}")
        return self._optimal_precision
    
    def apply_precision_to_model(self, model: nn.Module) -> nn.Module:
        """Apply precision settings to a model"""
        precision = self.detect_optimal_precision()
        
        try:
            if precision == "float16":
                model = model.half()
                self.logger.info("Applied half precision to model")
            elif precision == "float32":
                model = model.float()
                self.logger.info("Applied full precision to model")
            
            return model
        except Exception as e:
            self.logger.warning(f"Failed to apply precision {precision}: {e}")
            return model.float()  # Fallback to full precision

class AsyncFrameQueue:
    """Asynchronous queue for video frames with batch aggregation"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AsyncFrameQueue")
        self._queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._batch_queue = asyncio.Queue()
        self._processing_results = {}
        self._batch_counter = 0
        self._running = False
        
    async def start(self):
        """Start the async frame queue processing"""
        self._running = True
        # Start batch aggregation task
        asyncio.create_task(self._batch_aggregator())
        self.logger.info("AsyncFrameQueue started")
    
    async def stop(self):
        """Stop the async frame queue processing"""
        self._running = False
        self.logger.info("AsyncFrameQueue stopped")
    
    async def put_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Add a frame to the processing queue"""
        frame_id = f"frame_{int(time.time() * 1000000)}"
        frame_data = {
            'frame': frame,
            'metadata': metadata,
            'timestamp': time.time(),
            'frame_id': frame_id
        }
        
        try:
            await asyncio.wait_for(
                self._queue.put(frame_data),
                timeout=1.0
            )
            return frame_id
        except asyncio.TimeoutError:
            self.logger.warning("Frame queue is full, dropping frame")
            return None
    
    async def get_batch(self) -> Optional[FrameBatch]:
        """Get a batch of frames for processing"""
        try:
            return await asyncio.wait_for(
                self._batch_queue.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            return None
    
    async def _batch_aggregator(self):
        """Aggregate frames into batches"""
        current_batch = []
        last_batch_time = time.time()
        
        while self._running:
            try:
                # Try to get a frame
                frame_data = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1
                )
                
                current_batch.append(frame_data)
                
                # Check if batch is ready
                should_send_batch = (
                    len(current_batch) >= self.config.batch_size or
                    (len(current_batch) > 0 and 
                     time.time() - last_batch_time > 0.1)  # 100ms timeout
                )
                
                if should_send_batch:
                    await self._send_batch(current_batch)
                    current_batch = []
                    last_batch_time = time.time()
                    
            except asyncio.TimeoutError:
                # Send partial batch if timeout
                if current_batch:
                    await self._send_batch(current_batch)
                    current_batch = []
                    last_batch_time = time.time()
            except Exception as e:
                self.logger.error(f"Error in batch aggregator: {e}")
    
    async def _send_batch(self, frame_data_list: List[Dict[str, Any]]):
        """Send a batch to the processing queue"""
        self._batch_counter += 1
        batch = FrameBatch(
            frames=[fd['frame'] for fd in frame_data_list],
            metadata=[fd['metadata'] for fd in frame_data_list],
            timestamps=[fd['timestamp'] for fd in frame_data_list],
            batch_id=f"batch_{self._batch_counter}"
        )
        
        await self._batch_queue.put(batch)

class ModelOptimizer:
    """Optimizes models for better performance"""
    
    def __init__(self, config: PerformanceConfig, device: str):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.ModelOptimizer")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply optimizations to a model"""
        optimized_model = model
        
        try:
            # Apply compilation optimizations
            if self.config.optimization_level >= OptimizationLevel.BASIC:
                optimized_model = self._apply_basic_optimizations(optimized_model)
            
            if self.config.optimization_level >= OptimizationLevel.AGGRESSIVE:
                optimized_model = self._apply_aggressive_optimizations(optimized_model)
            
            if self.config.optimization_level >= OptimizationLevel.MAXIMUM:
                optimized_model = self._apply_maximum_optimizations(optimized_model)
            
            return optimized_model
        except Exception as e:
            self.logger.warning(f"Model optimization failed: {e}")
            return model
    
    def _apply_basic_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply basic optimizations"""
        # Set to evaluation mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply torch.jit.script if possible
        try:
            if hasattr(torch.jit, 'script'):
                model = torch.jit.script(model)
                self.logger.info("Applied torch.jit.script optimization")
        except Exception as e:
            self.logger.debug(f"Failed to apply jit.script: {e}")
        
        return model
    
    def _apply_aggressive_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply aggressive optimizations"""
        try:
            # Apply torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='max-autotune')
                self.logger.info("Applied torch.compile optimization")
        except Exception as e:
            self.logger.debug(f"Failed to apply torch.compile: {e}")
        
        return model
    
    def _apply_maximum_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply maximum optimizations"""
        # TensorRT optimization (if enabled and available)
        if self.config.enable_tensorrt and 'cuda' in self.device:
            try:
                import torch_tensorrt
                model = torch_tensorrt.compile(
                    model,
                    inputs=[torch.randn(1, 3, 640, 640).to(self.device)],
                    enabled_precisions={torch.float16} if "float16" in str(model.dtype) else {torch.float32}
                )
                self.logger.info("Applied TensorRT optimization")
            except ImportError:
                self.logger.debug("TensorRT not available")
            except Exception as e:
                self.logger.debug(f"Failed to apply TensorRT: {e}")
        
        return model

class PerformanceProfiler:
    """Profiles performance metrics"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = {}
        self.logger = logging.getLogger(f"{__name__}.PerformanceProfiler")
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        if not self.enabled:
            return
        self.metrics[name] = {'start_time': time.time()}
    
    def end_timer(self, name: str):
        """End timing an operation"""
        if not self.enabled or name not in self.metrics:
            return
        
        end_time = time.time()
        duration = end_time - self.metrics[name]['start_time']
        self.metrics[name]['duration'] = duration
        return duration
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric"""
        if not self.enabled:
            return
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()

class PerformanceOptimizer:
    """Main performance optimization class"""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger(f"{__name__}.PerformanceOptimizer")
        
        # Initialize components
        self.device_manager = DeviceManager(self.config)
        self.device = self.device_manager.detect_optimal_device()
        self.device_manager.configure_device_memory(self.device)
        
        self.precision_manager = PrecisionManager(self.config, self.device)
        self.model_optimizer = ModelOptimizer(self.config, self.device)
        self.profiler = PerformanceProfiler(self.config.enable_profiling)
        
        if self.config.async_processing:
            self.frame_queue = AsyncFrameQueue(self.config)
        
        self._models = {}
        self._processing_executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        self.logger.info(f"PerformanceOptimizer initialized with device: {self.device}")
    
    async def initialize(self):
        """Initialize async components"""
        if self.config.async_processing:
            await self.frame_queue.start()
    
    async def shutdown(self):
        """Shutdown optimizer"""
        if self.config.async_processing:
            await self.frame_queue.stop()
        self._processing_executor.shutdown(wait=True)
    
    def optimize_model(self, model: nn.Module, model_name: str = "model") -> nn.Module:
        """Optimize a model for performance"""
        self.profiler.start_timer(f"optimize_{model_name}")
        
        # Move model to optimal device
        model = model.to(self.device)
        
        # Apply precision settings
        model = self.precision_manager.apply_precision_to_model(model)
        
        # Apply optimizations
        model = self.model_optimizer.optimize_model(model)
        
        # Cache the model
        self._models[model_name] = model
        
        self.profiler.end_timer(f"optimize_{model_name}")
        self.logger.info(f"Model '{model_name}' optimized for {self.device}")
        
        return model
    
    async def process_frame_async(self, 
                                 frame: np.ndarray, 
                                 model_name: str,
                                 inference_func: Callable,
                                 metadata: Dict[str, Any] = None) -> Optional[str]:
        """Process a single frame asynchronously"""
        if not self.config.async_processing:
            raise ValueError("Async processing not enabled")
        
        metadata = metadata or {}
        metadata['model_name'] = model_name
        metadata['inference_func'] = inference_func
        
        return await self.frame_queue.put_frame(frame, metadata)
    
    async def process_batch_async(self, processor_func: Callable) -> Optional[ProcessingResult]:
        """Process a batch of frames asynchronously"""
        batch = await self.frame_queue.get_batch()
        if not batch:
            return None
        
        self.profiler.start_timer(f"batch_processing_{batch.batch_id}")
        
        try:
            # Get model and inference function from first frame metadata
            first_metadata = batch.metadata[0]
            model_name = first_metadata.get('model_name')
            inference_func = first_metadata.get('inference_func')
            
            if not model_name or not inference_func:
                raise ValueError("Missing model_name or inference_func in metadata")
            
            model = self._models.get(model_name)
            if not model:
                raise ValueError(f"Model '{model_name}' not found")
            
            # Process batch
            results = await self._process_frame_batch(
                batch.frames,
                model,
                inference_func,
                batch.metadata
            )
            
            processing_time = self.profiler.end_timer(f"batch_processing_{batch.batch_id}")
            
            return ProcessingResult(
                batch_id=batch.batch_id,
                results=results,
                processing_time=processing_time,
                device_used=self.device,
                precision_used=self.precision_manager.detect_optimal_precision()
            )
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch.batch_id}: {e}")
            return ProcessingResult(
                batch_id=batch.batch_id,
                results=[],
                processing_time=0,
                device_used=self.device,
                precision_used=self.precision_manager.detect_optimal_precision(),
                error=str(e)
            )
    
    async def _process_frame_batch(self,
                                  frames: List[np.ndarray],
                                  model: nn.Module,
                                  inference_func: Callable,
                                  metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of frames with the model"""
        
        # Prepare batch tensor
        batch_tensors = []
        for frame in frames:
            # Convert frame to tensor and preprocess
            if len(frame.shape) == 3:  # HWC format
                frame = cv2.resize(frame, (640, 640))  # Standard size
                frame = frame.transpose(2, 0, 1)  # HWC to CHW
            
            tensor = torch.from_numpy(frame).float() / 255.0
            if self.precision_manager.detect_optimal_precision() == "float16":
                tensor = tensor.half()
            
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Run inference
        with torch.no_grad():
            if hasattr(torch.cuda, 'amp') and 'cuda' in self.device:
                with torch.cuda.amp.autocast(enabled=(self.precision_manager.detect_optimal_precision() == "float16")):
                    batch_results = model(batch_tensor)
            else:
                batch_results = model(batch_tensor)
        
        # Process results
        results = []
        for i, (result, metadata) in enumerate(zip(batch_results, metadata_list)):
            processed_result = {
                'frame_index': i,
                'timestamp': metadata.get('timestamp', time.time()),
                'raw_result': result,
                'metadata': metadata
            }
            
            # Apply custom inference function if provided
            if inference_func:
                try:
                    custom_result = inference_func(result, frames[i], metadata)
                    processed_result.update(custom_result)
                except Exception as e:
                    self.logger.warning(f"Custom inference function failed: {e}")
            
            results.append(processed_result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = {
            'device_info': self.device_manager.get_device_info(),
            'precision': self.precision_manager.detect_optimal_precision(),
            'config': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'optimization_level': self.config.optimization_level.name
            }
        }
        
        # Add system stats
        if 'cuda' in self.device:
            try:
                gpu_stats = GPUtil.getGPUs()[0]
                stats['gpu_stats'] = {
                    'memory_used': gpu_stats.memoryUsed,
                    'memory_total': gpu_stats.memoryTotal,
                    'memory_percent': gpu_stats.memoryUtil * 100,
                    'temperature': gpu_stats.temperature,
                    'load': gpu_stats.load * 100
                }
            except Exception:
                pass
        
        # Add CPU stats
        stats['cpu_stats'] = {
            'usage_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'core_count': mp.cpu_count()
        }
        
        # Add profiler metrics
        if self.config.enable_profiling:
            stats['profiler_metrics'] = self.profiler.get_metrics()
        
        return stats
