#!/usr/bin/env python3
"""
Example Performance Optimization Integration

This example demonstrates how to integrate the performance optimization system
with camera streams and AI vision models for real-time video processing.

Features demonstrated:
- GPU acceleration with CUDA auto-fallback to CPU
- Half-precision inference for improved performance
- Batch aggregation for efficient processing
- Asynchronous queue for video frames
- Real-time performance monitoring
"""

import asyncio
import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path

# Import performance optimization components
try:
    from components.ai_vision.performance_optimizer import (
        PerformanceOptimizer,
        PerformanceConfig,
        DeviceType,
        PrecisionType,
        OptimizationLevel
    )
    from components.device_manager import CameraStream, StreamConfig, FrameData
    from components.ai_vision.object_detection import ObjectDetector
    from components.ai_vision.image_classification import ImageClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all components are properly installed and accessible")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for processing performance"""
    frames_processed: int = 0
    total_processing_time: float = 0.0
    avg_fps: float = 0.0
    current_fps: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0

class OptimizedVideoProcessor:
    """Optimized video processor using performance optimization system"""
    
    def __init__(self, config: PerformanceConfig = None):
        # Create performance configuration
        self.perf_config = config or PerformanceConfig(
            device=DeviceType.AUTO,
            precision=PrecisionType.AUTO,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            batch_size=8,
            max_queue_size=50,
            async_processing=True,
            enable_profiling=True
        )
        
        # Initialize performance optimizer
        self.optimizer = PerformanceOptimizer(self.perf_config)
        
        # Initialize AI models
        self.object_detector = None
        self.image_classifier = None
        
        # Statistics tracking
        self.stats = ProcessingStats()
        self.last_stats_update = time.time()
        
        # Processing callbacks
        self.result_callbacks = []
        
        logger.info(f"OptimizedVideoProcessor initialized with {self.perf_config.device.value} device")
    
    async def initialize(self):
        """Initialize the video processor"""
        await self.optimizer.initialize()
        
        # Initialize and optimize AI models
        logger.info("Loading and optimizing AI models...")
        
        # Load object detection model
        try:
            self.object_detector = ObjectDetector()
            if hasattr(self.object_detector, 'model'):
                optimized_model = self.optimizer.optimize_model(
                    self.object_detector.model, 
                    "object_detector"
                )
                self.object_detector.model = optimized_model
        except Exception as e:
            logger.warning(f"Failed to load object detector: {e}")
        
        # Load image classification model
        try:
            self.image_classifier = ImageClassifier()
            if hasattr(self.image_classifier, 'model'):
                optimized_model = self.optimizer.optimize_model(
                    self.image_classifier.model,
                    "image_classifier"
                )
                self.image_classifier.model = optimized_model
        except Exception as e:
            logger.warning(f"Failed to load image classifier: {e}")
        
        logger.info("AI models loaded and optimized")
    
    async def shutdown(self):
        """Shutdown the video processor"""
        await self.optimizer.shutdown()
        logger.info("OptimizedVideoProcessor shutdown complete")
    
    def add_result_callback(self, callback):
        """Add a callback for processing results"""
        self.result_callbacks.append(callback)
    
    async def process_camera_stream(self, camera_id: int = 0, duration_seconds: int = 30):
        """Process live camera stream with optimization"""
        
        # Configure camera stream
        stream_config = StreamConfig(
            camera_id=camera_id,
            resolution=(640, 640),
            fps=30,
            buffer_size=10
        )
        
        camera_stream = CameraStream(stream_config)
        
        # Add frame callback to process frames
        camera_stream.add_frame_callback(self._on_frame_received)
        
        # Start camera stream
        if not camera_stream.start():
            logger.error("Failed to start camera stream")
            return
        
        logger.info(f"Processing camera stream for {duration_seconds} seconds...")
        
        # Start processing loop
        processing_task = asyncio.create_task(self._processing_loop())
        stats_task = asyncio.create_task(self._stats_loop())
        
        try:
            # Wait for specified duration
            await asyncio.sleep(duration_seconds)
        finally:
            # Cleanup
            camera_stream.stop()
            processing_task.cancel()
            stats_task.cancel()
            
            # Wait for tasks to complete
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
            
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Camera stream processing completed")
    
    async def process_video_file(self, video_path: str):
        """Process video file with optimization"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        
        # Start processing loop
        processing_task = asyncio.create_task(self._processing_loop())
        stats_task = asyncio.create_task(self._stats_loop())
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to match expected input size
                frame = cv2.resize(frame, (640, 640))
                
                # Submit frame for processing
                await self._on_frame_received_async(frame, {
                    'frame_index': frame_count,
                    'video_path': video_path,
                    'timestamp': time.time()
                })
                
                frame_count += 1
                
                # Limit processing rate to prevent queue overflow
                if frame_count % 10 == 0:
                    await asyncio.sleep(0.01)
            
            logger.info(f"Submitted {frame_count} frames for processing")
            
            # Wait for processing to complete
            await asyncio.sleep(5.0)
            
        finally:
            cap.release()
            processing_task.cancel()
            stats_task.cancel()
            
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
            
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Video file processing completed")
    
    def _on_frame_received(self, frame_data: FrameData):
        """Callback for received camera frames"""
        asyncio.create_task(self._on_frame_received_async(
            frame_data.frame,
            {
                'camera_id': frame_data.camera_id,
                'frame_number': frame_data.frame_number,
                'timestamp': frame_data.timestamp,
                'metadata': frame_data.metadata
            }
        ))
    
    async def _on_frame_received_async(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Async handler for received frames"""
        # Submit frame to performance optimizer
        frame_id = await self.optimizer.process_frame_async(
            frame,
            "object_detector",  # Default to object detection
            self._inference_function,
            metadata
        )
        
        if not frame_id:
            logger.warning("Failed to submit frame for processing")
    
    def _inference_function(self, model_output, frame, metadata):
        """Custom inference function for processing results"""
        results = {
            'timestamp': metadata.get('timestamp', time.time()),
            'frame_shape': frame.shape,
            'processing_device': self.optimizer.device
        }
        
        # Process object detection results
        if self.object_detector and hasattr(model_output, 'shape'):
            try:
                # Simulate object detection processing
                detections = []
                if len(model_output.shape) >= 2:
                    # Mock detection results
                    num_detections = min(5, model_output.shape[-1])
                    for i in range(num_detections):
                        detection = {
                            'class_id': i % 80,  # COCO classes
                            'confidence': float(np.random.random()),
                            'bbox': [
                                int(np.random.random() * frame.shape[1]),
                                int(np.random.random() * frame.shape[0]),
                                int(np.random.random() * 100),
                                int(np.random.random() * 100)
                            ]
                        }
                        detections.append(detection)
                
                results['detections'] = detections
                results['num_detections'] = len(detections)
                
            except Exception as e:
                logger.warning(f"Object detection processing failed: {e}")
        
        return results
    
    async def _processing_loop(self):
        """Main processing loop for handling batched results"""
        while True:
            try:
                # Process batch
                result = await self.optimizer.process_batch_async(self._inference_function)
                
                if result and not result.error:
                    # Update statistics
                    self.stats.frames_processed += len(result.results)
                    self.stats.total_processing_time += result.processing_time
                    
                    # Notify callbacks
                    for callback in self.result_callbacks:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.warning(f"Result callback failed: {e}")
                
                elif result and result.error:
                    logger.error(f"Processing batch error: {result.error}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _stats_loop(self):
        """Statistics monitoring loop"""
        while True:
            try:
                current_time = time.time()
                time_diff = current_time - self.last_stats_update
                
                if time_diff >= 2.0:  # Update every 2 seconds
                    # Calculate FPS
                    if time_diff > 0:
                        frames_since_update = self.stats.frames_processed - getattr(self, '_last_frame_count', 0)
                        self.stats.current_fps = frames_since_update / time_diff
                        self._last_frame_count = self.stats.frames_processed
                    
                    # Calculate average FPS
                    if self.stats.total_processing_time > 0:
                        self.stats.avg_fps = self.stats.frames_processed / self.stats.total_processing_time
                    
                    # Get system stats
                    perf_stats = self.optimizer.get_performance_stats()
                    
                    # Update GPU utilization
                    gpu_stats = perf_stats.get('gpu_stats', {})
                    self.stats.gpu_utilization = gpu_stats.get('load', 0.0)
                    
                    # Update memory usage
                    cpu_stats = perf_stats.get('cpu_stats', {})
                    self.stats.memory_usage = cpu_stats.get('memory_percent', 0.0)
                    
                    # Log statistics
                    logger.info(f"Stats - Frames: {self.stats.frames_processed}, "
                              f"Current FPS: {self.stats.current_fps:.2f}, "
                              f"Avg FPS: {self.stats.avg_fps:.2f}, "
                              f"GPU: {self.stats.gpu_utilization:.1f}%, "
                              f"Memory: {self.stats.memory_usage:.1f}%")
                    
                    self.last_stats_update = current_time
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats loop error: {e}")
                await asyncio.sleep(1.0)

async def result_callback(processing_result):
    """Example result callback"""
    logger.debug(f"Processed batch {processing_result.batch_id} with "
                f"{len(processing_result.results)} frames in "
                f"{processing_result.processing_time:.4f}s")

async def main():
    """Main example function"""
    logger.info("Starting Performance Optimization Example")
    
    # Create optimized video processor
    processor = OptimizedVideoProcessor()
    
    # Add result callback
    processor.add_result_callback(result_callback)
    
    try:
        # Initialize processor
        await processor.initialize()
        
        # Demo 1: Process live camera stream
        logger.info("=== Demo 1: Live Camera Stream ===")
        try:
            await processor.process_camera_stream(camera_id=0, duration_seconds=15)
        except Exception as e:
            logger.warning(f"Camera stream demo failed: {e}")
        
        # Demo 2: Process video file (if available)
        logger.info("=== Demo 2: Video File Processing ===")
        video_path = "sample_video.mp4"  # Replace with actual video path
        if Path(video_path).exists():
            await processor.process_video_file(video_path)
        else:
            logger.info(f"Video file {video_path} not found, skipping demo")
        
        # Show final statistics
        logger.info("=== Final Statistics ===")
        stats = processor.stats
        perf_stats = processor.optimizer.get_performance_stats()
        
        print(f"""
Performance Summary:
  Total frames processed: {stats.frames_processed}
  Average FPS: {stats.avg_fps:.2f}
  Total processing time: {stats.total_processing_time:.2f}s
  Device used: {perf_stats.get('device_info', {}).get('device', 'unknown')}
  Precision: {perf_stats.get('precision', 'unknown')}
  Final GPU utilization: {stats.gpu_utilization:.1f}%
  Final memory usage: {stats.memory_usage:.1f}%
        """)
        
    finally:
        # Shutdown processor
        await processor.shutdown()
        logger.info("Performance Optimization Example completed")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
