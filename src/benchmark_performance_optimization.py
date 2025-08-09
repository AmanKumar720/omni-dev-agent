#!/usr/bin/env python3
"""
Performance Optimization Benchmarking Script

This script tests GPU acceleration with CUDA, auto-fallback to CPU, half-precision inference,
batch aggregation, and asynchronous queue for video frames.

Usage:
    python benchmark_performance_optimization.py [--config config.json] [--output results.json]
"""

import asyncio
import json
import time
import logging
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import asdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import os
import psutil
import GPUtil

# Import our performance optimizer
try:
    from components.ai_vision.performance_optimizer import (
        PerformanceOptimizer,
        PerformanceConfig,
        DeviceType,
        PrecisionType,
        OptimizationLevel
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('components/ai_vision')
    from performance_optimizer import (
        PerformanceOptimizer,
        PerformanceConfig,
        DeviceType,
        PrecisionType,
        OptimizationLevel
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTestModel(nn.Module):
    """Simple test model for benchmarking"""
    
    def __init__(self, input_size: int = 640, num_classes: int = 80):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def generate_test_frames(num_frames: int, resolution: Tuple[int, int] = (640, 640)) -> List[np.ndarray]:
    """Generate synthetic test frames"""
    frames = []
    for i in range(num_frames):
        # Generate random frame with some pattern
        frame = np.random.randint(0, 255, (*resolution, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        cv2.rectangle(frame, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.circle(frame, (400, 300), 50, (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames

def simple_inference_function(model_output, frame, metadata):
    """Simple inference function for testing"""
    predictions = torch.softmax(model_output, dim=-1)
    top_pred = torch.argmax(predictions, dim=-1)
    confidence = torch.max(predictions, dim=-1)[0]
    
    return {
        'predictions': predictions.cpu().numpy().tolist(),
        'top_class': top_pred.cpu().item(),
        'confidence': confidence.cpu().item(),
        'frame_shape': frame.shape
    }

class BenchmarkSuite:
    """Comprehensive benchmark suite for performance optimization"""
    
    def __init__(self, config_file: str = None, output_file: str = None):
        self.config_file = config_file
        self.output_file = output_file or f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.results = {}
        
        # Load configuration
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.benchmark_config = json.load(f)
        else:
            self.benchmark_config = self._default_benchmark_config()
        
        logger.info(f"Benchmark suite initialized. Results will be saved to: {self.output_file}")
    
    def _default_benchmark_config(self) -> Dict[str, Any]:
        """Default benchmark configuration"""
        return {
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
                    "name": "optimized_cpu",
                    "device": "CPU",
                    "precision": "FULL",
                    "optimization_level": "AGGRESSIVE",
                    "batch_size": 4,
                    "async_processing": True
                },
                {
                    "name": "baseline_gpu",
                    "device": "CUDA",
                    "precision": "FULL",
                    "optimization_level": "NONE",
                    "batch_size": 1,
                    "async_processing": False
                },
                {
                    "name": "optimized_gpu",
                    "device": "CUDA",
                    "precision": "HALF",
                    "optimization_level": "AGGRESSIVE",
                    "batch_size": 8,
                    "async_processing": True
                },
                {
                    "name": "maximum_optimization",
                    "device": "AUTO",
                    "precision": "AUTO",
                    "optimization_level": "MAXIMUM",
                    "batch_size": 16,
                    "async_processing": True,
                    "enable_tensorrt": True
                }
            ],
            "test_parameters": {
                "num_frames": 100,
                "frame_resolution": [640, 640],
                "warmup_iterations": 10,
                "measurement_iterations": 5,
                "timeout_seconds": 300
            }
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        logger.info("Starting full benchmark suite")
        
        # System info
        self.results['system_info'] = self._get_system_info()
        self.results['benchmark_config'] = self.benchmark_config
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['scenarios'] = {}
        
        # Test each scenario
        for scenario in self.benchmark_config['test_scenarios']:
            scenario_name = scenario['name']
            logger.info(f"Running scenario: {scenario_name}")
            
            try:
                scenario_results = await self._run_scenario(scenario)
                self.results['scenarios'][scenario_name] = scenario_results
                logger.info(f"Completed scenario: {scenario_name}")
            except Exception as e:
                logger.error(f"Failed scenario {scenario_name}: {e}")
                self.results['scenarios'][scenario_name] = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Generate summary
        self.results['summary'] = self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Generate plots
        self._generate_plots()
        
        return self.results
    
    async def _run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark scenario"""
        scenario_name = scenario['name']
        
        # Create configuration
        config = PerformanceConfig(
            device=DeviceType[scenario.get('device', 'AUTO')],
            precision=PrecisionType[scenario.get('precision', 'AUTO')],
            optimization_level=OptimizationLevel[scenario.get('optimization_level', 'BASIC')],
            batch_size=scenario.get('batch_size', 4),
            async_processing=scenario.get('async_processing', True),
            enable_tensorrt=scenario.get('enable_tensorrt', False),
            enable_profiling=True
        )
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer(config)
        await optimizer.initialize()
        
        try:
            # Create and optimize test model
            model = SimpleTestModel()
            optimized_model = optimizer.optimize_model(model, f"test_model_{scenario_name}")
            
            # Generate test data
            test_frames = generate_test_frames(
                self.benchmark_config['test_parameters']['num_frames'],
                tuple(self.benchmark_config['test_parameters']['frame_resolution'])
            )
            
            # Warmup
            logger.info(f"Warming up scenario: {scenario_name}")
            await self._warmup_scenario(optimizer, test_frames[:10])
            
            # Measure performance
            logger.info(f"Measuring performance for scenario: {scenario_name}")
            measurements = []
            
            for iteration in range(self.benchmark_config['test_parameters']['measurement_iterations']):
                logger.info(f"Measurement iteration {iteration + 1}")
                
                if config.async_processing:
                    measurement = await self._measure_async_performance(optimizer, test_frames)
                else:
                    measurement = await self._measure_sync_performance(optimizer, test_frames, optimized_model)
                
                measurements.append(measurement)
                
                # Clear GPU cache between iterations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Analyze measurements
            results = self._analyze_measurements(measurements, optimizer, config)
            
            return results
            
        finally:
            await optimizer.shutdown()
    
    async def _warmup_scenario(self, optimizer: PerformanceOptimizer, warmup_frames: List[np.ndarray]):
        """Warmup the scenario"""
        if optimizer.config.async_processing:
            # Add frames to async queue
            for frame in warmup_frames:
                await optimizer.process_frame_async(frame, "test_model", simple_inference_function)
            
            # Process a few batches
            for _ in range(3):
                result = await optimizer.process_batch_async(simple_inference_function)
                if result:
                    logger.debug(f"Warmup batch processed in {result.processing_time:.4f}s")
        else:
            # Direct synchronous processing
            model = list(optimizer._models.values())[0]
            for frame in warmup_frames:
                # Simulate direct inference
                tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                tensor = tensor.to(optimizer.device)
                
                if optimizer.precision_manager.detect_optimal_precision() == "float16":
                    tensor = tensor.half()
                
                with torch.no_grad():
                    _ = model(tensor)
    
    async def _measure_async_performance(self, optimizer: PerformanceOptimizer, test_frames: List[np.ndarray]) -> Dict[str, Any]:
        """Measure async performance"""
        start_time = time.time()
        frame_ids = []
        processing_times = []
        throughput_data = []
        
        # Submit all frames
        submit_start = time.time()
        for i, frame in enumerate(test_frames):
            frame_id = await optimizer.process_frame_async(
                frame, 
                "test_model", 
                simple_inference_function,
                metadata={'frame_index': i}
            )
            if frame_id:
                frame_ids.append(frame_id)
        submit_time = time.time() - submit_start
        
        # Process batches
        process_start = time.time()
        results_collected = 0
        batch_count = 0
        
        while results_collected < len(test_frames):
            result = await optimizer.process_batch_async(simple_inference_function)
            if result and not result.error:
                batch_count += 1
                processing_times.append(result.processing_time)
                results_collected += len(result.results)
                
                # Calculate instantaneous throughput
                throughput = len(result.results) / result.processing_time
                throughput_data.append(throughput)
            elif result and result.error:
                logger.warning(f"Batch processing error: {result.error}")
                break
            
            # Timeout protection
            if time.time() - process_start > self.benchmark_config['test_parameters']['timeout_seconds']:
                logger.warning("Processing timeout reached")
                break
        
        total_time = time.time() - start_time
        process_time = time.time() - process_start
        
        return {
            'type': 'async',
            'total_time': total_time,
            'submit_time': submit_time,
            'process_time': process_time,
            'frames_processed': results_collected,
            'batch_count': batch_count,
            'avg_batch_time': np.mean(processing_times) if processing_times else 0,
            'total_throughput': results_collected / total_time,
            'processing_throughput': results_collected / process_time,
            'avg_throughput_per_batch': np.mean(throughput_data) if throughput_data else 0,
            'throughput_std': np.std(throughput_data) if throughput_data else 0,
            'batch_times': processing_times
        }
    
    async def _measure_sync_performance(self, optimizer: PerformanceOptimizer, test_frames: List[np.ndarray], model: nn.Module) -> Dict[str, Any]:
        """Measure synchronous performance"""
        start_time = time.time()
        inference_times = []
        
        for i, frame in enumerate(test_frames):
            frame_start = time.time()
            
            # Convert frame to tensor
            tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            tensor = tensor.to(optimizer.device)
            
            if optimizer.precision_manager.detect_optimal_precision() == "float16":
                tensor = tensor.half()
            
            # Run inference
            with torch.no_grad():
                if hasattr(torch.cuda, 'amp') and 'cuda' in optimizer.device:
                    with torch.cuda.amp.autocast(enabled=(optimizer.precision_manager.detect_optimal_precision() == "float16")):
                        output = model(tensor)
                else:
                    output = model(tensor)
            
            # Apply inference function
            result = simple_inference_function(output[0], frame, {'frame_index': i})
            
            inference_time = time.time() - frame_start
            inference_times.append(inference_time)
        
        total_time = time.time() - start_time
        
        return {
            'type': 'sync',
            'total_time': total_time,
            'frames_processed': len(test_frames),
            'avg_inference_time': np.mean(inference_times),
            'total_throughput': len(test_frames) / total_time,
            'inference_time_std': np.std(inference_times),
            'inference_times': inference_times
        }
    
    def _analyze_measurements(self, measurements: List[Dict[str, Any]], optimizer: PerformanceOptimizer, config: PerformanceConfig) -> Dict[str, Any]:
        """Analyze measurement results"""
        
        # Aggregate measurements
        total_times = [m['total_time'] for m in measurements]
        throughputs = [m['total_throughput'] for m in measurements]
        
        analysis = {
            'config': asdict(config),
            'device_info': optimizer.device_manager.get_device_info(),
            'precision': optimizer.precision_manager.detect_optimal_precision(),
            'measurements': measurements,
            'aggregated_stats': {
                'avg_total_time': np.mean(total_times),
                'std_total_time': np.std(total_times),
                'avg_throughput': np.mean(throughputs),
                'std_throughput': np.std(throughputs),
                'max_throughput': np.max(throughputs),
                'min_throughput': np.min(throughputs)
            },
            'performance_stats': optimizer.get_performance_stats()
        }
        
        # Add type-specific analysis
        if measurements[0]['type'] == 'async':
            batch_times = [bt for m in measurements for bt in m.get('batch_times', [])]
            if batch_times:
                analysis['batch_stats'] = {
                    'avg_batch_time': np.mean(batch_times),
                    'std_batch_time': np.std(batch_times),
                    'min_batch_time': np.min(batch_times),
                    'max_batch_time': np.max(batch_times)
                }
        else:
            inference_times = [it for m in measurements for it in m.get('inference_times', [])]
            if inference_times:
                analysis['inference_stats'] = {
                    'avg_inference_time': np.mean(inference_times),
                    'std_inference_time': np.std(inference_times),
                    'min_inference_time': np.min(inference_times),
                    'max_inference_time': np.max(inference_times)
                }
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary"""
        summary = {
            'scenarios_completed': len([s for s in self.results['scenarios'].values() if 'error' not in s]),
            'scenarios_failed': len([s for s in self.results['scenarios'].values() if 'error' in s]),
            'performance_comparison': {}
        }
        
        # Compare scenarios
        baseline_throughput = None
        for scenario_name, scenario_results in self.results['scenarios'].items():
            if 'error' in scenario_results:
                continue
            
            throughput = scenario_results['aggregated_stats']['avg_throughput']
            
            if 'baseline' in scenario_name:
                baseline_throughput = throughput
            
            summary['performance_comparison'][scenario_name] = {
                'throughput_fps': throughput,
                'avg_time_per_frame': 1.0 / throughput if throughput > 0 else float('inf')
            }
            
            if baseline_throughput and baseline_throughput > 0:
                summary['performance_comparison'][scenario_name]['speedup_vs_baseline'] = throughput / baseline_throughput
        
        # Find best performing scenario
        best_scenario = max(
            [(name, results) for name, results in self.results['scenarios'].items() if 'error' not in results],
            key=lambda x: x[1]['aggregated_stats']['avg_throughput'],
            default=(None, None)
        )
        
        if best_scenario[0]:
            summary['best_scenario'] = {
                'name': best_scenario[0],
                'throughput': best_scenario[1]['aggregated_stats']['avg_throughput']
            }
        
        return summary
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'python_version': f"{os.sys.version}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_info': {
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory_info': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            }
        }
        
        if torch.cuda.is_available():
            info['cuda_info'] = {
                'device_count': torch.cuda.device_count(),
                'devices': []
            }
            
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'name': torch.cuda.get_device_name(i),
                    'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'compute_capability': torch.cuda.get_device_properties(i).major
                }
                info['cuda_info']['devices'].append(device_info)
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_utilization'] = [{
                    'name': gpu.name,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'utilization_percent': gpu.load * 100
                } for gpu in gpus]
        except Exception:
            pass
        
        return info
    
    def _save_results(self):
        """Save benchmark results to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {self.output_file}")
    
    def _generate_plots(self):
        """Generate performance visualization plots"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Throughput comparison plot
            scenario_names = []
            throughputs = []
            
            for scenario_name, scenario_results in self.results['scenarios'].items():
                if 'error' not in scenario_results:
                    scenario_names.append(scenario_name)
                    throughputs.append(scenario_results['aggregated_stats']['avg_throughput'])
            
            if len(scenario_names) > 1:
                plt.figure(figsize=(12, 8))
                
                # Throughput comparison
                plt.subplot(2, 2, 1)
                bars = plt.bar(scenario_names, throughputs)
                plt.title('Throughput Comparison (FPS)')
                plt.xlabel('Scenario')
                plt.ylabel('Frames Per Second')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, throughputs):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value:.2f}', ha='center', va='bottom')
                
                # Speedup comparison (if baseline exists)
                speedups = []
                baseline_throughput = None
                for name in scenario_names:
                    if 'baseline' in name and baseline_throughput is None:
                        baseline_throughput = self.results['scenarios'][name]['aggregated_stats']['avg_throughput']
                        break
                
                if baseline_throughput and baseline_throughput > 0:
                    speedups = [throughput / baseline_throughput for throughput in throughputs]
                    
                    plt.subplot(2, 2, 2)
                    bars = plt.bar(scenario_names, speedups)
                    plt.title('Speedup vs Baseline')
                    plt.xlabel('Scenario')
                    plt.ylabel('Speedup Factor')
                    plt.xticks(rotation=45)
                    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, speedups):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                f'{value:.2f}x', ha='center', va='bottom')
                
                plt.tight_layout()
                plot_file = self.output_file.replace('.json', '_plots.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Plots saved to {plot_file}")
            
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Performance Optimization Benchmarking")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, help='Output results file path')
    parser.add_argument('--scenarios', nargs='+', help='Specific scenarios to run')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with fewer iterations')
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = BenchmarkSuite(args.config, args.output)
    
    # Modify config for quick run
    if args.quick:
        benchmark.benchmark_config['test_parameters']['num_frames'] = 20
        benchmark.benchmark_config['test_parameters']['measurement_iterations'] = 2
        benchmark.benchmark_config['test_parameters']['warmup_iterations'] = 2
        logger.info("Running quick benchmark mode")
    
    # Filter scenarios if specified
    if args.scenarios:
        filtered_scenarios = [
            s for s in benchmark.benchmark_config['test_scenarios']
            if s['name'] in args.scenarios
        ]
        benchmark.benchmark_config['test_scenarios'] = filtered_scenarios
        logger.info(f"Running filtered scenarios: {args.scenarios}")
    
    # Run benchmark
    async def run_benchmark():
        try:
            results = await benchmark.run_full_benchmark()
            
            # Print summary
            print("\n" + "="*60)
            print("BENCHMARK RESULTS SUMMARY")
            print("="*60)
            
            summary = results.get('summary', {})
            print(f"Scenarios completed: {summary.get('scenarios_completed', 0)}")
            print(f"Scenarios failed: {summary.get('scenarios_failed', 0)}")
            
            if 'performance_comparison' in summary:
                print("\nThroughput Comparison:")
                for scenario, stats in summary['performance_comparison'].items():
                    speedup = stats.get('speedup_vs_baseline', 1.0)
                    print(f"  {scenario:25s}: {stats['throughput_fps']:8.2f} FPS ({speedup:.2f}x)")
            
            if 'best_scenario' in summary:
                best = summary['best_scenario']
                print(f"\nBest performing scenario: {best['name']} ({best['throughput']:.2f} FPS)")
            
            print(f"\nDetailed results saved to: {benchmark.output_file}")
            
        except KeyboardInterrupt:
            logger.info("Benchmark interrupted by user")
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            traceback.print_exc()
    
    # Run the async benchmark
    asyncio.run(run_benchmark())

if __name__ == "__main__":
    main()
