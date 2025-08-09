"""
Example: LLM Overload Handler Integration with Omni-Dev Agent
Demonstrates how to integrate overload handling with existing AI vision components.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import the LLM overload handler
try:
    from ..error_handling.llm_overload_handler import (
        LLMOverloadHandler, OverloadConfig, with_llm_overload_protection
    )
except ImportError:
    from error_handling.llm_overload_handler import (
        LLMOverloadHandler, OverloadConfig, with_llm_overload_protection
    )

# Mock LLM API functions for demonstration
class MockLLMAPI:
    """Mock LLM API client for demonstration purposes"""
    
    def __init__(self, simulate_limits: bool = True):
        self.simulate_limits = simulate_limits
        self.request_count = 0
        self.total_tokens_used = 0
        
    async def chat_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Mock chat completion API call"""
        self.request_count += 1
        
        # Simulate token usage
        prompt_tokens = len(prompt.split()) * 1.3
        completion_tokens = prompt_tokens * 0.3
        total_tokens = int(prompt_tokens + completion_tokens)
        
        self.total_tokens_used += total_tokens
        
        # Simulate API delays and potential errors
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate rate limiting
        if self.simulate_limits and self.request_count % 20 == 0:
            raise Exception("Rate limit exceeded")
            
        # Simulate token limit errors
        if self.simulate_limits and total_tokens > 4000:
            raise Exception("Token limit exceeded")
        
        return {
            "choices": [{
                "message": {
                    "content": f"Mock response for: {prompt[:50]}..."
                }
            }],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": total_tokens
            }
        }


class VisionAnalysisService:
    """Enhanced vision analysis service with LLM overload protection"""
    
    def __init__(self):
        # Configure overload handler for vision analysis workloads
        overload_config = OverloadConfig(
            context_window_size=8192,
            max_context_utilization=0.85,  # Leave some buffer
            token_warning_threshold=0.7,
            token_critical_threshold=0.9,
            retry_attempts=3,
            enable_token_chunking=True,
            chunk_overlap=200  # More overlap for vision context
        )
        
        self.overload_handler = LLMOverloadHandler(overload_config)
        self.llm_api = MockLLMAPI()
        self.logger = logging.getLogger(__name__ + ".VisionAnalysisService")
        
    @with_llm_overload_protection()
    async def analyze_scene_description(self, image_objects: List[str], 
                                       scene_context: str) -> str:
        """
        Analyze scene with automatic overload protection
        """
        prompt = self._build_scene_analysis_prompt(image_objects, scene_context)
        
        try:
            response = await self.llm_api.chat_completion(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Scene analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    async def analyze_multiple_scenes(self, scenes: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze multiple scenes with manual overload handling
        """
        results = []
        
        for i, scene in enumerate(scenes):
            self.logger.info(f"Analyzing scene {i+1}/{len(scenes)}")
            
            # Build prompt
            prompt = self._build_scene_analysis_prompt(
                scene.get('objects', []),
                scene.get('context', '')
            )
            
            # Check for overload conditions before making request
            estimated_tokens = len(prompt.split()) * 1.3
            overload_events = await self.overload_handler.check_overload_conditions(
                int(estimated_tokens),
                {'scene_index': i, 'total_scenes': len(scenes)}
            )
            
            if overload_events:
                self.logger.warning(f"Overload detected for scene {i}: {[e.overload_type.value for e in overload_events]}")
                
            # Apply mitigation and execute request
            try:
                result = await self.overload_handler.apply_mitigation_strategy(
                    overload_events,
                    self.llm_api.chat_completion,
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.3
                )
                
                if isinstance(result, list):
                    # Handle chunked results
                    combined_result = " ".join([r["choices"][0]["message"]["content"] for r in result])
                    results.append(combined_result)
                else:
                    results.append(result["choices"][0]["message"]["content"])
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze scene {i}: {e}")
                results.append(f"Analysis failed: {str(e)}")
                
            # Small delay between scenes
            await asyncio.sleep(0.5)
            
        return results
    
    def _build_scene_analysis_prompt(self, objects: List[str], context: str) -> str:
        """Build a comprehensive scene analysis prompt"""
        objects_text = ", ".join(objects) if objects else "no specific objects detected"
        
        prompt = f"""
        Analyze this visual scene based on the detected elements:
        
        Detected Objects: {objects_text}
        Scene Context: {context}
        
        Please provide:
        1. A brief description of what's happening in the scene
        2. The likely setting or location
        3. Any notable interactions or activities
        4. Safety considerations if applicable
        5. Suggested actions or recommendations
        
        Keep the analysis concise but comprehensive.
        """
        
        return prompt.strip()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status including overload metrics"""
        usage_stats = self.overload_handler.get_usage_stats()
        
        return {
            "service_status": "active",
            "llm_usage": usage_stats,
            "api_stats": {
                "total_requests": self.llm_api.request_count,
                "total_tokens": self.llm_api.total_tokens_used
            },
            "overload_protection": {
                "enabled": True,
                "config": {
                    "context_window": self.overload_handler.config.context_window_size,
                    "token_chunking": self.overload_handler.config.enable_token_chunking,
                    "retry_attempts": self.overload_handler.config.retry_attempts
                }
            }
        }


class AdvancedLLMManager:
    """Advanced LLM manager with comprehensive overload handling"""
    
    def __init__(self):
        self.handlers = {}
        self.global_handler = LLMOverloadHandler()
        self.logger = logging.getLogger(__name__ + ".AdvancedLLMManager")
        
    def get_handler(self, service_name: str, custom_config: Optional[OverloadConfig] = None) -> LLMOverloadHandler:
        """Get or create a handler for a specific service"""
        if service_name not in self.handlers:
            config = custom_config or OverloadConfig()
            self.handlers[service_name] = LLMOverloadHandler(config)
            self.logger.info(f"Created overload handler for service: {service_name}")
            
        return self.handlers[service_name]
    
    async def batch_process_with_protection(self, 
                                           requests: List[Dict[str, Any]],
                                           service_name: str = "default") -> List[Any]:
        """Process a batch of requests with overload protection"""
        handler = self.get_handler(service_name)
        results = []
        
        self.logger.info(f"Processing batch of {len(requests)} requests for {service_name}")
        
        for i, request in enumerate(requests):
            prompt = request.get('prompt', '')
            estimated_tokens = len(prompt.split()) * 1.3
            
            # Check overload for this request
            events = await handler.check_overload_conditions(
                int(estimated_tokens),
                {'batch_index': i, 'service': service_name}
            )
            
            if events:
                self.logger.info(f"Overload detected for request {i}: applying mitigation")
                
            # Execute with protection
            try:
                api_client = MockLLMAPI()
                result = await handler.apply_mitigation_strategy(
                    events,
                    api_client.chat_completion,
                    **request
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Request {i} failed: {e}")
                results.append({"error": str(e)})
                
        return results
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overview of all handlers and their status"""
        overview = {
            "global_handler": self.global_handler.get_usage_stats(),
            "service_handlers": {}
        }
        
        for service_name, handler in self.handlers.items():
            overview["service_handlers"][service_name] = handler.get_usage_stats()
            
        return overview


async def demonstrate_overload_handling():
    """Demonstrate various overload handling scenarios"""
    print("🚀 LLM Overload Handler Demonstration")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize services
    vision_service = VisionAnalysisService()
    llm_manager = AdvancedLLMManager()
    
    print("\n1. Testing basic overload protection...")
    
    # Test scene analysis with overload protection
    scene_data = {
        'objects': ['person', 'car', 'traffic_light', 'crosswalk'],
        'context': 'Urban intersection during daytime with moderate traffic'
    }
    
    result = await vision_service.analyze_scene_description(
        scene_data['objects'], 
        scene_data['context']
    )
    print(f"Scene Analysis Result: {result}")
    
    print("\n2. Testing multiple scene analysis...")
    
    # Test multiple scenes
    scenes = [
        {'objects': ['dog', 'person', 'park'], 'context': 'Dog park on sunny day'},
        {'objects': ['car', 'road', 'trees'], 'context': 'Rural highway'},
        {'objects': ['people', 'building', 'urban'], 'context': 'City street'},
    ]
    
    scene_results = await vision_service.analyze_multiple_scenes(scenes)
    for i, result in enumerate(scene_results):
        print(f"Scene {i+1}: {result[:100]}...")
    
    print("\n3. Testing batch processing with different services...")
    
    # Test batch processing
    requests = [
        {'prompt': 'Analyze this simple scene with basic objects', 'max_tokens': 100},
        {'prompt': 'Provide detailed analysis of complex urban environment with multiple vehicles, pedestrians, and infrastructure elements', 'max_tokens': 300},
        {'prompt': 'Quick object detection summary', 'max_tokens': 50},
    ]
    
    batch_results = await llm_manager.batch_process_with_protection(requests, "vision_analysis")
    print(f"Processed {len(batch_results)} batch requests")
    
    print("\n4. System Status Report:")
    print("-" * 30)
    
    # Get status reports
    vision_status = await vision_service.get_system_status()
    manager_overview = llm_manager.get_system_overview()
    
    print(f"Vision Service Status: {vision_status['service_status']}")
    print(f"Token Utilization: {vision_status['llm_usage']['token_utilization_minute']:.2%}")
    print(f"Request Utilization: {vision_status['llm_usage']['request_utilization_minute']:.2%}")
    print(f"Active Requests: {vision_status['llm_usage']['active_requests']}")
    print(f"Recent Overload Events: {vision_status['llm_usage']['recent_overload_events']}")
    
    print(f"\nTotal Service Handlers: {len(manager_overview['service_handlers'])}")
    for service, stats in manager_overview['service_handlers'].items():
        print(f"  {service}: {stats['minute_requests']} requests/min")
    
    print("\n✅ Demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("- Automatic overload detection")
    print("- Token limit handling with chunking")
    print("- Rate limit management with backoff")
    print("- Context overflow protection")
    print("- Multi-service overload handling")
    print("- Real-time usage monitoring")


if __name__ == "__main__":
    asyncio.run(demonstrate_overload_handling())
