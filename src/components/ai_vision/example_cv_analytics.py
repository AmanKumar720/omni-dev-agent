#!/usr/bin/env python3
"""
Example usage of Computer Vision Analytics Module

This script demonstrates how to use the computer vision analytics module
for motion detection, scene segmentation, and visual reasoning.
"""

import sys
import asyncio
import cv2
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.components.ai_vision.computer_vision_analytics import (
    ComputerVisionAnalyticsAgent,
    ComputerVisionAnalyticsTask,
    AnalyticsEvent,
    AnalyticsEventType,
    DEFAULT_ANALYTICS_CONFIG,
    DoorOpenRule,
    ActivityDetectionRule
)

async def demo_single_frame_processing():
    """Demonstrate processing a single frame"""
    print("=== Single Frame Processing Demo ===")
    
    # Create a synthetic test frame (blue background with white rectangle)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [100, 50, 50]  # Blue background
    cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)  # White rectangle
    
    # Create analytics agent
    agent = ComputerVisionAnalyticsAgent(
        agent_id="demo_agent",
        analytics_config=DEFAULT_ANALYTICS_CONFIG
    )
    
    # Create analytics task
    task_id = agent.create_analytics_task()
    
    # Execute task
    result = await agent.execute_task(task_id, frame)
    
    if result and result.data:
        print(f"Task completed with confidence: {result.confidence:.2f}")
        print(f"Motion detected: {result.data.get('motion_result', {}).get('motion_detected', False)}")
        print(f"Scene segments: {len(result.data.get('segmentation_result', {}).get('segments', []))}")
        print(f"Rules triggered: {len(result.data.get('reasoning_result', {}).get('triggered_rules', []))}")
        print(f"Events generated: {len(result.data.get('events', []))}")
        
        # Print event details
        for i, event in enumerate(result.data.get('events', [])):
            print(f"  Event {i+1}: {event.get('event_type')} (confidence: {event.get('confidence', 0):.2f})")
    
    print()

async def demo_video_sequence_processing():
    """Demonstrate processing a sequence of frames (simulated video)"""
    print("=== Video Sequence Processing Demo ===")
    
    # Create a sequence of frames with moving object
    frames = []
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [50, 100, 50]  # Green background
        
        # Moving white rectangle
        x = 50 + i * 30
        cv2.rectangle(frame, (x, 200), (x + 80, 280), (255, 255, 255), -1)
        frames.append(frame)
    
    # Create analytics agent with custom config
    custom_config = DEFAULT_ANALYTICS_CONFIG.copy()
    custom_config['motion_config']['threshold'] = 15  # More sensitive to motion
    
    agent = ComputerVisionAnalyticsAgent(
        agent_id="video_demo_agent",
        analytics_config=custom_config
    )
    
    # Create and execute task
    task_id = agent.create_analytics_task()
    result = await agent.execute_task(task_id, frames)
    
    if result and result.data:
        print(f"Processed {result.data.get('frames_processed', 0)} frames")
        print(f"Overall confidence: {result.confidence:.2f}")
        print(f"Total events generated: {len(result.data.get('events_generated', []))}")
        
        # Print analytics summary
        summary = result.data.get('analytics_summary', {})
        if summary:
            print(f"Event type distribution: {summary.get('event_types', {})}")
            print(f"Average confidence: {summary.get('average_confidence', 0):.2f}")
            print(f"Most common event: {summary.get('most_common_event', 'None')}")
    
    print()

async def demo_custom_rules():
    """Demonstrate adding custom visual reasoning rules"""
    print("=== Custom Rules Demo ===")
    
    # Create a custom rule class
    class RedObjectRule:
        def __init__(self):
            self.rule_id = "red_object_detection"
            self.name = "Red Object Detection"
            self.description = "Detects red objects in the scene"
        
        def evaluate(self, frame, motion_result, segmentation_result):
            try:
                # Convert to HSV for better red detection
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Define red color range
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])
                
                # Create masks
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)
                
                # Calculate red area
                red_area = np.sum(red_mask > 0)
                frame_area = frame.shape[0] * frame.shape[1]
                red_ratio = red_area / frame_area if frame_area > 0 else 0
                
                triggered = red_ratio > 0.01  # Trigger if >1% of frame is red
                confidence = min(red_ratio * 20, 1.0)  # Scale for confidence
                
                return {
                    'triggered': triggered,
                    'red_area': red_area,
                    'red_ratio': red_ratio,
                    'confidence': confidence
                }
                
            except Exception as e:
                return {
                    'triggered': False,
                    'error': str(e),
                    'confidence': 0.0
                }
    
    # Create test frame with red object
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [50, 100, 150]  # Background
    cv2.rectangle(frame, (250, 200), (390, 280), (0, 0, 255), -1)  # Red rectangle
    
    # Create analytics task with custom rule
    task_config = DEFAULT_ANALYTICS_CONFIG.copy()
    task = ComputerVisionAnalyticsTask("custom_rule_task", task_config)
    
    # Add custom rule to the visual reasoner
    custom_rule = RedObjectRule()
    task.visual_reasoner.rules.append(custom_rule)
    
    # Execute task
    result = await task.execute(frame)
    
    if result and result.data:
        print(f"Custom rule processing completed")
        print(f"Confidence: {result.confidence:.2f}")
        
        reasoning_result = result.data.get('reasoning_result', {})
        scene_state = reasoning_result.get('scene_state', {})
        
        if 'red_object_detection' in scene_state:
            red_result = scene_state['red_object_detection']
            print(f"Red object detected: {red_result.get('triggered', False)}")
            print(f"Red area ratio: {red_result.get('red_ratio', 0):.4f}")
            print(f"Red detection confidence: {red_result.get('confidence', 0):.2f}")
    
    print()

def demo_event_handler():
    """Demonstrate custom event handling"""
    print("=== Custom Event Handler Demo ===")
    
    # Define custom event handler
    def custom_event_handler(event: AnalyticsEvent):
        print(f"Custom handler received event: {event.event_type.value}")
        print(f"  Event ID: {event.event_id}")
        print(f"  Timestamp: {event.timestamp}")
        print(f"  Confidence: {event.confidence:.2f}")
        
        if event.metadata:
            print(f"  Metadata: {event.metadata}")
        print()
    
    # Create agent and add event handler
    agent = ComputerVisionAnalyticsAgent(
        agent_id="event_demo_agent",
        analytics_config=DEFAULT_ANALYTICS_CONFIG
    )
    
    agent.event_emitter.add_event_handler(custom_event_handler)
    
    print("Event handler added. Future events will be processed by custom handler.")
    print()

async def demo_learning_engine_integration():
    """Demonstrate Learning Engine integration"""
    print("=== Learning Engine Integration Demo ===")
    
    # Import learning engine
    from src.learning.learning_engine import global_learning_engine
    
    # Create test frames with different patterns
    frames = []
    
    # Frame 1: Static scene
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[:, :] = [100, 100, 100]  # Gray background
    frames.append(frame1)
    
    # Frame 2: Scene with motion
    frame2 = frame1.copy()
    cv2.rectangle(frame2, (300, 200), (340, 240), (255, 255, 255), -1)  # White square
    frames.append(frame2)
    
    # Create agent and process frames
    agent = ComputerVisionAnalyticsAgent(
        agent_id="learning_demo_agent",
        analytics_config=DEFAULT_ANALYTICS_CONFIG
    )
    
    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}")
        task_id = agent.create_analytics_task()
        await agent.execute_task(task_id, frame)
    
    # Give some time for events to be processed
    await asyncio.sleep(1)
    
    # Check learning insights
    insights = global_learning_engine.get_learning_insights()
    print(f"Learning Engine Insights:")
    print(f"  Total experiences: {insights.get('total_experiences', 0)}")
    print(f"  Overall success rate: {insights.get('overall_success_rate', 0):.2f}")
    print(f"  Recent success rate: {insights.get('recent_success_rate', 0):.2f}")
    print(f"  Context patterns learned: {insights.get('context_patterns_learned', 0)}")
    
    # Get recommendations
    recommendations = insights.get('recommendations', [])
    if recommendations:
        print(f"  Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")
    
    print()

async def main():
    """Main demo function"""
    print("Computer Vision Analytics Module - Demo")
    print("=" * 50)
    
    # Run demos
    await demo_single_frame_processing()
    await demo_video_sequence_processing()
    await demo_custom_rules()
    demo_event_handler()
    await demo_learning_engine_integration()
    
    print("Demo completed!")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
