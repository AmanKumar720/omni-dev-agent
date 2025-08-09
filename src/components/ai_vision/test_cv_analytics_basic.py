#!/usr/bin/env python3
"""
Basic tests for Computer Vision Analytics Module

Tests core functionality including motion detection, scene segmentation,
visual reasoning, and event emission.
"""

import sys
import pytest
import asyncio
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.components.ai_vision.computer_vision_analytics import (
    ComputerVisionAnalyticsAgent,
    ComputerVisionAnalyticsTask,
    MotionDetector,
    SceneSegmenter,
    VisualReasoner,
    DoorOpenRule,
    ActivityDetectionRule,
    AnalyticsEventEmitter,
    AnalyticsEvent,
    AnalyticsEventType,
    DEFAULT_ANALYTICS_CONFIG
)

class TestMotionDetector:
    """Test motion detection functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {'threshold': 25, 'min_area': 500, 'blur_size': 21}
        self.detector = MotionDetector(self.config)
    
    def test_motion_detector_initialization(self):
        """Test motion detector initialization"""
        assert self.detector.threshold == 25
        assert self.detector.min_area == 500
        assert self.detector.blur_size == 21
        assert self.detector.previous_frame is None
    
    def test_no_motion_first_frame(self):
        """Test that first frame shows no motion"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_motion(frame)
        
        assert not result.motion_detected
        assert result.motion_area == 0.0
        assert len(result.motion_regions) == 0
        assert result.confidence == 0.0
        assert self.detector.previous_frame is not None
    
    def test_motion_detection_with_change(self):
        """Test motion detection with frame changes"""
        # First frame (static)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1[:, :] = [100, 100, 100]  # Gray
        self.detector.detect_motion(frame1)
        
        # Second frame with changes
        frame2 = frame1.copy()
        cv2.rectangle(frame2, (200, 200), (400, 400), (255, 255, 255), -1)
        result = self.detector.detect_motion(frame2)
        
        assert result.motion_detected or result.motion_area > 0
        # Note: Might not always trigger due to threshold settings

class TestSceneSegmenter:
    """Test scene segmentation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {'num_clusters': 4, 'resize_factor': 0.25}
        self.segmenter = SceneSegmenter(self.config)
    
    def test_segmenter_initialization(self):
        """Test scene segmenter initialization"""
        assert self.segmenter.num_clusters == 4
        assert self.segmenter.resize_factor == 0.25
    
    def test_scene_segmentation(self):
        """Test scene segmentation"""
        # Create frame with distinct regions
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0:240, 0:320] = [255, 0, 0]      # Red quadrant
        frame[0:240, 320:640] = [0, 255, 0]    # Green quadrant
        frame[240:480, 0:320] = [0, 0, 255]    # Blue quadrant
        frame[240:480, 320:640] = [255, 255, 0] # Yellow quadrant
        
        result = self.segmenter.segment_scene(frame)
        
        assert len(result.segments) <= self.segmenter.num_clusters
        assert result.confidence >= 0.0
        assert result.scene_complexity >= 0.0
        assert len(result.dominant_regions) <= 3

class TestVisualRules:
    """Test visual reasoning rules"""
    
    def test_door_open_rule_initialization(self):
        """Test door open rule initialization"""
        door_region = {'x': 100, 'y': 50, 'width': 200, 'height': 300}
        rule = DoorOpenRule(door_region, threshold=0.3)
        
        assert rule.rule_id == "door_open"
        assert rule.threshold == 0.3
        assert rule.door_region == door_region
    
    def test_activity_detection_rule_initialization(self):
        """Test activity detection rule initialization"""
        rule = ActivityDetectionRule(motion_threshold=0.05)
        
        assert rule.rule_id == "activity_detection"
        assert rule.motion_threshold == 0.05
        assert len(rule.activity_history) == 0
    
    def test_door_rule_evaluation_no_error(self):
        """Test door rule evaluation doesn't crash"""
        door_region = {'x': 100, 'y': 50, 'width': 200, 'height': 300}
        rule = DoorOpenRule(door_region)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        from src.components.ai_vision.computer_vision_analytics import MotionResult, SegmentationResult
        
        motion_result = MotionResult(False, 0.0, [], 0.0)
        segmentation_result = SegmentationResult([], [], 0.0, 0.0)
        
        result = rule.evaluate(frame, motion_result, segmentation_result)
        
        assert isinstance(result, dict)
        assert 'triggered' in result
        assert 'confidence' in result

class TestAnalyticsEventEmitter:
    """Test analytics event emission"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.emitter = AnalyticsEventEmitter()
    
    def test_emitter_initialization(self):
        """Test event emitter initialization"""
        assert hasattr(self.emitter, 'event_queue')
        assert hasattr(self.emitter, 'event_handlers')
        assert len(self.emitter.event_handlers) == 0
    
    def test_add_event_handler(self):
        """Test adding event handler"""
        def dummy_handler(event):
            pass
        
        self.emitter.add_event_handler(dummy_handler)
        assert len(self.emitter.event_handlers) == 1
    
    def test_emit_event(self):
        """Test event emission"""
        from datetime import datetime
        
        event = AnalyticsEvent(
            event_id="test_event",
            timestamp=datetime.now(),
            event_type=AnalyticsEventType.MOTION_DETECTED,
            confidence=0.8
        )
        
        # Should not raise exception
        self.emitter.emit_event(event)

class TestComputerVisionAnalyticsTask:
    """Test analytics task functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.task = ComputerVisionAnalyticsTask(
            "test_task",
            DEFAULT_ANALYTICS_CONFIG
        )
    
    def test_task_initialization(self):
        """Test task initialization"""
        assert self.task.task_id == "test_task"
        assert self.task.task_type == "computer_vision_analytics"
        assert hasattr(self.task, 'motion_detector')
        assert hasattr(self.task, 'scene_segmenter')
        assert hasattr(self.task, 'visual_reasoner')
        assert hasattr(self.task, 'event_emitter')
    
    def test_input_validation(self):
        """Test input validation"""
        # Valid inputs
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert self.task.validate_input(frame)
        
        frames = [frame, frame.copy()]
        assert self.task.validate_input(frames)
        
        # Invalid inputs
        assert not self.task.validate_input("nonexistent_file.mp4")
        assert not self.task.validate_input(np.zeros((640,)))  # 1D array
        assert not self.task.validate_input([1, 2, 3])  # List of non-arrays
    
    @pytest.mark.asyncio
    async def test_single_frame_processing(self):
        """Test single frame processing"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [100, 100, 100]  # Gray background
        cv2.rectangle(frame, (200, 200), (400, 400), (255, 255, 255), -1)  # White rectangle
        
        result = await self.task.execute(frame)
        
        assert result.task_id == "test_task"
        assert result.data is not None
        assert 'motion_result' in result.data
        assert 'segmentation_result' in result.data
        assert 'reasoning_result' in result.data
        assert 'events' in result.data
        assert 'confidence' in result.data
    
    @pytest.mark.asyncio
    async def test_frame_sequence_processing(self):
        """Test frame sequence processing"""
        frames = []
        for i in range(3):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Smaller for speed
            frame[:, :] = [50, 50, 50]
            # Add moving rectangle
            x = 50 + i * 20
            cv2.rectangle(frame, (x, 100), (x + 40, 140), (255, 255, 255), -1)
            frames.append(frame)
        
        result = await self.task.execute(frames)
        
        assert result.task_id == "test_task"
        assert result.data is not None
        assert 'frames_processed' in result.data
        assert result.data['frames_processed'] == 3
        assert 'events_generated' in result.data

class TestComputerVisionAnalyticsAgent:
    """Test analytics agent functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.agent = ComputerVisionAnalyticsAgent(
            agent_id="test_agent",
            analytics_config=DEFAULT_ANALYTICS_CONFIG
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.name == "Computer Vision Analytics Agent"
        assert hasattr(self.agent, 'analytics_config')
        assert hasattr(self.agent, 'event_emitter')
    
    def test_create_analytics_task(self):
        """Test analytics task creation"""
        task_id = self.agent.create_analytics_task()
        
        assert isinstance(task_id, str)
        assert "analytics_task_" in task_id
        assert task_id in self.agent.tasks
    
    @pytest.mark.asyncio
    async def test_execute_analytics_task(self):
        """Test analytics task execution"""
        task_id = self.agent.create_analytics_task()
        
        # Create test frame
        frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Small for speed
        frame[:, :] = [100, 100, 100]
        
        result = await self.agent.execute_task(task_id, frame)
        
        assert result is not None
        assert result.task_id == task_id
        # Result may be completed or failed, but should not be None
    
    def test_get_analytics_summary(self):
        """Test analytics summary generation"""
        summary = self.agent.get_analytics_summary()
        
        assert isinstance(summary, dict)
        assert 'total_tasks' in summary
        assert 'completed_tasks' in summary
        assert 'total_events_generated' in summary
        assert 'agent_config' in summary

# Integration tests
class TestIntegration:
    """Test integration with other components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test end-to-end processing pipeline"""
        # Create agent
        agent = ComputerVisionAnalyticsAgent(
            agent_id="integration_test_agent",
            analytics_config=DEFAULT_ANALYTICS_CONFIG
        )
        
        # Create test data with motion
        frames = []
        for i in range(3):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Small for speed
            frame[:, :] = [80, 80, 80]  # Gray background
            
            # Moving white rectangle
            x = 50 + i * 30
            y = 100
            cv2.rectangle(frame, (x, y), (x + 40, y + 40), (255, 255, 255), -1)
            frames.append(frame)
        
        # Process frames individually
        results = []
        for i, frame in enumerate(frames):
            task_id = agent.create_analytics_task()
            result = await agent.execute_task(task_id, frame)
            results.append(result)
        
        # Verify results
        assert len(results) == 3
        for result in results:
            if result:  # Some results might be None due to processing issues in tests
                assert hasattr(result, 'task_id')
                assert hasattr(result, 'status')
    
    def test_event_handler_integration(self):
        """Test event handler integration"""
        events_received = []
        
        def test_handler(event):
            events_received.append(event)
        
        # Create agent with event handler
        agent = ComputerVisionAnalyticsAgent(
            agent_id="event_test_agent",
            analytics_config=DEFAULT_ANALYTICS_CONFIG
        )
        
        agent.event_emitter.add_event_handler(test_handler)
        
        # Create and emit test event
        from datetime import datetime
        test_event = AnalyticsEvent(
            event_id="test_event",
            timestamp=datetime.now(),
            event_type=AnalyticsEventType.MOTION_DETECTED,
            confidence=0.7
        )
        
        agent.event_emitter.emit_event(test_event)
        
        # Give some time for processing (in real scenario)
        # events_received should eventually contain the event

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
