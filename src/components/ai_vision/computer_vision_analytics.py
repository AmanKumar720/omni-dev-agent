"""
Computer Vision Analytics Module

Implements motion detection via frame differencing, scene segmentation, 
and basic visual reasoning with analytics event emission for Learning Engine integration.
"""

import cv2
import numpy as np
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import logging
from pathlib import Path
import threading
import queue
import time

# Import from existing modules
from .core import VisionTask, VisionResult, TaskStatus, AIVisionAgent
from ...learning.learning_engine import global_learning_engine

class AnalyticsEventType(Enum):
    """Types of analytics events"""
    MOTION_DETECTED = "motion_detected"
    SCENE_CHANGE = "scene_change"
    OBJECT_ENTERED = "object_entered"
    OBJECT_EXITED = "object_exited"
    DOOR_STATE_CHANGE = "door_state_change"
    ACTIVITY_DETECTED = "activity_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    RULE_TRIGGERED = "rule_triggered"

@dataclass
class AnalyticsEvent:
    """Analytics event structure for Learning Engine consumption"""
    event_id: str
    timestamp: datetime
    event_type: AnalyticsEventType
    confidence: float
    location: Optional[Dict[str, Any]] = None  # Bounding box, region, etc.
    metadata: Optional[Dict[str, Any]] = None
    frame_info: Optional[Dict[str, Any]] = None

@dataclass
class MotionResult:
    """Result of motion detection"""
    motion_detected: bool
    motion_area: float
    motion_regions: List[Dict[str, Any]]
    confidence: float
    frame_diff: Optional[np.ndarray] = None

@dataclass
class SegmentationResult:
    """Result of scene segmentation"""
    segments: List[Dict[str, Any]]
    dominant_regions: List[Dict[str, Any]]
    scene_complexity: float
    confidence: float

@dataclass
class ReasoningResult:
    """Result of visual reasoning"""
    rules_evaluated: List[str]
    triggered_rules: List[Dict[str, Any]]
    scene_state: Dict[str, Any]
    confidence: float

class VisualRule:
    """Base class for visual reasoning rules"""
    
    def __init__(self, rule_id: str, name: str, description: str):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.last_triggered = None
    
    @abstractmethod
    def evaluate(self, frame: np.ndarray, motion_result: MotionResult, 
                 segmentation_result: SegmentationResult) -> Dict[str, Any]:
        """Evaluate the rule against current frame data"""
        pass

class DoorOpenRule(VisualRule):
    """Rule to detect if a door is open"""
    
    def __init__(self, door_region: Dict[str, int], threshold: float = 0.3):
        super().__init__("door_open", "Door Open Detection", "Detects if a door is open based on visual cues")
        self.door_region = door_region  # {'x': x, 'y': y, 'width': w, 'height': h}
        self.threshold = threshold
        self.last_door_state = None
    
    def evaluate(self, frame: np.ndarray, motion_result: MotionResult, 
                 segmentation_result: SegmentationResult) -> Dict[str, Any]:
        """Evaluate if door is open based on edge detection and motion"""
        try:
            # Extract door region
            x, y, w, h = self.door_region['x'], self.door_region['y'], \
                        self.door_region['width'], self.door_region['height']
            door_roi = frame[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray_door = cv2.cvtColor(door_roi, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray_door, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Check for motion in door region
            motion_in_door = any(
                self._regions_overlap(region, self.door_region) 
                for region in motion_result.motion_regions
            )
            
            # Simple heuristic: door is open if low edge density or significant motion
            is_open = edge_density < self.threshold or motion_in_door
            confidence = abs(edge_density - self.threshold) + (0.3 if motion_in_door else 0)
            confidence = min(confidence, 1.0)
            
            # Check for state change
            state_changed = self.last_door_state is not None and self.last_door_state != is_open
            self.last_door_state = is_open
            
            return {
                'triggered': is_open,
                'state_changed': state_changed,
                'door_open': is_open,
                'edge_density': edge_density,
                'motion_detected': motion_in_door,
                'confidence': confidence
            }
            
        except Exception as e:
            logging.error(f"Error in door open rule evaluation: {e}")
            return {
                'triggered': False,
                'state_changed': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _regions_overlap(self, region1: Dict, region2: Dict) -> bool:
        """Check if two regions overlap"""
        x1, y1, w1, h1 = region1.get('x', 0), region1.get('y', 0), \
                         region1.get('width', 0), region1.get('height', 0)
        x2, y2, w2, h2 = region2.get('x', 0), region2.get('y', 0), \
                         region2.get('width', 0), region2.get('height', 0)
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

class ActivityDetectionRule(VisualRule):
    """Rule to detect general activity/movement"""
    
    def __init__(self, motion_threshold: float = 0.05):
        super().__init__("activity_detection", "Activity Detection", 
                        "Detects general activity based on motion patterns")
        self.motion_threshold = motion_threshold
        self.activity_history = []
        self.max_history = 10
    
    def evaluate(self, frame: np.ndarray, motion_result: MotionResult, 
                 segmentation_result: SegmentationResult) -> Dict[str, Any]:
        """Evaluate activity level"""
        try:
            # Calculate normalized motion area
            frame_area = frame.shape[0] * frame.shape[1]
            motion_ratio = motion_result.motion_area / frame_area if frame_area > 0 else 0
            
            # Add to history
            self.activity_history.append(motion_ratio)
            if len(self.activity_history) > self.max_history:
                self.activity_history.pop(0)
            
            # Calculate activity metrics
            avg_motion = np.mean(self.activity_history) if self.activity_history else 0
            motion_variance = np.var(self.activity_history) if len(self.activity_history) > 1 else 0
            
            # Activity detected if motion exceeds threshold
            activity_detected = motion_ratio > self.motion_threshold
            confidence = min(motion_ratio / self.motion_threshold, 1.0) if self.motion_threshold > 0 else 0
            
            return {
                'triggered': activity_detected,
                'activity_level': motion_ratio,
                'average_activity': avg_motion,
                'activity_variance': motion_variance,
                'confidence': confidence,
                'num_motion_regions': len(motion_result.motion_regions)
            }
            
        except Exception as e:
            logging.error(f"Error in activity detection rule evaluation: {e}")
            return {
                'triggered': False,
                'error': str(e),
                'confidence': 0.0
            }

class ComputerVisionAnalyticsTask(VisionTask):
    """Task for computer vision analytics processing"""
    
    def __init__(self, task_id: str, analytics_config: Dict[str, Any]):
        super().__init__(task_id, "computer_vision_analytics", **analytics_config)
        self.analytics_config = analytics_config
        self.motion_detector = MotionDetector(analytics_config.get('motion_config', {}))
        self.scene_segmenter = SceneSegmenter(analytics_config.get('segmentation_config', {}))
        self.visual_reasoner = VisualReasoner(analytics_config.get('reasoning_config', {}))
        self.event_emitter = AnalyticsEventEmitter()
    
    async def execute(self, input_data: Any) -> VisionResult:
        """Execute computer vision analytics on input data"""
        try:
            self.status = TaskStatus.IN_PROGRESS
            
            if isinstance(input_data, str):  # Video file path
                results = await self._process_video_file(input_data)
            elif isinstance(input_data, np.ndarray):  # Single frame
                results = await self._process_single_frame(input_data)
            elif isinstance(input_data, list):  # Multiple frames
                results = await self._process_frame_sequence(input_data)
            else:
                raise ValueError(f"Unsupported input data type: {type(input_data)}")
            
            return VisionResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                data=results,
                confidence=results.get('overall_confidence', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in analytics task execution: {e}")
            return VisionResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                data=None,
                confidence=0.0,
                error_message=str(e)
            )
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if isinstance(input_data, str):
            return Path(input_data).exists()
        elif isinstance(input_data, np.ndarray):
            return len(input_data.shape) == 3  # H, W, C
        elif isinstance(input_data, list):
            return all(isinstance(frame, np.ndarray) for frame in input_data)
        return False
    
    async def _process_video_file(self, video_path: str) -> Dict[str, Any]:
        """Process video file for analytics"""
        cap = cv2.VideoCapture(video_path)
        results = {
            'frames_processed': 0,
            'events_generated': [],
            'analytics_summary': {},
            'overall_confidence': 0.0
        }
        
        frame_count = 0
        total_confidence = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_result = await self._process_single_frame(frame, frame_count)
                results['events_generated'].extend(frame_result.get('events', []))
                total_confidence += frame_result.get('confidence', 0.0)
                frame_count += 1
                
                if frame_count % 100 == 0:  # Log progress
                    self.logger.info(f"Processed {frame_count} frames")
                    
        finally:
            cap.release()
        
        results['frames_processed'] = frame_count
        results['overall_confidence'] = total_confidence / max(frame_count, 1)
        results['analytics_summary'] = self._generate_analytics_summary(results['events_generated'])
        
        return results
    
    async def _process_single_frame(self, frame: np.ndarray, frame_index: int = 0) -> Dict[str, Any]:
        """Process single frame for analytics"""
        # Motion detection
        motion_result = self.motion_detector.detect_motion(frame)
        
        # Scene segmentation
        segmentation_result = self.scene_segmenter.segment_scene(frame)
        
        # Visual reasoning
        reasoning_result = self.visual_reasoner.evaluate_rules(
            frame, motion_result, segmentation_result
        )
        
        # Generate events
        events = self._generate_events(frame, motion_result, segmentation_result, 
                                     reasoning_result, frame_index)
        
        # Emit events to learning engine
        for event in events:
            self.event_emitter.emit_event(event)
        
        # Calculate overall confidence
        confidences = [
            motion_result.confidence,
            segmentation_result.confidence,
            reasoning_result.confidence
        ]
        overall_confidence = np.mean([c for c in confidences if c > 0])
        
        return {
            'motion_result': asdict(motion_result),
            'segmentation_result': asdict(segmentation_result),
            'reasoning_result': asdict(reasoning_result),
            'events': [asdict(event) for event in events],
            'confidence': overall_confidence
        }
    
    async def _process_frame_sequence(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Process sequence of frames"""
        all_events = []
        total_confidence = 0.0
        
        for i, frame in enumerate(frames):
            frame_result = await self._process_single_frame(frame, i)
            all_events.extend(frame_result.get('events', []))
            total_confidence += frame_result.get('confidence', 0.0)
        
        return {
            'frames_processed': len(frames),
            'events_generated': all_events,
            'analytics_summary': self._generate_analytics_summary(all_events),
            'overall_confidence': total_confidence / max(len(frames), 1)
        }
    
    def _generate_events(self, frame: np.ndarray, motion_result: MotionResult,
                        segmentation_result: SegmentationResult, 
                        reasoning_result: ReasoningResult,
                        frame_index: int) -> List[AnalyticsEvent]:
        """Generate analytics events based on processing results"""
        events = []
        
        # Motion events
        if motion_result.motion_detected:
            events.append(AnalyticsEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type=AnalyticsEventType.MOTION_DETECTED,
                confidence=motion_result.confidence,
                metadata={
                    'motion_area': motion_result.motion_area,
                    'num_regions': len(motion_result.motion_regions),
                    'regions': motion_result.motion_regions
                },
                frame_info={'frame_index': frame_index, 'frame_shape': frame.shape}
            ))
        
        # Rule-triggered events
        for rule_result in reasoning_result.triggered_rules:
            rule_name = rule_result.get('rule_name', 'unknown')
            
            if rule_name == 'door_open' and rule_result.get('state_changed', False):
                events.append(AnalyticsEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    event_type=AnalyticsEventType.DOOR_STATE_CHANGE,
                    confidence=rule_result.get('confidence', 0.0),
                    metadata={
                        'door_open': rule_result.get('door_open', False),
                        'edge_density': rule_result.get('edge_density', 0.0),
                        'motion_detected': rule_result.get('motion_detected', False)
                    }
                ))
            
            elif rule_name == 'activity_detection' and rule_result.get('triggered', False):
                events.append(AnalyticsEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    event_type=AnalyticsEventType.ACTIVITY_DETECTED,
                    confidence=rule_result.get('confidence', 0.0),
                    metadata={
                        'activity_level': rule_result.get('activity_level', 0.0),
                        'average_activity': rule_result.get('average_activity', 0.0),
                        'num_motion_regions': rule_result.get('num_motion_regions', 0)
                    }
                ))
            
            # General rule triggered event
            events.append(AnalyticsEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type=AnalyticsEventType.RULE_TRIGGERED,
                confidence=rule_result.get('confidence', 0.0),
                metadata={
                    'rule_name': rule_name,
                    'rule_result': rule_result
                }
            ))
        
        return events
    
    def _generate_analytics_summary(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of analytics events"""
        if not events:
            return {'total_events': 0}
        
        event_types = {}
        total_confidence = 0.0
        
        for event in events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
            total_confidence += event.get('confidence', 0.0)
        
        return {
            'total_events': len(events),
            'event_types': event_types,
            'average_confidence': total_confidence / len(events),
            'most_common_event': max(event_types, key=event_types.get) if event_types else None
        }

class MotionDetector:
    """Motion detection using frame differencing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config.get('threshold', 25)
        self.min_area = config.get('min_area', 500)
        self.blur_size = config.get('blur_size', 21)
        self.previous_frame = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def detect_motion(self, frame: np.ndarray) -> MotionResult:
        """Detect motion using frame differencing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            
            if self.previous_frame is None:
                self.previous_frame = gray
                return MotionResult(False, 0.0, [], 0.0)
            
            # Compute frame difference
            frame_diff = cv2.absdiff(self.previous_frame, gray)
            thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_regions = []
            total_motion_area = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append({
                    'x': int(x), 'y': int(y), 
                    'width': int(w), 'height': int(h),
                    'area': float(area)
                })
                total_motion_area += area
            
            # Calculate confidence based on motion area and number of regions
            frame_area = frame.shape[0] * frame.shape[1]
            motion_ratio = total_motion_area / frame_area
            confidence = min(motion_ratio * 10, 1.0)  # Scale up for better confidence values
            
            # Update previous frame
            self.previous_frame = gray.copy()
            
            return MotionResult(
                motion_detected=len(motion_regions) > 0,
                motion_area=total_motion_area,
                motion_regions=motion_regions,
                confidence=confidence,
                frame_diff=thresh
            )
            
        except Exception as e:
            self.logger.error(f"Error in motion detection: {e}")
            return MotionResult(False, 0.0, [], 0.0)

class SceneSegmenter:
    """Scene segmentation using color-based clustering"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_clusters = config.get('num_clusters', 8)
        self.resize_factor = config.get('resize_factor', 0.25)  # For performance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def segment_scene(self, frame: np.ndarray) -> SegmentationResult:
        """Segment scene using K-means clustering"""
        try:
            # Resize for performance
            height, width = frame.shape[:2]
            new_height = int(height * self.resize_factor)
            new_width = int(width * self.resize_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Reshape for clustering
            data = resized_frame.reshape(-1, 3).astype(np.float32)
            
            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, self.num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Reshape labels back to image shape
            labels = labels.reshape(new_height, new_width)
            
            # Calculate segment statistics
            segments = []
            for i in range(self.num_clusters):
                mask = (labels == i)
                area = np.sum(mask)
                if area > 0:
                    # Find bounding box
                    y_coords, x_coords = np.where(mask)
                    segments.append({
                        'cluster_id': int(i),
                        'area': int(area),
                        'color': centers[i].tolist(),
                        'bbox': {
                            'x': int(np.min(x_coords) / self.resize_factor),
                            'y': int(np.min(y_coords) / self.resize_factor),
                            'width': int((np.max(x_coords) - np.min(x_coords)) / self.resize_factor),
                            'height': int((np.max(y_coords) - np.min(y_coords)) / self.resize_factor)
                        }
                    })
            
            # Sort by area (dominant regions first)
            segments.sort(key=lambda x: x['area'], reverse=True)
            dominant_regions = segments[:3]  # Top 3 largest segments
            
            # Calculate scene complexity (entropy-based measure)
            areas = [s['area'] for s in segments]
            total_area = sum(areas)
            if total_area > 0:
                probs = [a / total_area for a in areas]
                scene_complexity = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
            else:
                scene_complexity = 0.0
            
            # Calculate confidence based on cluster separation
            if len(centers) > 1:
                # Calculate average distance between cluster centers
                distances = []
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        distances.append(dist)
                avg_separation = np.mean(distances) if distances else 0
                confidence = min(avg_separation / 255.0, 1.0)  # Normalize by max possible distance
            else:
                confidence = 0.5
            
            return SegmentationResult(
                segments=segments,
                dominant_regions=dominant_regions,
                scene_complexity=scene_complexity,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in scene segmentation: {e}")
            return SegmentationResult([], [], 0.0, 0.0)

class VisualReasoner:
    """Visual reasoning engine with rule-based logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: List[VisualRule] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default visual reasoning rules"""
        # Door detection rule (example configuration)
        door_config = self.config.get('door_detection', {})
        if door_config.get('enabled', True):
            door_region = door_config.get('region', {'x': 100, 'y': 50, 'width': 200, 'height': 300})
            threshold = door_config.get('threshold', 0.3)
            self.rules.append(DoorOpenRule(door_region, threshold))
        
        # Activity detection rule
        activity_config = self.config.get('activity_detection', {})
        if activity_config.get('enabled', True):
            motion_threshold = activity_config.get('motion_threshold', 0.05)
            self.rules.append(ActivityDetectionRule(motion_threshold))
    
    def add_rule(self, rule: VisualRule):
        """Add a custom visual reasoning rule"""
        self.rules.append(rule)
    
    def evaluate_rules(self, frame: np.ndarray, motion_result: MotionResult,
                      segmentation_result: SegmentationResult) -> ReasoningResult:
        """Evaluate all visual reasoning rules"""
        try:
            rules_evaluated = []
            triggered_rules = []
            scene_state = {}
            total_confidence = 0.0
            
            for rule in self.rules:
                rule_result = rule.evaluate(frame, motion_result, segmentation_result)
                rules_evaluated.append(rule.name)
                
                # Store in scene state
                scene_state[rule.rule_id] = rule_result
                
                # Check if rule was triggered
                if rule_result.get('triggered', False):
                    triggered_rules.append({
                        'rule_name': rule.rule_id,
                        'rule_description': rule.description,
                        'result': rule_result,
                        'confidence': rule_result.get('confidence', 0.0)
                    })
                
                total_confidence += rule_result.get('confidence', 0.0)
            
            # Calculate overall confidence
            overall_confidence = total_confidence / len(self.rules) if self.rules else 0.0
            
            return ReasoningResult(
                rules_evaluated=rules_evaluated,
                triggered_rules=triggered_rules,
                scene_state=scene_state,
                confidence=overall_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in visual reasoning: {e}")
            return ReasoningResult([], [], {}, 0.0)

class AnalyticsEventEmitter:
    """Emits analytics events to Learning Engine and other consumers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.event_queue = queue.Queue()
        self.learning_engine = global_learning_engine
        self.event_handlers = []
        
        # Start background event processing
        self._start_event_processor()
    
    def _start_event_processor(self):
        """Start background thread for event processing"""
        def process_events():
            while True:
                try:
                    event = self.event_queue.get(timeout=1)
                    self._process_event(event)
                    self.event_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing event: {e}")
        
        processor_thread = threading.Thread(target=process_events, daemon=True)
        processor_thread.start()
    
    def emit_event(self, event: AnalyticsEvent):
        """Emit an analytics event"""
        self.event_queue.put(event)
    
    def _process_event(self, event: AnalyticsEvent):
        """Process an analytics event"""
        try:
            # Send to Learning Engine
            self._send_to_learning_engine(event)
            
            # Log event
            self.logger.info(f"Analytics event: {event.event_type.value} "
                           f"(confidence: {event.confidence:.2f})")
            
            # Call registered event handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing analytics event: {e}")
    
    def _send_to_learning_engine(self, event: AnalyticsEvent):
        """Send event to Learning Engine for learning"""
        try:
            # Convert event to learning context
            context = {
                'component_type': 'computer_vision_analytics',
                'event_type': event.event_type.value,
                'confidence': event.confidence,
                'timestamp': event.timestamp.isoformat()
            }
            
            # Add metadata to context
            if event.metadata:
                context.update(event.metadata)
            
            # Record as learning experience
            action = f"detect_{event.event_type.value}"
            outcome = "successful_detection" if event.confidence > 0.5 else "low_confidence_detection"
            success = event.confidence > 0.3  # Threshold for success
            
            self.learning_engine.record_experience(
                context=context,
                action=action,
                outcome=outcome,
                success=success,
                feedback_score=event.confidence,
                metadata={
                    'event_id': event.event_id,
                    'location': event.location,
                    'frame_info': event.frame_info
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error sending event to learning engine: {e}")
    
    def add_event_handler(self, handler: callable):
        """Add custom event handler"""
        self.event_handlers.append(handler)

class ComputerVisionAnalyticsAgent(AIVisionAgent):
    """Specialized agent for computer vision analytics"""
    
    def __init__(self, agent_id: str = "cv_analytics_agent", **config):
        config.setdefault('max_concurrent_tasks', 3)
        super().__init__(agent_id, "Computer Vision Analytics Agent", **config)
        
        self.analytics_config = config.get('analytics_config', {})
        self.event_emitter = AnalyticsEventEmitter()
        
        # Add custom event handler for agent-level processing
        self.event_emitter.add_event_handler(self._handle_analytics_event)
        
        self.logger.info("Computer Vision Analytics Agent initialized")
    
    def create_analytics_task(self, analytics_config: Dict[str, Any] = None) -> str:
        """Create a new analytics task"""
        if analytics_config is None:
            analytics_config = self.analytics_config
        
        task_id = f"analytics_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        task = ComputerVisionAnalyticsTask(task_id, analytics_config)
        
        if self.register_task(task):
            self.logger.info(f"Created analytics task: {task_id}")
            return task_id
        else:
            raise RuntimeError(f"Failed to register analytics task: {task_id}")
    
    def _handle_analytics_event(self, event: AnalyticsEvent):
        """Handle analytics events at agent level"""
        # Agent can implement additional processing here
        # For example, maintaining statistics, triggering alerts, etc.
        pass
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of analytics processing"""
        completed_tasks = [task for task in self.tasks.values() 
                         if task.status == TaskStatus.COMPLETED]
        
        total_events = 0
        event_types = {}
        
        for task in completed_tasks:
            if task.result and task.result.data:
                task_events = task.result.data.get('events_generated', [])
                total_events += len(task_events)
                
                for event in task_events:
                    event_type = event.get('event_type', 'unknown')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'total_tasks': len(self.tasks),
            'completed_tasks': len(completed_tasks),
            'total_events_generated': total_events,
            'event_type_distribution': event_types,
            'agent_config': self.analytics_config
        }

# Default configuration
DEFAULT_ANALYTICS_CONFIG = {
    'motion_config': {
        'threshold': 25,
        'min_area': 500,
        'blur_size': 21
    },
    'segmentation_config': {
        'num_clusters': 8,
        'resize_factor': 0.25
    },
    'reasoning_config': {
        'door_detection': {
            'enabled': True,
            'region': {'x': 100, 'y': 50, 'width': 200, 'height': 300},
            'threshold': 0.3
        },
        'activity_detection': {
            'enabled': True,
            'motion_threshold': 0.05
        }
    }
}

# Global analytics agent instance
analytics_agent = ComputerVisionAnalyticsAgent(
    agent_id="global_cv_analytics",
    analytics_config=DEFAULT_ANALYTICS_CONFIG
)
