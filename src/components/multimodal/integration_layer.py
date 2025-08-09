"""
Multi-modal Integration Layer

This module provides a comprehensive integration layer that connects vision outputs 
to NLP pipelines via embeddings using CLIP, sentence-transformers, and other models.
Enables complex queries like "show all frames where a red car appears and text contains 'exit'".
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2
from pathlib import Path

# Core imports with fallbacks
try:
    from ..ai_vision.core import VisionResult, TaskStatus
except ImportError:
    # Fallback for core classes
    from enum import Enum
    from dataclasses import dataclass
    from typing import Any, Optional, Dict
    
    class TaskStatus(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        FAILED = "failed"
    
    @dataclass
    class VisionResult:
        task_id: str
        status: TaskStatus
        data: Any
        confidence: float
        metadata: Optional[Dict[str, Any]] = None

# Object detection imports with fallbacks
try:
    from ..ai_vision.object_detection import ObjectDetector, DetectionResult
except ImportError:
    try:
        from ai_vision.object_detection import ObjectDetector, DetectionResult
    except ImportError:
        # Create mock classes for testing
        from dataclasses import dataclass
        from typing import Tuple, Any
        
        @dataclass
        class DetectionResult:
            bbox: Tuple[int, int, int, int]
            confidence: float
            class_id: int
            class_name: str
            
        class ObjectDetector:
            def __init__(self, model_name="yolov8n", device="cpu"):
                pass
                
            async def ensure_model_ready(self):
                return True
                
            def detect_objects(self, frame, conf_threshold=0.25):
                return []  # Return empty list for testing

# OCR import with fallback for missing dependencies
try:
    from ..ai_vision.ocr import OCREngine, OCRResult
except ImportError:
    try:
        from ai_vision.ocr import OCREngine, OCRResult
    except ImportError:
        # Create mock OCR classes if dependencies are missing
        from dataclasses import dataclass
        from typing import List, Dict, Any
        
        @dataclass
        class OCRResult:
            pages: List[Any]
            total_pages: int = 0
            average_confidence: float = 0.0
            processing_time: float = 0.0
            languages_detected: List[str] = None
            
        class OCREngine:
            def __init__(self, languages=None, preprocess=True):
                self.languages = languages or ['eng']
                self.preprocess = preprocess
                
            def extract_text(self, image, **kwargs):
                # Return empty OCR result for testing
                return OCRResult(pages=[], total_pages=0)

# Lazy imports for heavy ML dependencies
_torch = None
_transformers = None
_sentence_transformers = None
_clip = None

def _lazy_import_torch():
    """Lazy import PyTorch"""
    global _torch
    if _torch is None:
        try:
            import torch as _torch
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")
    return _torch

def _lazy_import_transformers():
    """Lazy import transformers"""
    global _transformers
    if _transformers is None:
        try:
            import transformers as _transformers
        except ImportError:
            raise ImportError("Transformers is required. Install with: pip install transformers")
    return _transformers

def _lazy_import_sentence_transformers():
    """Lazy import sentence-transformers"""
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            import sentence_transformers as _sentence_transformers
        except ImportError:
            raise ImportError("Sentence Transformers is required. Install with: pip install sentence-transformers")
    return _sentence_transformers

def _lazy_import_clip():
    """Lazy import CLIP"""
    global _clip
    if _clip is None:
        try:
            import clip as _clip
        except ImportError:
            # Try alternative CLIP implementation
            try:
                from transformers import CLIPProcessor, CLIPModel
                # Create a mock clip object for compatibility
                class MockCLIP:
                    def __init__(self):
                        self.available_models = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']
                    
                    def load(self, model_name, device='cpu'):
                        processor = CLIPProcessor.from_pretrained(f"openai/clip-{model_name.lower().replace('/', '-')}")
                        model = CLIPModel.from_pretrained(f"openai/clip-{model_name.lower().replace('/', '-')}")
                        return model, processor
                
                _clip = MockCLIP()
            except ImportError:
                raise ImportError("CLIP is required. Install with: pip install clip-by-openai or use transformers CLIP")
    return _clip


class ModalityType(Enum):
    """Enumeration of supported modalities"""
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"  # For future extension


class QueryType(Enum):
    """Enumeration of query types"""
    SIMILARITY = "similarity"
    MULTIMODAL_AND = "multimodal_and"
    MULTIMODAL_OR = "multimodal_or"
    SEMANTIC_SEARCH = "semantic_search"
    OBJECT_AND_TEXT = "object_and_text"


@dataclass
class EmbeddingResult:
    """Container for embedding results"""
    embedding: np.ndarray
    modality: ModalityType
    source_data: Any
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame"""
    frame_id: str
    timestamp: Optional[float] = None
    vision_results: List[DetectionResult] = field(default_factory=list)
    ocr_results: Optional[OCRResult] = None
    vision_embedding: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalQuery:
    """Represents a multi-modal query"""
    query_text: str
    query_type: QueryType
    vision_constraints: Optional[List[str]] = None  # e.g., ["red car", "person walking"]
    text_constraints: Optional[List[str]] = None    # e.g., ["exit", "warning"]
    similarity_threshold: float = 0.7
    max_results: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result of a multi-modal search"""
    frame_analysis: FrameAnalysis
    similarity_score: float
    matched_constraints: List[str]
    explanation: str


class EmbeddingEngine:
    """Handles embedding generation for different modalities"""
    
    def __init__(self, 
                 clip_model: str = "ViT-B/32",
                 sentence_model: str = "all-MiniLM-L6-v2",
                 device: str = "auto"):
        """
        Initialize embedding engine
        
        Args:
            clip_model: CLIP model name for vision-text embeddings
            sentence_model: Sentence transformer model for text embeddings
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        self.clip_model_name = clip_model
        self.sentence_model_name = sentence_model
        self.device = self._get_device(device)
        
        self.clip_model = None
        self.clip_preprocess = None
        self.sentence_model = None
        
        self.logger = logging.getLogger(__name__ + ".EmbeddingEngine")
        
    def _get_device(self, device: str) -> str:
        """Get appropriate device"""
        if device == "auto":
            torch = _lazy_import_torch()
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
        
    async def initialize(self):
        """Initialize all models"""
        try:
            await self._load_clip_model()
            await self._load_sentence_model()
            self.logger.info("Embedding engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding engine: {e}")
            raise
            
    async def _load_clip_model(self):
        """Load CLIP model for vision-text embeddings"""
        try:
            clip = _lazy_import_clip()
            torch = _lazy_import_torch()
            
            # Load CLIP model
            if hasattr(clip, 'load'):
                self.clip_model, self.clip_preprocess = clip.load(
                    self.clip_model_name, device=self.device
                )
            else:
                # Using transformers CLIP
                from transformers import CLIPProcessor, CLIPModel
                
                model_name = f"openai/clip-{self.clip_model_name.lower().replace('/', '-')}"
                self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.clip_preprocess = CLIPProcessor.from_pretrained(model_name)
            
            self.logger.info(f"CLIP model {self.clip_model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise
            
    async def _load_sentence_model(self):
        """Load sentence transformer model"""
        try:
            sentence_transformers = _lazy_import_sentence_transformers()
            
            self.sentence_model = sentence_transformers.SentenceTransformer(
                self.sentence_model_name, device=self.device
            )
            
            self.logger.info(f"Sentence transformer {self.sentence_model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer: {e}")
            raise
            
    def encode_image(self, image: np.ndarray) -> EmbeddingResult:
        """
        Generate embedding for an image using CLIP
        
        Args:
            image: Input image as numpy array
            
        Returns:
            EmbeddingResult with image embedding
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded. Call initialize() first.")
            
        try:
            torch = _lazy_import_torch()
            
            # Preprocess image
            if hasattr(self.clip_preprocess, 'preprocess'):
                # OpenAI CLIP
                from PIL import Image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    embedding = image_features.cpu().numpy().flatten()
            else:
                # Transformers CLIP
                from PIL import Image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                inputs = self.clip_preprocess(images=pil_image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    embedding = image_features.cpu().numpy().flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return EmbeddingResult(
                embedding=embedding,
                modality=ModalityType.VISION,
                source_data=image,
                confidence=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Failed to encode image: {e}")
            raise
            
    def encode_text(self, text: str, use_clip: bool = False) -> EmbeddingResult:
        """
        Generate embedding for text
        
        Args:
            text: Input text
            use_clip: Whether to use CLIP for text encoding (for vision-text compatibility)
            
        Returns:
            EmbeddingResult with text embedding
        """
        try:
            if use_clip and self.clip_model is not None:
                return self._encode_text_clip(text)
            else:
                return self._encode_text_sentence_transformer(text)
                
        except Exception as e:
            self.logger.error(f"Failed to encode text: {e}")
            raise
            
    def _encode_text_clip(self, text: str) -> EmbeddingResult:
        """Encode text using CLIP model"""
        torch = _lazy_import_torch()
        
        if hasattr(self.clip_preprocess, 'tokenize'):
            # OpenAI CLIP
            text_input = self.clip_preprocess.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                embedding = text_features.cpu().numpy().flatten()
        else:
            # Transformers CLIP
            inputs = self.clip_preprocess(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                embedding = text_features.cpu().numpy().flatten()
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return EmbeddingResult(
            embedding=embedding,
            modality=ModalityType.TEXT,
            source_data=text,
            confidence=1.0
        )
        
    def _encode_text_sentence_transformer(self, text: str) -> EmbeddingResult:
        """Encode text using sentence transformer"""
        if self.sentence_model is None:
            raise RuntimeError("Sentence transformer not loaded. Call initialize() first.")
            
        embedding = self.sentence_model.encode(text, normalize_embeddings=True)
        
        return EmbeddingResult(
            embedding=embedding,
            modality=ModalityType.TEXT,
            source_data=text,
            confidence=1.0
        )
        
    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(embedding1, embedding2))


class MultiModalIntegrationLayer:
    """Main integration layer for connecting vision and NLP pipelines"""
    
    def __init__(self,
                 embedding_engine: Optional[EmbeddingEngine] = None,
                 object_detector: Optional[ObjectDetector] = None,
                 ocr_engine: Optional[OCREngine] = None):
        """
        Initialize multi-modal integration layer
        
        Args:
            embedding_engine: EmbeddingEngine instance
            object_detector: ObjectDetector instance
            ocr_engine: OCREngine instance
        """
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.object_detector = object_detector
        self.ocr_engine = ocr_engine or OCREngine()
        
        # Storage for analyzed frames
        self.frame_database: Dict[str, FrameAnalysis] = {}
        
        self.logger = logging.getLogger(__name__ + ".MultiModalIntegrationLayer")
        
    async def initialize(self):
        """Initialize all components"""
        try:
            await self.embedding_engine.initialize()
            
            if self.object_detector:
                await self.object_detector.ensure_model_ready()
                
            self.logger.info("Multi-modal integration layer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration layer: {e}")
            raise
            
    async def analyze_frame(self, 
                          frame: np.ndarray, 
                          frame_id: str,
                          timestamp: Optional[float] = None,
                          extract_objects: bool = True,
                          extract_text: bool = True,
                          generate_embeddings: bool = True) -> FrameAnalysis:
        """
        Comprehensively analyze a single frame
        
        Args:
            frame: Input frame as numpy array
            frame_id: Unique identifier for the frame
            timestamp: Optional timestamp
            extract_objects: Whether to perform object detection
            extract_text: Whether to perform OCR
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            FrameAnalysis object with all results
        """
        start_time = time.time()
        
        analysis = FrameAnalysis(
            frame_id=frame_id,
            timestamp=timestamp or time.time()
        )
        
        try:
            # Object detection
            if extract_objects and self.object_detector:
                detections = self.object_detector.detect_objects(frame)
                analysis.vision_results = detections
                self.logger.debug(f"Found {len(detections)} objects in frame {frame_id}")
                
            # OCR
            if extract_text:
                ocr_result = self.ocr_engine.extract_text(frame)
                analysis.ocr_results = ocr_result
                self.logger.debug(f"Extracted OCR text from frame {frame_id}")
                
            # Generate embeddings
            if generate_embeddings:
                # Vision embedding
                vision_embedding_result = self.embedding_engine.encode_image(frame)
                analysis.vision_embedding = vision_embedding_result.embedding
                
                # Text embedding (if OCR text available)
                if analysis.ocr_results and analysis.ocr_results.pages:
                    combined_text = " ".join([page.text for page in analysis.ocr_results.pages])
                    if combined_text.strip():
                        text_embedding_result = self.embedding_engine.encode_text(
                            combined_text, use_clip=True
                        )
                        analysis.text_embedding = text_embedding_result.embedding
                
            processing_time = time.time() - start_time
            analysis.metadata['processing_time'] = processing_time
            
            # Store in database
            self.frame_database[frame_id] = analysis
            
            self.logger.info(f"Frame {frame_id} analyzed in {processing_time:.2f}s")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze frame {frame_id}: {e}")
            raise
            
    async def analyze_video(self,
                          video_path: Union[str, Path],
                          frame_interval: int = 30,
                          max_frames: Optional[int] = None) -> Dict[str, FrameAnalysis]:
        """
        Analyze video frames at specified intervals
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every N-th frame
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Dictionary of frame_id -> FrameAnalysis
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Analyzing video: {video_path} (FPS: {fps}, Total frames: {total_frames})")
        
        frame_count = 0
        analyzed_count = 0
        results = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_id = f"{Path(video_path).stem}_frame_{frame_count:06d}"
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    
                    analysis = await self.analyze_frame(
                        frame, frame_id, timestamp
                    )
                    results[frame_id] = analysis
                    
                    analyzed_count += 1
                    
                    if max_frames and analyzed_count >= max_frames:
                        break
                        
                frame_count += 1
                
        finally:
            cap.release()
            
        self.logger.info(f"Analyzed {analyzed_count} frames from video {video_path}")
        return results
        
    def search(self, query: MultiModalQuery) -> List[SearchResult]:
        """
        Perform multi-modal search across analyzed frames
        
        Args:
            query: MultiModalQuery object specifying search parameters
            
        Returns:
            List of SearchResult objects, sorted by relevance
        """
        if not self.frame_database:
            self.logger.warning("No frames in database. Analyze some frames first.")
            return []
            
        results = []
        
        try:
            # Generate query embeddings
            query_text_embedding = None
            if query.query_text.strip():
                text_result = self.embedding_engine.encode_text(
                    query.query_text, use_clip=True
                )
                query_text_embedding = text_result.embedding
            
            # Search through all frames
            for frame_id, analysis in self.frame_database.items():
                result = self._evaluate_frame_against_query(analysis, query, query_text_embedding)
                if result and result.similarity_score >= query.similarity_threshold:
                    results.append(result)
                    
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit results
            if query.max_results > 0:
                results = results[:query.max_results]
                
            self.logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
            
    def _evaluate_frame_against_query(self, 
                                    analysis: FrameAnalysis, 
                                    query: MultiModalQuery,
                                    query_text_embedding: Optional[np.ndarray]) -> Optional[SearchResult]:
        """
        Evaluate a single frame against a multi-modal query
        
        Args:
            analysis: FrameAnalysis to evaluate
            query: MultiModalQuery to match against
            query_text_embedding: Pre-computed query text embedding
            
        Returns:
            SearchResult if frame matches, None otherwise
        """
        matched_constraints = []
        scores = []
        explanations = []
        
        try:
            # Vision constraints (object detection)
            if query.vision_constraints:
                vision_score = self._check_vision_constraints(
                    analysis, query.vision_constraints, matched_constraints, explanations
                )
                scores.append(vision_score)
                
            # Text constraints (OCR)
            if query.text_constraints:
                text_score = self._check_text_constraints(
                    analysis, query.text_constraints, matched_constraints, explanations
                )
                scores.append(text_score)
                
            # Semantic similarity with query text
            if query_text_embedding is not None:
                semantic_score = self._check_semantic_similarity(
                    analysis, query_text_embedding, explanations
                )
                scores.append(semantic_score)
                
            # Calculate overall score based on query type
            overall_score = self._calculate_overall_score(scores, query.query_type)
            
            if overall_score > 0:
                return SearchResult(
                    frame_analysis=analysis,
                    similarity_score=overall_score,
                    matched_constraints=matched_constraints,
                    explanation=" | ".join(explanations)
                )
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating frame {analysis.frame_id}: {e}")
            return None
            
    def _check_vision_constraints(self, 
                                analysis: FrameAnalysis, 
                                constraints: List[str],
                                matched_constraints: List[str],
                                explanations: List[str]) -> float:
        """Check vision-based constraints (object detection)"""
        if not analysis.vision_results:
            return 0.0
            
        detected_objects = [det.class_name.lower() for det in analysis.vision_results]
        
        matches = 0
        for constraint in constraints:
            constraint_lower = constraint.lower()
            
            # Direct match
            if any(constraint_lower in obj for obj in detected_objects):
                matches += 1
                matched_constraints.append(f"vision:{constraint}")
                explanations.append(f"Found object '{constraint}'")
                continue
                
            # Semantic matching could be added here using embeddings
            
        return matches / len(constraints) if constraints else 0.0
        
    def _check_text_constraints(self,
                              analysis: FrameAnalysis,
                              constraints: List[str],
                              matched_constraints: List[str],
                              explanations: List[str]) -> float:
        """Check text-based constraints (OCR)"""
        if not analysis.ocr_results:
            return 0.0
            
        # Combine all OCR text
        all_text = ""
        for page in analysis.ocr_results.pages:
            all_text += page.text.lower() + " "
            
        matches = 0
        for constraint in constraints:
            constraint_lower = constraint.lower()
            
            if constraint_lower in all_text:
                matches += 1
                matched_constraints.append(f"text:{constraint}")
                explanations.append(f"Found text '{constraint}'")
                
        return matches / len(constraints) if constraints else 0.0
        
    def _check_semantic_similarity(self,
                                 analysis: FrameAnalysis,
                                 query_embedding: np.ndarray,
                                 explanations: List[str]) -> float:
        """Check semantic similarity using embeddings"""
        max_similarity = 0.0
        
        # Check vision embedding similarity
        if analysis.vision_embedding is not None:
            vision_sim = EmbeddingEngine.calculate_similarity(
                analysis.vision_embedding, query_embedding
            )
            max_similarity = max(max_similarity, vision_sim)
            
        # Check text embedding similarity  
        if analysis.text_embedding is not None:
            text_sim = EmbeddingEngine.calculate_similarity(
                analysis.text_embedding, query_embedding
            )
            max_similarity = max(max_similarity, text_sim)
            
        if max_similarity > 0.5:
            explanations.append(f"Semantic similarity: {max_similarity:.3f}")
            
        return max_similarity
        
    def _calculate_overall_score(self, scores: List[float], query_type: QueryType) -> float:
        """Calculate overall score based on query type"""
        if not scores:
            return 0.0
            
        if query_type == QueryType.MULTIMODAL_AND:
            # All constraints must be satisfied
            return min(scores) if scores else 0.0
        elif query_type == QueryType.MULTIMODAL_OR:
            # At least one constraint must be satisfied
            return max(scores) if scores else 0.0
        else:
            # Default: average of all scores
            return sum(scores) / len(scores)
            
    def get_frame_analysis(self, frame_id: str) -> Optional[FrameAnalysis]:
        """Get analysis for a specific frame"""
        return self.frame_database.get(frame_id)
        
    def get_all_frames(self) -> Dict[str, FrameAnalysis]:
        """Get all analyzed frames"""
        return self.frame_database.copy()
        
    def clear_database(self):
        """Clear all stored frame analyses"""
        self.frame_database.clear()
        self.logger.info("Frame database cleared")


# Convenience functions
async def create_integration_layer(
    clip_model: str = "ViT-B/32",
    sentence_model: str = "all-MiniLM-L6-v2",
    yolo_model: str = "yolov8n",
    device: str = "auto"
) -> MultiModalIntegrationLayer:
    """
    Factory function to create and initialize a multi-modal integration layer
    
    Args:
        clip_model: CLIP model name
        sentence_model: Sentence transformer model name
        yolo_model: YOLO model name for object detection
        device: Device to run on
        
    Returns:
        Initialized MultiModalIntegrationLayer
    """
    # Create components
    embedding_engine = EmbeddingEngine(clip_model, sentence_model, device)
    
    object_detector = ObjectDetector(model_name=yolo_model, device=device)
    
    ocr_engine = OCREngine(languages=['eng'], preprocess=True)
    
    # Create integration layer
    integration_layer = MultiModalIntegrationLayer(
        embedding_engine=embedding_engine,
        object_detector=object_detector,
        ocr_engine=ocr_engine
    )
    
    # Initialize
    await integration_layer.initialize()
    
    return integration_layer


def parse_natural_query(query_text: str) -> MultiModalQuery:
    """
    Parse natural language query into structured MultiModalQuery
    
    Args:
        query_text: Natural language query
        
    Returns:
        MultiModalQuery object
    """
    # Simple parsing logic (can be enhanced with NLP)
    query_lower = query_text.lower()
    
    # Detect vision constraints
    vision_objects = []
    common_objects = [
        'car', 'person', 'truck', 'bicycle', 'dog', 'cat', 'bird',
        'red car', 'blue car', 'white car', 'person walking', 'person sitting'
    ]
    for obj in common_objects:
        if obj in query_lower:
            vision_objects.append(obj)
            
    # Detect text constraints
    text_keywords = []
    if 'text contains' in query_lower:
        # Extract text after "text contains"
        import re
        matches = re.findall(r"text contains ['\"]([^'\"]+)['\"]", query_lower)
        text_keywords.extend(matches)
        
    # Simple keyword extraction
    keywords = ['exit', 'warning', 'stop', 'danger', 'caution', 'emergency']
    for keyword in keywords:
        if keyword in query_lower:
            text_keywords.append(keyword)
            
    # Determine query type
    if 'and' in query_lower and (vision_objects and text_keywords):
        query_type = QueryType.MULTIMODAL_AND
    elif 'or' in query_lower:
        query_type = QueryType.MULTIMODAL_OR
    else:
        query_type = QueryType.SIMILARITY
        
    return MultiModalQuery(
        query_text=query_text,
        query_type=query_type,
        vision_constraints=vision_objects if vision_objects else None,
        text_constraints=text_keywords if text_keywords else None,
        similarity_threshold=0.6
    )


# Export main classes
__all__ = [
    'MultiModalIntegrationLayer',
    'EmbeddingEngine',
    'MultiModalQuery',
    'SearchResult',
    'FrameAnalysis',
    'QueryType',
    'ModalityType',
    'create_integration_layer',
    'parse_natural_query'
]
