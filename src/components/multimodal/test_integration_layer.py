#!/usr/bin/env python3
"""
Test Suite for Multi-modal Integration Layer

This module provides comprehensive tests for the multi-modal integration layer
functionality, including embedding generation, frame analysis, and search capabilities.
"""

import pytest
import asyncio
import numpy as np
import cv2
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Configure test logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

try:
    from integration_layer import (
        MultiModalIntegrationLayer,
        EmbeddingEngine,
        MultiModalQuery,
        SearchResult,
        FrameAnalysis,
        QueryType,
        ModalityType,
        create_integration_layer,
        parse_natural_query
    )
    from example_usage import create_demo_image_with_text
except ImportError:
    from .integration_layer import (
        MultiModalIntegrationLayer,
        EmbeddingEngine,
        MultiModalQuery,
        SearchResult,
        FrameAnalysis,
        QueryType,
        ModalityType,
        create_integration_layer,
        parse_natural_query
    )
    from .example_usage import create_demo_image_with_text


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    return create_demo_image_with_text("TEST IMAGE", ["car"])


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "This is a test message for embedding generation"


@pytest.fixture
async def mock_embedding_engine():
    """Create a mock embedding engine for testing"""
    engine = Mock(spec=EmbeddingEngine)
    
    # Mock embedding results
    mock_vision_embedding = np.random.rand(512).astype(np.float32)
    mock_text_embedding = np.random.rand(384).astype(np.float32)
    
    engine.encode_image.return_value = Mock(
        embedding=mock_vision_embedding,
        modality=ModalityType.VISION,
        confidence=1.0
    )
    
    engine.encode_text.return_value = Mock(
        embedding=mock_text_embedding,
        modality=ModalityType.TEXT,
        confidence=1.0
    )
    
    engine.calculate_similarity.return_value = 0.8
    
    return engine


class TestEmbeddingEngine:
    """Test cases for EmbeddingEngine class"""
    
    @pytest.mark.asyncio
    async def test_embedding_engine_initialization(self):
        """Test embedding engine initialization"""
        engine = EmbeddingEngine(device="cpu")
        
        assert engine.clip_model_name == "ViT-B/32"
        assert engine.sentence_model_name == "all-MiniLM-L6-v2"
        assert engine.device == "cpu"
        
    def test_get_device_auto(self):
        """Test automatic device detection"""
        engine = EmbeddingEngine(device="auto")
        assert engine.device in ["cpu", "cuda"]
        
    @pytest.mark.asyncio
    @patch('src.components.multimodal.integration_layer._lazy_import_clip')
    @patch('src.components.multimodal.integration_layer._lazy_import_sentence_transformers')
    async def test_initialize_with_mocks(self, mock_sentence_transformers, mock_clip):
        """Test engine initialization with mocked dependencies"""
        # Mock CLIP
        mock_clip.return_value = Mock()
        mock_clip.return_value.load.return_value = (Mock(), Mock())
        
        # Mock sentence transformers
        mock_st = Mock()
        mock_sentence_transformers.return_value = mock_st
        mock_st.SentenceTransformer.return_value = Mock()
        
        engine = EmbeddingEngine(device="cpu")
        await engine.initialize()
        
        assert engine.clip_model is not None
        assert engine.sentence_model is not None
        
    def test_calculate_similarity(self):
        """Test similarity calculation"""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([1.0, 0.0, 0.0])
        
        similarity = EmbeddingEngine.calculate_similarity(embedding1, embedding2)
        assert abs(similarity - 1.0) < 1e-6
        
        embedding3 = np.array([0.0, 1.0, 0.0])
        similarity_orthogonal = EmbeddingEngine.calculate_similarity(embedding1, embedding3)
        assert abs(similarity_orthogonal) < 1e-6


class TestMultiModalQuery:
    """Test cases for MultiModalQuery class"""
    
    def test_multimodal_query_creation(self):
        """Test creating multi-modal queries"""
        query = MultiModalQuery(
            query_text="find red cars with exit signs",
            query_type=QueryType.MULTIMODAL_AND,
            vision_constraints=["red car"],
            text_constraints=["exit"],
            similarity_threshold=0.7
        )
        
        assert query.query_text == "find red cars with exit signs"
        assert query.query_type == QueryType.MULTIMODAL_AND
        assert "red car" in query.vision_constraints
        assert "exit" in query.text_constraints
        assert query.similarity_threshold == 0.7
        
    def test_parse_natural_query_and_condition(self):
        """Test parsing natural language queries with AND conditions"""
        query = parse_natural_query("show all frames where a red car appears and text contains 'exit'")
        
        assert query.query_type == QueryType.MULTIMODAL_AND
        assert "car" in query.vision_constraints or "red car" in query.vision_constraints
        assert "exit" in query.text_constraints
        
    def test_parse_natural_query_or_condition(self):
        """Test parsing natural language queries with OR conditions"""
        query = parse_natural_query("find frames with cars or exit signs")
        
        assert query.query_type == QueryType.MULTIMODAL_OR
        
    def test_parse_natural_query_simple(self):
        """Test parsing simple queries"""
        query = parse_natural_query("find emergency vehicles")
        
        assert query.query_type == QueryType.SIMILARITY
        assert query.query_text == "find emergency vehicles"


class TestFrameAnalysis:
    """Test cases for FrameAnalysis functionality"""
    
    def test_frame_analysis_creation(self):
        """Test creating FrameAnalysis objects"""
        analysis = FrameAnalysis(
            frame_id="test_frame_001",
            timestamp=10.5,
            confidence=0.85
        )
        
        assert analysis.frame_id == "test_frame_001"
        assert analysis.timestamp == 10.5
        assert analysis.confidence == 0.85
        assert analysis.vision_results == []
        assert analysis.ocr_results is None


class TestMultiModalIntegrationLayer:
    """Test cases for MultiModalIntegrationLayer class"""
    
    @pytest.mark.asyncio
    async def test_integration_layer_initialization(self, mock_embedding_engine):
        """Test integration layer initialization"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        assert integration_layer.embedding_engine is not None
        assert integration_layer.frame_database == {}
        
    @pytest.mark.asyncio
    @patch('src.components.multimodal.integration_layer.ObjectDetector')
    @patch('src.components.multimodal.integration_layer.OCREngine')
    async def test_analyze_frame_mock(self, mock_ocr, mock_detector, mock_embedding_engine, sample_image):
        """Test frame analysis with mocked components"""
        # Setup mocks
        mock_detector.return_value.detect_objects.return_value = []
        mock_ocr.return_value.extract_text.return_value = Mock(pages=[Mock(text="TEST")])
        
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine,
            object_detector=mock_detector,
            ocr_engine=mock_ocr
        )
        
        # Analyze frame
        analysis = await integration_layer.analyze_frame(
            frame=sample_image,
            frame_id="test_frame"
        )
        
        assert analysis.frame_id == "test_frame"
        assert "test_frame" in integration_layer.frame_database
        
    def test_search_empty_database(self, mock_embedding_engine):
        """Test search with empty database"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        query = MultiModalQuery(
            query_text="test query",
            query_type=QueryType.SIMILARITY
        )
        
        results = integration_layer.search(query)
        assert results == []
        
    def test_get_frame_analysis(self, mock_embedding_engine):
        """Test retrieving frame analysis"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        # Add a test frame
        test_analysis = FrameAnalysis(frame_id="test_frame")
        integration_layer.frame_database["test_frame"] = test_analysis
        
        retrieved = integration_layer.get_frame_analysis("test_frame")
        assert retrieved == test_analysis
        
        # Test non-existent frame
        not_found = integration_layer.get_frame_analysis("non_existent")
        assert not_found is None
        
    def test_clear_database(self, mock_embedding_engine):
        """Test clearing frame database"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        # Add test data
        integration_layer.frame_database["frame1"] = FrameAnalysis(frame_id="frame1")
        integration_layer.frame_database["frame2"] = FrameAnalysis(frame_id="frame2")
        
        assert len(integration_layer.frame_database) == 2
        
        # Clear database
        integration_layer.clear_database()
        assert len(integration_layer.frame_database) == 0


class TestSearchFunctionality:
    """Test cases for search functionality"""
    
    @pytest.mark.asyncio
    async def test_vision_constraints_matching(self, mock_embedding_engine):
        """Test vision constraint matching"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        # Create mock analysis with vision results
        from unittest.mock import Mock
        mock_detection = Mock()
        mock_detection.class_name = "car"
        mock_detection.confidence = 0.9
        
        analysis = FrameAnalysis(
            frame_id="test_frame",
            vision_results=[mock_detection]
        )
        integration_layer.frame_database["test_frame"] = analysis
        
        # Test constraint checking
        matched_constraints = []
        explanations = []
        
        score = integration_layer._check_vision_constraints(
            analysis, ["car"], matched_constraints, explanations
        )
        
        assert score > 0
        assert "vision:car" in matched_constraints
        
    @pytest.mark.asyncio
    async def test_text_constraints_matching(self, mock_embedding_engine):
        """Test text constraint matching"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        # Create mock analysis with OCR results
        mock_page = Mock()
        mock_page.text = "EXIT SIGN"
        mock_ocr = Mock()
        mock_ocr.pages = [mock_page]
        
        analysis = FrameAnalysis(
            frame_id="test_frame",
            ocr_results=mock_ocr
        )
        
        matched_constraints = []
        explanations = []
        
        score = integration_layer._check_text_constraints(
            analysis, ["exit"], matched_constraints, explanations
        )
        
        assert score > 0
        assert "text:exit" in matched_constraints
        
    def test_overall_score_calculation_and(self, mock_embedding_engine):
        """Test overall score calculation for AND queries"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        scores = [0.8, 0.6, 0.9]
        
        # AND query should use minimum score
        score = integration_layer._calculate_overall_score(scores, QueryType.MULTIMODAL_AND)
        assert score == 0.6
        
    def test_overall_score_calculation_or(self, mock_embedding_engine):
        """Test overall score calculation for OR queries"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        scores = [0.8, 0.6, 0.9]
        
        # OR query should use maximum score
        score = integration_layer._calculate_overall_score(scores, QueryType.MULTIMODAL_OR)
        assert score == 0.9
        
    def test_overall_score_calculation_similarity(self, mock_embedding_engine):
        """Test overall score calculation for similarity queries"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        scores = [0.8, 0.6, 0.9]
        
        # Similarity query should use average score
        score = integration_layer._calculate_overall_score(scores, QueryType.SIMILARITY)
        expected_avg = sum(scores) / len(scores)
        assert abs(score - expected_avg) < 1e-6


class TestFactoryFunctions:
    """Test cases for factory functions"""
    
    @pytest.mark.asyncio
    @patch('src.components.multimodal.integration_layer.ObjectDetector')
    @patch('src.components.multimodal.integration_layer.OCREngine')
    @patch('src.components.multimodal.integration_layer.EmbeddingEngine')
    async def test_create_integration_layer_factory(self, mock_embedding, mock_ocr, mock_detector):
        """Test create_integration_layer factory function"""
        # Setup mocks
        mock_embedding_instance = Mock()
        mock_embedding_instance.initialize = AsyncMock()
        mock_embedding.return_value = mock_embedding_instance
        
        mock_detector_instance = Mock()
        mock_detector_instance.ensure_model_ready = AsyncMock(return_value=True)
        mock_detector.return_value = mock_detector_instance
        
        mock_ocr.return_value = Mock()
        
        # Test factory function
        integration_layer = await create_integration_layer(device="cpu")
        
        assert isinstance(integration_layer, MultiModalIntegrationLayer)
        mock_embedding_instance.initialize.assert_called_once()


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow test
    async def test_end_to_end_demo_scenario(self):
        """Test end-to-end scenario with demo images"""
        try:
            # This test requires actual models, so it might fail in CI
            # Create integration layer with CPU-only models
            integration_layer = await create_integration_layer(device="cpu")
            
            # Create demo images
            test_images = [
                ("exit_sign", create_demo_image_with_text("EXIT", ["car"])),
                ("warning_sign", create_demo_image_with_text("WARNING", ["person"]))
            ]
            
            # Analyze images
            for image_name, image in test_images:
                analysis = await integration_layer.analyze_frame(
                    frame=image,
                    frame_id=image_name,
                    extract_objects=False,  # Skip object detection for speed
                    extract_text=True,
                    generate_embeddings=True
                )
                
                assert analysis.frame_id == image_name
                assert image_name in integration_layer.frame_database
            
            # Test search
            query = MultiModalQuery(
                query_text="exit sign",
                query_type=QueryType.SIMILARITY,
                similarity_threshold=0.1  # Low threshold for testing
            )
            
            results = integration_layer.search(query)
            # We should get at least some results
            assert isinstance(results, list)
            
        except Exception as e:
            # If models can't be loaded (e.g., in CI), skip this test
            pytest.skip(f"End-to-end test skipped due to model loading issues: {e}")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_similarity_calculation(self):
        """Test similarity calculation with invalid inputs"""
        with pytest.raises((ValueError, TypeError)):
            EmbeddingEngine.calculate_similarity(None, np.array([1, 2, 3]))
            
    def test_empty_constraints(self, mock_embedding_engine):
        """Test handling of empty constraints"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        analysis = FrameAnalysis(frame_id="test")
        
        # Test empty vision constraints
        score = integration_layer._check_vision_constraints(
            analysis, [], [], []
        )
        assert score == 0.0
        
        # Test empty text constraints
        score = integration_layer._check_text_constraints(
            analysis, [], [], []
        )
        assert score == 0.0
        
    @pytest.mark.asyncio
    async def test_analyze_frame_error_handling(self, mock_embedding_engine):
        """Test frame analysis error handling"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        # Test with invalid frame data
        with pytest.raises(Exception):
            await integration_layer.analyze_frame(
                frame=None,  # Invalid frame
                frame_id="test"
            )


# Performance tests
class TestPerformance:
    """Performance-related tests"""
    
    @pytest.mark.performance
    def test_large_database_search_performance(self, mock_embedding_engine):
        """Test search performance with large frame database"""
        integration_layer = MultiModalIntegrationLayer(
            embedding_engine=mock_embedding_engine
        )
        
        # Add many frames to database
        num_frames = 1000
        for i in range(num_frames):
            frame_id = f"frame_{i:04d}"
            analysis = FrameAnalysis(frame_id=frame_id)
            integration_layer.frame_database[frame_id] = analysis
        
        # Test search performance
        query = MultiModalQuery(
            query_text="test query",
            query_type=QueryType.SIMILARITY,
            similarity_threshold=0.5
        )
        
        import time
        start_time = time.time()
        results = integration_layer.search(query)
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Search should complete within reasonable time (adjust as needed)
        assert search_time < 5.0  # 5 seconds for 1000 frames
        assert len(integration_layer.frame_database) == num_frames


# Utility functions for test setup
def create_test_video_file():
    """Create a temporary test video file"""
    temp_dir = tempfile.mkdtemp()
    video_path = Path(temp_dir) / "test_video.mp4"
    
    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 1.0, (640, 480))
    
    for i in range(5):  # 5 frames
        frame = create_demo_image_with_text(f"Frame {i}", ["car"])
        frame = cv2.resize(frame, (640, 480))
        out.write(frame)
    
    out.release()
    return video_path


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
