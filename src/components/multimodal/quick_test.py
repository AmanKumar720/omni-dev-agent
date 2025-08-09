#!/usr/bin/env python3
"""
Quick Integration Test for Multi-modal Integration Layer

This script performs a basic test to ensure the multi-modal integration layer
is working correctly without requiring heavy model downloads.
"""

import asyncio
import logging
import numpy as np
import cv2
from pathlib import Path

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "TEST EXIT SIGN", (50, 100), font, 0.7, (0, 0, 0), 2)
    
    # Add a simple rectangle to represent a car
    cv2.rectangle(img, (50, 120), (150, 160), (0, 0, 255), -1)
    cv2.putText(img, "CAR", (80, 145), font, 0.5, (255, 255, 255), 1)
    
    return img

async def test_basic_functionality():
    """Test basic functionality with mocked components"""
    logger.info("Testing multi-modal integration layer basic functionality...")
    
    try:
        # Import with fallbacks
        from integration_layer import (
            MultiModalIntegrationLayer,
            EmbeddingEngine, 
            MultiModalQuery,
            QueryType,
            FrameAnalysis
        )
        
        logger.info("✓ Successfully imported multi-modal components")
        
        # Test data structures
        query = MultiModalQuery(
            query_text="find cars with exit signs",
            query_type=QueryType.MULTIMODAL_AND,
            vision_constraints=["car"],
            text_constraints=["exit"],
            similarity_threshold=0.6
        )
        
        logger.info("✓ Successfully created MultiModalQuery")
        
        # Test frame analysis structure
        analysis = FrameAnalysis(
            frame_id="test_frame_001",
            timestamp=10.5
        )
        
        assert analysis.frame_id == "test_frame_001"
        assert analysis.timestamp == 10.5
        logger.info("✓ Successfully created FrameAnalysis")
        
        # Test embedding engine initialization (without models)
        embedding_engine = EmbeddingEngine(device="cpu")
        assert embedding_engine.device == "cpu"
        logger.info("✓ Successfully initialized EmbeddingEngine")
        
        # Test integration layer creation (without initialization)
        integration_layer = MultiModalIntegrationLayer(embedding_engine=embedding_engine)
        assert integration_layer.embedding_engine is not None
        assert len(integration_layer.frame_database) == 0
        logger.info("✓ Successfully created MultiModalIntegrationLayer")
        
        # Test database operations
        test_analysis = FrameAnalysis(frame_id="test", timestamp=0.0)
        integration_layer.frame_database["test"] = test_analysis
        
        retrieved = integration_layer.get_frame_analysis("test")
        assert retrieved is not None
        assert retrieved.frame_id == "test"
        logger.info("✓ Database operations working correctly")
        
        integration_layer.clear_database()
        assert len(integration_layer.frame_database) == 0
        logger.info("✓ Database clearing working correctly")
        
        logger.info("🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

def test_query_parsing():
    """Test natural language query parsing"""
    logger.info("Testing query parsing...")
    
    try:
        from integration_layer import parse_natural_query, QueryType
        
        # Test AND query
        query1 = parse_natural_query("show all frames where a red car appears and text contains 'exit'")
        assert query1.query_type == QueryType.MULTIMODAL_AND
        logger.info("✓ AND query parsing works")
        
        # Test OR query  
        query2 = parse_natural_query("find frames with cars or exit signs")
        assert query2.query_type == QueryType.MULTIMODAL_OR
        logger.info("✓ OR query parsing works")
        
        # Test simple query
        query3 = parse_natural_query("emergency vehicles")
        assert query3.query_type == QueryType.SIMILARITY
        logger.info("✓ Simple query parsing works")
        
        logger.info("🎉 All query parsing tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Query parsing test failed: {e}")
        return False

def test_similarity_calculation():
    """Test embedding similarity calculation"""
    logger.info("Testing similarity calculation...")
    
    try:
        from integration_layer import EmbeddingEngine
        
        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        sim = EmbeddingEngine.calculate_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 1e-6
        logger.info("✓ Identical vector similarity = 1.0")
        
        # Test orthogonal vectors
        vec3 = np.array([0.0, 1.0, 0.0])
        sim_orthogonal = EmbeddingEngine.calculate_similarity(vec1, vec3)
        assert abs(sim_orthogonal) < 1e-6
        logger.info("✓ Orthogonal vector similarity ≈ 0.0")
        
        logger.info("🎉 Similarity calculation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Similarity calculation test failed: {e}")
        return False

def test_search_logic():
    """Test search logic without models"""
    logger.info("Testing search logic...")
    
    try:
        from integration_layer import MultiModalIntegrationLayer, QueryType
        from unittest.mock import Mock
        
        # Create mock embedding engine
        mock_engine = Mock()
        mock_engine.encode_text.return_value = Mock(embedding=np.random.rand(512))
        
        integration_layer = MultiModalIntegrationLayer(embedding_engine=mock_engine)
        
        # Test score calculation for different query types
        scores = [0.8, 0.6, 0.9]
        
        # AND query (minimum)
        and_score = integration_layer._calculate_overall_score(scores, QueryType.MULTIMODAL_AND)
        assert and_score == 0.6
        logger.info("✓ AND query scoring (minimum) works")
        
        # OR query (maximum)
        or_score = integration_layer._calculate_overall_score(scores, QueryType.MULTIMODAL_OR)
        assert or_score == 0.9
        logger.info("✓ OR query scoring (maximum) works")
        
        # Similarity query (average)
        sim_score = integration_layer._calculate_overall_score(scores, QueryType.SIMILARITY)
        expected_avg = sum(scores) / len(scores)
        assert abs(sim_score - expected_avg) < 1e-6
        logger.info("✓ Similarity query scoring (average) works")
        
        logger.info("🎉 Search logic tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Search logic test failed: {e}")
        return False

async def main():
    """Run all quick tests"""
    logger.info("=" * 60)
    logger.info("Multi-modal Integration Layer - Quick Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality()),
        ("Query Parsing", test_query_parsing()),
        ("Similarity Calculation", test_similarity_calculation()),
        ("Search Logic", test_search_logic()),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Tests ---")
        
        if asyncio.iscoroutine(test_func):
            result = await test_func
        else:
            result = test_func
            
        if result:
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Multi-modal integration layer is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Install required ML models: pip install torch transformers sentence-transformers")
        logger.info("2. Run the full example: python example_usage.py")
        logger.info("3. Check out the comprehensive tests: python -m pytest test_integration_layer.py")
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
