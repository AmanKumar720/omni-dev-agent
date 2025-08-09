"""
Multi-modal Integration Layer Package

This package provides comprehensive integration between vision and NLP pipelines
using state-of-the-art embedding models like CLIP and sentence transformers.

Key Features:
- Vision-text embedding alignment using CLIP
- Object detection integration with YOLO models  
- OCR text extraction and embedding
- Multi-modal search and query capabilities
- Natural language query parsing
- Video frame analysis and temporal search

Main Classes:
- MultiModalIntegrationLayer: Main integration layer
- EmbeddingEngine: Handles embedding generation for different modalities
- MultiModalQuery: Represents structured multi-modal queries
- FrameAnalysis: Analysis results for individual frames
- SearchResult: Results from multi-modal search operations

Example Usage:
    ```python
    import asyncio
    from multimodal import create_integration_layer, parse_natural_query
    
    async def main():
        # Create and initialize integration layer
        integration_layer = await create_integration_layer()
        
        # Analyze a frame
        analysis = await integration_layer.analyze_frame(frame, "frame_001")
        
        # Perform natural language search
        query = parse_natural_query("show all frames where a red car appears and text contains 'exit'")
        results = integration_layer.search(query)
        
        for result in results:
            print(f"Found match: {result.frame_analysis.frame_id}")
            print(f"Score: {result.similarity_score:.3f}")
            print(f"Explanation: {result.explanation}")
    
    asyncio.run(main())
    ```
"""

from .integration_layer import (
    # Main classes
    MultiModalIntegrationLayer,
    EmbeddingEngine,
    
    # Data structures
    MultiModalQuery,
    SearchResult,
    FrameAnalysis,
    EmbeddingResult,
    
    # Enums
    QueryType,
    ModalityType,
    
    # Factory functions
    create_integration_layer,
    parse_natural_query
)

# Version information
__version__ = "1.0.0"
__author__ = "Omni Dev Agent"
__description__ = "Multi-modal integration layer connecting vision and NLP pipelines"

# Package metadata
__all__ = [
    # Main classes
    "MultiModalIntegrationLayer",
    "EmbeddingEngine",
    
    # Data structures
    "MultiModalQuery", 
    "SearchResult",
    "FrameAnalysis",
    "EmbeddingResult",
    
    # Enums
    "QueryType",
    "ModalityType",
    
    # Factory functions
    "create_integration_layer",
    "parse_natural_query",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]

# Optional: Set up package-level logging
import logging

# Create package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Multi-modal integration layer package v{__version__} loaded")
