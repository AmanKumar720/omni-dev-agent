#!/usr/bin/env python3
"""
Multi-modal Integration Layer - Example Usage

This script demonstrates the capabilities of the multi-modal integration layer,
showing how to connect vision outputs to NLP pipelines via embeddings.
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from integration_layer import (
        MultiModalIntegrationLayer, 
        MultiModalQuery, 
        QueryType,
        create_integration_layer,
        parse_natural_query
    )
except ImportError:
    from .integration_layer import (
        MultiModalIntegrationLayer, 
        MultiModalQuery, 
        QueryType,
        create_integration_layer,
        parse_natural_query
    )


def create_demo_image_with_text(text: str, objects: list = None) -> np.ndarray:
    """
    Create a demo image with text and simple objects for testing
    
    Args:
        text: Text to render in the image
        objects: List of simple objects to draw (e.g., ['car', 'person'])
    
    Returns:
        Demo image as numpy array
    """
    # Create a blank image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 0)  # Black
    thickness = 2
    
    # Calculate text position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = img.shape[0] - 50
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    
    # Add simple objects if specified
    if objects:
        for i, obj in enumerate(objects):
            if obj.lower() == 'car':
                # Draw a simple car (rectangle with wheels)
                cv2.rectangle(img, (100 + i*200, 150), (200 + i*200, 220), (0, 0, 255), -1)
                cv2.circle(img, (120 + i*200, 230), 15, (0, 0, 0), -1)
                cv2.circle(img, (180 + i*200, 230), 15, (0, 0, 0), -1)
                cv2.putText(img, 'CAR', (125 + i*200, 185), font, 0.7, (255, 255, 255), 2)
                
            elif obj.lower() == 'person':
                # Draw a simple person (stick figure)
                center_x = 150 + i*200
                # Head
                cv2.circle(img, (center_x, 80), 20, (0, 255, 0), -1)
                # Body
                cv2.line(img, (center_x, 100), (center_x, 180), (0, 255, 0), 3)
                # Arms
                cv2.line(img, (center_x, 130), (center_x-30, 150), (0, 255, 0), 3)
                cv2.line(img, (center_x, 130), (center_x+30, 150), (0, 255, 0), 3)
                # Legs
                cv2.line(img, (center_x, 180), (center_x-20, 230), (0, 255, 0), 3)
                cv2.line(img, (center_x, 180), (center_x+20, 230), (0, 255, 0), 3)
                cv2.putText(img, 'PERSON', (center_x-35, 250), font, 0.5, (0, 255, 0), 2)
    
    return img


async def demo_basic_analysis():
    """Demonstrate basic frame analysis capabilities"""
    logger.info("=== Basic Frame Analysis Demo ===")
    
    # Create integration layer
    integration_layer = await create_integration_layer(device="cpu")
    
    # Create demo images
    images = [
        ("exit_with_car", create_demo_image_with_text("EXIT", ["car"])),
        ("warning_with_person", create_demo_image_with_text("WARNING", ["person"])),
        ("stop_sign", create_demo_image_with_text("STOP")),
        ("emergency_exit", create_demo_image_with_text("EMERGENCY EXIT", ["person", "car"]))
    ]
    
    # Analyze each image
    for image_name, image in images:
        logger.info(f"\nAnalyzing image: {image_name}")
        
        try:
            analysis = await integration_layer.analyze_frame(
                frame=image,
                frame_id=image_name,
                extract_objects=True,
                extract_text=True,
                generate_embeddings=True
            )
            
            logger.info(f"  Objects detected: {len(analysis.vision_results)}")
            for detection in analysis.vision_results:
                logger.info(f"    - {detection.class_name}: {detection.confidence:.2f}")
            
            if analysis.ocr_results:
                logger.info(f"  OCR text: '{analysis.ocr_results.pages[0].text.strip()}'")
                logger.info(f"  OCR confidence: {analysis.ocr_results.pages[0].confidence:.2f}%")
            
            logger.info(f"  Vision embedding shape: {analysis.vision_embedding.shape if analysis.vision_embedding is not None else 'None'}")
            logger.info(f"  Text embedding shape: {analysis.text_embedding.shape if analysis.text_embedding is not None else 'None'}")
            logger.info(f"  Processing time: {analysis.metadata.get('processing_time', 0):.2f}s")
            
        except Exception as e:
            logger.error(f"  Error analyzing {image_name}: {e}")
    
    return integration_layer


async def demo_multimodal_search(integration_layer):
    """Demonstrate multi-modal search capabilities"""
    logger.info("\n=== Multi-modal Search Demo ===")
    
    # Define various search queries
    queries = [
        "show all frames where a red car appears and text contains 'exit'",
        "find frames with warning signs",
        "locate emergency exits with people",
        "find any frames with cars or exit text"
    ]
    
    for query_text in queries:
        logger.info(f"\nSearching for: '{query_text}'")
        
        try:
            # Parse natural language query
            query = parse_natural_query(query_text)
            
            logger.info(f"  Parsed query type: {query.query_type.value}")
            logger.info(f"  Vision constraints: {query.vision_constraints}")
            logger.info(f"  Text constraints: {query.text_constraints}")
            
            # Perform search
            results = integration_layer.search(query)
            
            logger.info(f"  Found {len(results)} matching frames:")
            for i, result in enumerate(results):
                logger.info(f"    {i+1}. Frame: {result.frame_analysis.frame_id}")
                logger.info(f"       Similarity score: {result.similarity_score:.3f}")
                logger.info(f"       Matched constraints: {result.matched_constraints}")
                logger.info(f"       Explanation: {result.explanation}")
                
        except Exception as e:
            logger.error(f"  Error searching for '{query_text}': {e}")


async def demo_custom_queries():
    """Demonstrate custom query creation and advanced search"""
    logger.info("\n=== Custom Query Demo ===")
    
    # Create integration layer
    integration_layer = await create_integration_layer(device="cpu")
    
    # Add some demo frames
    demo_frames = [
        ("traffic_scene", create_demo_image_with_text("DANGER ZONE", ["car", "person"])),
        ("parking_lot", create_demo_image_with_text("PARKING EXIT", ["car"])),
        ("building_entrance", create_demo_image_with_text("BUILDING ENTRANCE", ["person"])),
    ]
    
    for frame_id, frame in demo_frames:
        await integration_layer.analyze_frame(frame, frame_id)
    
    # Create custom queries
    custom_queries = [
        MultiModalQuery(
            query_text="dangerous traffic situations",
            query_type=QueryType.MULTIMODAL_AND,
            vision_constraints=["car", "person"],
            text_constraints=["danger", "zone"],
            similarity_threshold=0.3
        ),
        MultiModalQuery(
            query_text="parking areas",
            query_type=QueryType.MULTIMODAL_OR,
            vision_constraints=["car"],
            text_constraints=["parking", "exit"],
            similarity_threshold=0.4
        ),
        MultiModalQuery(
            query_text="people near buildings",
            query_type=QueryType.SEMANTIC_SEARCH,
            similarity_threshold=0.2
        )
    ]
    
    for query in custom_queries:
        logger.info(f"\nCustom query: '{query.query_text}'")
        results = integration_layer.search(query)
        
        logger.info(f"  Results: {len(results)} frames")
        for result in results:
            logger.info(f"    - {result.frame_analysis.frame_id}: {result.similarity_score:.3f}")


async def demo_video_analysis():
    """Demonstrate video analysis (simulated with image sequence)"""
    logger.info("\n=== Video Analysis Demo (Simulated) ===")
    
    # Create integration layer
    integration_layer = await create_integration_layer(device="cpu")
    
    # Simulate video frames
    video_frames = []
    scenarios = [
        ("Normal traffic", create_demo_image_with_text("SPEED LIMIT 50", ["car"])),
        ("Emergency vehicle", create_demo_image_with_text("EMERGENCY VEHICLE", ["car", "person"])),
        ("Construction zone", create_demo_image_with_text("CONSTRUCTION AHEAD", ["person"])),
        ("School zone", create_demo_image_with_text("SCHOOL ZONE", ["person"])),
        ("Exit ramp", create_demo_image_with_text("EXIT 42", ["car"])),
    ]
    
    # Analyze "video" frames
    logger.info("Analyzing simulated video frames...")
    for i, (scene_name, frame) in enumerate(scenarios):
        frame_id = f"video_frame_{i:03d}_{scene_name.replace(' ', '_').lower()}"
        timestamp = i * 2.0  # 2 seconds per frame
        
        analysis = await integration_layer.analyze_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp
        )
        
        logger.info(f"  Frame {i}: {scene_name} (t={timestamp}s)")
    
    # Search across all video frames
    video_queries = [
        "show all frames with emergency vehicles",
        "find construction zones with people",
        "locate exit ramps with cars"
    ]
    
    for query_text in video_queries:
        logger.info(f"\nVideo search: '{query_text}'")
        query = parse_natural_query(query_text)
        results = integration_layer.search(query)
        
        for result in results:
            timestamp = result.frame_analysis.timestamp
            logger.info(f"  Found at t={timestamp:.1f}s: {result.frame_analysis.frame_id}")


async def demo_similarity_comparison():
    """Demonstrate embedding similarity comparison"""
    logger.info("\n=== Embedding Similarity Demo ===")
    
    # Create integration layer
    integration_layer = await create_integration_layer(device="cpu")
    
    # Create reference image
    ref_image = create_demo_image_with_text("EXIT SIGN", ["car"])
    ref_analysis = await integration_layer.analyze_frame(ref_image, "reference")
    
    # Create comparison images
    test_images = [
        ("similar_exit", create_demo_image_with_text("EXIT DOOR", ["car"])),
        ("different_sign", create_demo_image_with_text("PARKING SIGN", ["car"])),
        ("similar_vehicle", create_demo_image_with_text("VEHICLE EXIT", ["car"])),
        ("no_match", create_demo_image_with_text("WELCOME", ["person"]))
    ]
    
    logger.info("Comparing images to reference 'EXIT SIGN with car':")
    
    for test_name, test_image in test_images:
        test_analysis = await integration_layer.analyze_frame(test_image, test_name)
        
        # Calculate vision similarity
        if ref_analysis.vision_embedding is not None and test_analysis.vision_embedding is not None:
            vision_sim = integration_layer.embedding_engine.calculate_similarity(
                ref_analysis.vision_embedding,
                test_analysis.vision_embedding
            )
        else:
            vision_sim = 0.0
        
        # Calculate text similarity
        if ref_analysis.text_embedding is not None and test_analysis.text_embedding is not None:
            text_sim = integration_layer.embedding_engine.calculate_similarity(
                ref_analysis.text_embedding, 
                test_analysis.text_embedding
            )
        else:
            text_sim = 0.0
        
        logger.info(f"  {test_name}:")
        logger.info(f"    Vision similarity: {vision_sim:.3f}")
        logger.info(f"    Text similarity: {text_sim:.3f}")
        logger.info(f"    Combined score: {(vision_sim + text_sim) / 2:.3f}")


async def main():
    """Run all demo functions"""
    logger.info("Starting Multi-modal Integration Layer Demo")
    logger.info("=" * 50)
    
    try:
        # Basic analysis
        integration_layer = await demo_basic_analysis()
        
        # Multi-modal search
        await demo_multimodal_search(integration_layer)
        
        # Custom queries
        await demo_custom_queries()
        
        # Video analysis
        await demo_video_analysis()
        
        # Similarity comparison
        await demo_similarity_comparison()
        
        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
