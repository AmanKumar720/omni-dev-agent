# Multi-modal Integration Layer

This module provides a comprehensive integration layer that connects vision outputs to NLP pipelines via embeddings using CLIP, sentence-transformers, and other state-of-the-art models. It enables complex multi-modal queries like "show all frames where a red car appears and text contains 'exit'".

## Features

- **Vision-Text Embedding Alignment**: Uses CLIP models to create aligned embeddings for images and text
- **Object Detection Integration**: Seamlessly integrates with YOLO-based object detection
- **OCR Text Extraction**: Extracts and analyzes text from images using Tesseract
- **Multi-modal Search**: Performs complex searches combining visual and textual constraints
- **Natural Language Queries**: Parse natural language queries into structured search operations
- **Video Analysis**: Analyze video frames with temporal context
- **Similarity Matching**: Calculate semantic similarity between images and text
- **Flexible Query Types**: Support for AND, OR, and similarity-based queries

## Architecture

### Core Components

1. **EmbeddingEngine**: Generates and manages embeddings for different modalities
2. **MultiModalIntegrationLayer**: Main orchestration layer connecting all components
3. **MultiModalQuery**: Structured representation of search queries
4. **FrameAnalysis**: Analysis results for individual frames
5. **SearchResult**: Results from multi-modal search operations

### Data Flow

```
Input Frame/Video → Object Detection → OCR → Embedding Generation → Storage
                                                      ↓
Natural Language Query → Query Parsing → Multi-modal Search → Ranked Results
```

## Installation

### Requirements

The multi-modal integration layer requires several ML libraries:

```bash
pip install torch torchvision
pip install transformers sentence-transformers
pip install ultralytics opencv-python
pip install pytesseract pillow numpy
pip install clip-by-openai  # Optional, for OpenAI CLIP
```

### System Dependencies

- **Tesseract OCR**: Required for text extraction
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Quick Start

### Basic Usage

```python
import asyncio
import cv2
from multimodal import create_integration_layer, parse_natural_query

async def main():
    # Create and initialize integration layer
    integration_layer = await create_integration_layer(
        clip_model="ViT-B/32",
        sentence_model="all-MiniLM-L6-v2",
        yolo_model="yolov8n",
        device="auto"  # Uses GPU if available
    )
    
    # Load an image
    image = cv2.imread("path/to/image.jpg")
    
    # Analyze the image
    analysis = await integration_layer.analyze_frame(
        frame=image,
        frame_id="image_001",
        extract_objects=True,
        extract_text=True,
        generate_embeddings=True
    )
    
    print(f"Detected objects: {len(analysis.vision_results)}")
    if analysis.ocr_results:
        print(f"OCR text: {analysis.ocr_results.pages[0].text}")
    
    # Perform a natural language search
    query = parse_natural_query("show all frames where a red car appears and text contains 'exit'")
    results = integration_layer.search(query)
    
    for result in results:
        print(f"Match: {result.frame_analysis.frame_id}")
        print(f"Score: {result.similarity_score:.3f}")
        print(f"Explanation: {result.explanation}")

asyncio.run(main())
```

### Video Analysis

```python
import asyncio
from multimodal import create_integration_layer

async def analyze_video():
    integration_layer = await create_integration_layer()
    
    # Analyze video frames (every 30th frame)
    results = await integration_layer.analyze_video(
        video_path="path/to/video.mp4",
        frame_interval=30,
        max_frames=100
    )
    
    print(f"Analyzed {len(results)} frames")
    
    # Search for specific content
    from multimodal import MultiModalQuery, QueryType
    
    query = MultiModalQuery(
        query_text="emergency vehicle with warning signs",
        query_type=QueryType.MULTIMODAL_AND,
        vision_constraints=["car", "truck"],
        text_constraints=["emergency", "warning"],
        similarity_threshold=0.6
    )
    
    search_results = integration_layer.search(query)
    for result in search_results:
        timestamp = result.frame_analysis.timestamp
        print(f"Found emergency vehicle at {timestamp:.1f}s")

asyncio.run(analyze_video())
```

## API Reference

### MultiModalIntegrationLayer

Main integration class that orchestrates vision and NLP components.

#### Methods

- `async initialize()`: Initialize all components
- `async analyze_frame(frame, frame_id, **kwargs) -> FrameAnalysis`: Analyze a single frame
- `async analyze_video(video_path, frame_interval=30) -> Dict[str, FrameAnalysis]`: Analyze video
- `search(query: MultiModalQuery) -> List[SearchResult]`: Perform multi-modal search
- `get_frame_analysis(frame_id: str) -> FrameAnalysis`: Retrieve analysis by ID
- `clear_database()`: Clear all stored analyses

### EmbeddingEngine

Handles embedding generation for different modalities.

#### Methods

- `async initialize()`: Load all models
- `encode_image(image: np.ndarray) -> EmbeddingResult`: Generate image embedding
- `encode_text(text: str, use_clip=False) -> EmbeddingResult`: Generate text embedding
- `calculate_similarity(embedding1, embedding2) -> float`: Calculate cosine similarity

### MultiModalQuery

Structured representation of search queries.

#### Parameters

- `query_text: str`: Natural language description
- `query_type: QueryType`: Type of query (AND, OR, SIMILARITY)
- `vision_constraints: List[str]`: Required visual elements
- `text_constraints: List[str]`: Required text elements
- `similarity_threshold: float`: Minimum similarity score
- `max_results: int`: Maximum number of results

### Query Types

- `QueryType.MULTIMODAL_AND`: All constraints must be satisfied
- `QueryType.MULTIMODAL_OR`: At least one constraint must be satisfied
- `QueryType.SIMILARITY`: Semantic similarity search
- `QueryType.SEMANTIC_SEARCH`: Advanced semantic matching

## Advanced Usage

### Custom Embedding Models

```python
from multimodal import EmbeddingEngine, MultiModalIntegrationLayer

# Use custom models
embedding_engine = EmbeddingEngine(
    clip_model="ViT-L/14",  # Larger CLIP model
    sentence_model="all-mpnet-base-v2",  # Better sentence model
    device="cuda"
)

integration_layer = MultiModalIntegrationLayer(
    embedding_engine=embedding_engine
)
```

### Batch Processing

```python
import asyncio
import cv2
from multimodal import create_integration_layer

async def batch_process_images(image_paths):
    integration_layer = await create_integration_layer()
    
    analyses = []
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        analysis = await integration_layer.analyze_frame(
            frame=image,
            frame_id=f"batch_image_{i:03d}"
        )
        analyses.append(analysis)
    
    return analyses

# Process multiple images
image_files = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = asyncio.run(batch_process_images(image_files))
```

### Custom Query Parsing

```python
from multimodal import MultiModalQuery, QueryType

def create_traffic_query():
    return MultiModalQuery(
        query_text="dangerous traffic situations",
        query_type=QueryType.MULTIMODAL_AND,
        vision_constraints=["car", "truck", "person"],
        text_constraints=["danger", "caution", "warning"],
        similarity_threshold=0.7,
        max_results=20
    )

query = create_traffic_query()
results = integration_layer.search(query)
```

## Performance Considerations

### Memory Management

- The integration layer caches analyzed frames in memory
- For large datasets, periodically clear the database: `integration_layer.clear_database()`
- Consider processing in batches for videos with many frames

### GPU Acceleration

- Set `device="cuda"` for GPU acceleration (requires CUDA)
- GPU memory requirements:
  - CLIP ViT-B/32: ~400MB
  - YOLO v8n: ~6MB
  - Sentence transformers: ~100MB

### Model Selection Trade-offs

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| CLIP ViT-B/32 | Small | Good | Fast |
| CLIP ViT-B/16 | Medium | Better | Medium |
| CLIP ViT-L/14 | Large | Best | Slow |
| YOLOv8n | Small | Good | Fast |
| YOLOv8s/m/l/x | Medium-XL | Better+ | Medium-Slow |

## Examples

### Traffic Monitoring

```python
import asyncio
from multimodal import create_integration_layer, parse_natural_query

async def traffic_monitoring():
    integration_layer = await create_integration_layer()
    
    # Analyze traffic camera feed
    video_results = await integration_layer.analyze_video(
        video_path="traffic_feed.mp4",
        frame_interval=60  # Every 2 seconds at 30fps
    )
    
    # Search for violations
    queries = [
        "vehicles running red lights",
        "pedestrians in dangerous areas", 
        "emergency vehicles with sirens",
        "accident scenes with damaged cars"
    ]
    
    for query_text in queries:
        query = parse_natural_query(query_text)
        results = integration_layer.search(query)
        
        print(f"\n{query_text}:")
        for result in results:
            timestamp = result.frame_analysis.timestamp
            print(f"  Alert at {timestamp:.1f}s: {result.explanation}")

asyncio.run(traffic_monitoring())
```

### Security Surveillance

```python
async def security_surveillance():
    integration_layer = await create_integration_layer()
    
    # Monitor for security events
    security_queries = [
        MultiModalQuery(
            query_text="unauthorized access",
            query_type=QueryType.MULTIMODAL_AND,
            vision_constraints=["person"],
            text_constraints=["restricted", "authorized personnel only"],
            similarity_threshold=0.6
        ),
        MultiModalQuery(
            query_text="emergency situations",
            query_type=QueryType.MULTIMODAL_OR,
            vision_constraints=["fire", "smoke"],
            text_constraints=["emergency", "evacuation", "alarm"],
            similarity_threshold=0.7
        )
    ]
    
    for query in security_queries:
        results = integration_layer.search(query)
        if results:
            print(f"Security alert: {query.query_text}")
            for result in results:
                print(f"  {result.frame_analysis.frame_id}: {result.similarity_score:.3f}")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest src/components/multimodal/test_integration_layer.py -v

# Run specific test class
python -m pytest src/components/multimodal/test_integration_layer.py::TestEmbeddingEngine -v

# Run performance tests
python -m pytest src/components/multimodal/test_integration_layer.py -m performance -v

# Skip slow tests
python -m pytest src/components/multimodal/test_integration_layer.py -m "not slow" -v
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'clip'**
   - Install CLIP: `pip install clip-by-openai`
   - Or use transformers CLIP (already supported)

2. **CUDA out of memory**
   - Use smaller models: `clip_model="ViT-B/32"`, `yolo_model="yolov8n"`
   - Set `device="cpu"` to use CPU only
   - Process frames in smaller batches

3. **Tesseract not found**
   - Install system Tesseract OCR package
   - Set TESSDATA_PREFIX environment variable if needed

4. **Slow performance**
   - Enable GPU acceleration: `device="cuda"`
   - Use smaller models for real-time applications
   - Reduce frame analysis frequency for videos

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI for CLIP models
- Hugging Face for transformers and sentence-transformers
- Ultralytics for YOLO models
- Google for Tesseract OCR
