# Image Classification Module

A comprehensive image classification module using TorchVision pretrained CNNs with support for ResNet50 and MobileNetV3 architectures, top-k predictions, and fine-tuning capabilities.

## Features

- **Multiple Model Architectures**: ResNet50, MobileNetV3 Large, MobileNetV3 Small
- **Top-k Predictions**: Get multiple predictions with confidence scores
- **Batch Processing**: Efficient batch image classification
- **Fine-tuning Support**: Fine-tune models on custom datasets with hooks
- **GPU Acceleration**: Automatic GPU detection and usage
- **Flexible Input**: Support for PIL Images and numpy arrays
- **Async Support**: Asynchronous model loading and operations

## Installation

Install the required dependencies:

```bash
pip install -r requirements_image_classification.txt
```

Core dependencies:
- PyTorch (torch >= 1.8.0)
- TorchVision (torchvision >= 0.9.0)
- Pillow (>= 8.0.0)
- NumPy (>= 1.21.0)

## Quick Start

### Basic Usage

```python
import asyncio
from image_classification import create_classifier

async def classify_example():
    # Create and initialize classifier
    classifier = await create_classifier(model_name="resnet50", device="auto")
    
    # Load an image
    from PIL import Image
    image = Image.open("path/to/your/image.jpg")
    
    # Classify with top-5 predictions
    result = classifier.classify_image(image, k=5)
    
    print(f"Processing time: {result.processing_time:.4f}s")
    for i, pred in enumerate(result.predictions):
        print(f"{i+1}. {pred.class_name} - Confidence: {pred.confidence:.4f}")

# Run the example
asyncio.run(classify_example())
```

### Batch Processing

```python
import asyncio
from image_classification import create_classifier
from PIL import Image

async def batch_classify_example():
    classifier = await create_classifier(model_name="mobilenet_v3_large")
    
    # Load multiple images
    images = [
        Image.open(f"image_{i}.jpg") 
        for i in range(5)
    ]
    
    # Classify batch
    batch_result = classifier.classify_images_batch(images, k=3)
    
    print(f"Batch size: {batch_result.batch_size}")
    print(f"Total time: {batch_result.processing_time:.4f}s")
    
    for i, result in enumerate(batch_result.results):
        print(f"\nImage {i+1} predictions:")
        for pred in result.predictions:
            print(f"  {pred.class_name}: {pred.confidence:.4f}")

asyncio.run(batch_classify_example())
```

### Using Convenience Functions

```python
from image_classification import classify_image, create_classifier
import asyncio

async def convenience_example():
    # Create classifier and get the model
    classifier = await create_classifier("resnet50")
    
    # Use convenience function
    from PIL import Image
    image = Image.open("test_image.jpg")
    
    result = classify_image(image, model=classifier.model, k=3)
    
    for pred in result.predictions:
        print(f"{pred.class_name}: {pred.confidence:.4f}")

asyncio.run(convenience_example())
```

## Fine-tuning

### Basic Fine-tuning

```python
import asyncio
import torch
from torch.utils.data import DataLoader
from image_classification import ImageClassifier, LoggingHook, CheckpointHook

async def fine_tuning_example():
    # Create classifier for custom dataset (10 classes)
    classifier = ImageClassifier(
        model_name="resnet50", 
        num_classes=10, 
        device="auto"
    )
    await classifier.ensure_model_ready()
    
    # Prepare your data loaders (replace with your actual data)
    # train_loader = DataLoader(your_train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(your_val_dataset, batch_size=32, shuffle=False)
    
    # Create training hooks
    hooks = [
        LoggingHook(log_every_n_batches=10),
        CheckpointHook("./checkpoints", save_every_n_epochs=5)
    ]
    
    # Fine-tune the model
    metrics = classifier.fine_tune(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        learning_rate=0.001,
        hooks=hooks
    )
    
    print("Training completed!")
    print(f"Final training loss: {metrics['final_train_loss']:.4f}")
    print(f"Final validation loss: {metrics['final_val_loss']:.4f}")
    
    # Save the fine-tuned model
    classifier.save_model("fine_tuned_resnet50.pth")

# asyncio.run(fine_tuning_example())
```

### Custom Training Hooks

```python
from image_classification import FineTuningHook

class CustomHook(FineTuningHook):
    def __init__(self):
        self.best_val_loss = float('inf')
    
    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"New best validation loss: {val_loss:.4f}")
            # Save best model
            torch.save(model.state_dict(), f"best_model_epoch_{epoch}.pth")
    
    def on_training_end(self, model, final_metrics):
        print(f"Training finished. Best validation loss: {self.best_val_loss:.4f}")

# Use the custom hook
hooks = [LoggingHook(), CustomHook()]
```

## API Reference

### Core Classes

#### `ImageClassifier`

Main classification class with full functionality.

```python
classifier = ImageClassifier(
    model_name="resnet50",        # Model architecture
    device="auto",                # Device: "cpu", "cuda", or "auto"
    num_classes=1000             # Number of output classes
)
```

**Methods:**
- `ensure_model_ready() -> bool`: Load and initialize the model
- `classify_image(image, k=5) -> TopKResult`: Classify single image
- `classify_images_batch(images, k=5) -> BatchClassificationResult`: Classify batch
- `fine_tune(train_loader, val_loader=None, **kwargs) -> Dict`: Fine-tune model
- `save_model(path)`: Save model state
- `load_model(path)`: Load model state

#### `ImageClassificationTask`

Vision task implementation for use with the AI Vision Agent framework.

```python
task = ImageClassificationTask(task_id="classify_1", model_name="resnet50")
result = await task.execute(input_data)
```

### Data Structures

#### `ClassificationResult`
```python
@dataclass
class ClassificationResult:
    class_id: int
    class_name: str
    confidence: float
```

#### `TopKResult`
```python
@dataclass
class TopKResult:
    predictions: List[ClassificationResult]
    processing_time: float
    k: int
```

#### `BatchClassificationResult`
```python
@dataclass
class BatchClassificationResult:
    results: List[TopKResult]
    processing_time: float
    batch_size: int
```

### Fine-tuning Hooks

#### `FineTuningHook` (Base Class)
- `on_training_start(model, optimizer, train_loader, val_loader=None)`
- `on_epoch_start(epoch, model, optimizer)`
- `on_batch_start(batch_idx, batch, model)`
- `on_batch_end(batch_idx, batch, outputs, loss, model)`
- `on_epoch_end(epoch, train_loss, val_loss, model)`
- `on_training_end(model, final_metrics)`

#### `LoggingHook`
```python
hook = LoggingHook(log_every_n_batches=100)
```

#### `CheckpointHook`
```python
hook = CheckpointHook("./checkpoints", save_every_n_epochs=5)
```

### Convenience Functions

```python
# Create and initialize classifier
classifier = await create_classifier(model_name="resnet50", device="auto")

# Single image classification
result = classify_image(image, model=loaded_model, k=5)

# Batch classification  
batch_result = classify_images_batch(images, model=loaded_model, k=5)
```

### Supported Models

- `"resnet50"`: ResNet-50 architecture
- `"mobilenet_v3_large"`: MobileNetV3 Large
- `"mobilenet_v3_small"`: MobileNetV3 Small

## Examples

Run the comprehensive example:

```bash
python example_image_classification.py
```

Run basic tests:

```bash
python test_image_classification_basic.py
```

## Error Handling

The module includes comprehensive error handling:

```python
try:
    classifier = await create_classifier("resnet50")
    result = classifier.classify_image(image)
except ImportError as e:
    print(f"Missing dependencies: {e}")
except RuntimeError as e:
    print(f"Model loading failed: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Performance Tips

1. **Batch Processing**: Use `classify_images_batch()` for multiple images
2. **GPU Acceleration**: Set `device="cuda"` if GPU is available
3. **Model Selection**: MobileNetV3 is faster, ResNet50 is more accurate
4. **Image Preprocessing**: Images are automatically resized to 224x224

## Integration with AI Vision Agent

The module integrates seamlessly with the AI Vision Agent framework:

```python
from ai_vision import AIVisionAgent
from image_classification import ImageClassificationTask

# Create agent and register classification task
agent = AIVisionAgent("vision_agent", "Main Vision Agent")
task = ImageClassificationTask("classify_task", model_name="resnet50")
agent.register_task(task)

# Execute task
result = await agent.execute_task("classify_task", {"image": your_image, "k": 5})
```

## Contributing

When extending the module:

1. Follow the existing code patterns
2. Add comprehensive error handling
3. Include type hints
4. Write unit tests
5. Update documentation

## License

This module is part of the AI Vision Agent framework. See the main project for license information.
