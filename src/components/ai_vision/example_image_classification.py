#!/usr/bin/env python3
# src/components/ai_vision/example_image_classification.py

"""
Image Classification Module Example

This example demonstrates how to use the image classification module for:
- Single image classification with ResNet50 and MobileNetV3
- Batch image processing
- Top-k predictions with probabilities
- Fine-tuning on custom datasets
- Using different hooks for training callbacks
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    # Try importing PIL for creating test images
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    print("PIL not available. Install with: pip install Pillow")
    PIL_AVAILABLE = False

try:
    # Import the image classification module
    from image_classification import (
        ImageClassifier,
        create_classifier,
        classify_image,
        classify_images_batch,
        LoggingHook,
        CheckpointHook,
        ClassificationModel,
        TopKResult
    )
except ImportError as e:
    print(f"Error importing image classification module: {e}")
    print("Make sure you're running from the correct directory and have dependencies installed.")
    exit(1)


def create_test_image(size=(224, 224), color='red') -> Image.Image:
    """Create a simple test image"""
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required to create test images")
    
    # Create a colored image with some patterns
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Add some simple patterns to make it more interesting
    draw.rectangle([50, 50, 174, 174], outline='white', width=3)
    draw.ellipse([75, 75, 149, 149], fill='white')
    
    return img


def create_test_images(count=5) -> List[Image.Image]:
    """Create multiple test images with different colors"""
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    return [create_test_image(color=colors[i % len(colors)]) for i in range(count)]


async def demonstrate_basic_classification():
    """Demonstrate basic image classification"""
    print("\n" + "="*60)
    print("1. BASIC IMAGE CLASSIFICATION")
    print("="*60)
    
    # Create test image
    if not PIL_AVAILABLE:
        print("Skipping basic classification demo - PIL not available")
        return
    
    test_image = create_test_image(color='blue')
    
    try:
        # Create and initialize classifier with ResNet50
        print("Creating ResNet50 classifier...")
        classifier = await create_classifier(model_name="resnet50", device="auto")
        
        # Classify single image
        print("Classifying test image...")
        result = classifier.classify_image(test_image, k=5)
        
        print(f"\nClassification Results (Top-{result.k}):")
        print(f"Processing time: {result.processing_time:.4f}s")
        
        for i, pred in enumerate(result.predictions):
            print(f"{i+1}. {pred.class_name} (ID: {pred.class_id}) - Confidence: {pred.confidence:.4f}")
        
    except Exception as e:
        print(f"Error in basic classification: {e}")


async def demonstrate_batch_classification():
    """Demonstrate batch image classification"""
    print("\n" + "="*60)
    print("2. BATCH IMAGE CLASSIFICATION")
    print("="*60)
    
    if not PIL_AVAILABLE:
        print("Skipping batch classification demo - PIL not available")
        return
    
    # Create multiple test images
    test_images = create_test_images(3)
    
    try:
        # Create classifier with MobileNetV3
        print("Creating MobileNetV3 Large classifier...")
        classifier = await create_classifier(model_name="mobilenet_v3_large", device="auto")
        
        # Classify batch of images
        print(f"Classifying batch of {len(test_images)} images...")
        batch_result = classifier.classify_images_batch(test_images, k=3)
        
        print(f"\nBatch Classification Results:")
        print(f"Batch size: {batch_result.batch_size}")
        print(f"Total processing time: {batch_result.processing_time:.4f}s")
        print(f"Average time per image: {batch_result.processing_time/batch_result.batch_size:.4f}s")
        
        for i, result in enumerate(batch_result.results):
            print(f"\n--- Image {i+1} Results (Top-{result.k}) ---")
            for j, pred in enumerate(result.predictions):
                print(f"  {j+1}. {pred.class_name} - Confidence: {pred.confidence:.4f}")
        
    except Exception as e:
        print(f"Error in batch classification: {e}")


async def demonstrate_different_models():
    """Demonstrate classification with different model architectures"""
    print("\n" + "="*60)
    print("3. DIFFERENT MODEL ARCHITECTURES")
    print("="*60)
    
    if not PIL_AVAILABLE:
        print("Skipping model comparison demo - PIL not available")
        return
    
    test_image = create_test_image(color='green')
    models = ["resnet50", "mobilenet_v3_large", "mobilenet_v3_small"]
    
    results = {}
    
    for model_name in models:
        try:
            print(f"\nTesting {model_name}...")
            classifier = await create_classifier(model_name=model_name, device="auto")
            
            start_time = time.time()
            result = classifier.classify_image(test_image, k=3)
            
            results[model_name] = {
                'processing_time': result.processing_time,
                'top_prediction': result.predictions[0] if result.predictions else None
            }
            
            print(f"Processing time: {result.processing_time:.4f}s")
            if result.predictions:
                top_pred = result.predictions[0]
                print(f"Top prediction: {top_pred.class_name} ({top_pred.confidence:.4f})")
        
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Compare results
    print(f"\n--- Model Comparison ---")
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name}: Error - {result['error']}")
        else:
            print(f"{model_name}: {result['processing_time']:.4f}s - {result['top_prediction'].class_name}")


def demonstrate_convenience_functions():
    """Demonstrate convenience functions"""
    print("\n" + "="*60)
    print("4. CONVENIENCE FUNCTIONS")
    print("="*60)
    
    if not PIL_AVAILABLE:
        print("Skipping convenience functions demo - PIL not available")
        return
    
    # Note: These functions require a pre-loaded model, which would typically
    # come from a global classifier or model hub. For demonstration purposes,
    # we'll show how they would be used.
    
    print("Convenience functions usage examples:")
    print("\n1. classify_image(img, model, k=5)")
    print("   - For single image classification")
    print("   - Requires pre-loaded model")
    
    print("\n2. classify_images_batch(imgs, model, k=5)")
    print("   - For batch image classification")
    print("   - Requires pre-loaded model")
    
    print("\nThese functions are typically used with pre-loaded models from")
    print("a model hub or global classifier instance.")


def create_dummy_dataset():
    """Create a dummy dataset for fine-tuning demonstration"""
    if not PIL_AVAILABLE:
        return None, None
    
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision import transforms
        
        # Create dummy data (normally you'd load real data)
        # 32 samples, 3 channels, 224x224 images
        dummy_images = torch.randn(32, 3, 224, 224)
        dummy_labels = torch.randint(0, 10, (32,))  # 10 classes
        
        # Create datasets
        train_dataset = TensorDataset(dummy_images[:24], dummy_labels[:24])
        val_dataset = TensorDataset(dummy_images[24:], dummy_labels[24:])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        return train_loader, val_loader
        
    except ImportError:
        print("PyTorch not available for dataset creation")
        return None, None


async def demonstrate_fine_tuning():
    """Demonstrate model fine-tuning with hooks"""
    print("\n" + "="*60)
    print("5. FINE-TUNING WITH HOOKS")
    print("="*60)
    
    # Create dummy dataset
    train_loader, val_loader = create_dummy_dataset()
    
    if train_loader is None:
        print("Skipping fine-tuning demo - PyTorch dataset creation failed")
        return
    
    try:
        # Create classifier with custom number of classes
        print("Creating classifier for fine-tuning (10 classes)...")
        classifier = ImageClassifier(model_name="resnet50", num_classes=10, device="auto")
        await classifier.ensure_model_ready()
        
        # Create training hooks
        hooks = [
            LoggingHook(log_every_n_batches=2),  # Log every 2 batches
            # CheckpointHook("./checkpoints", save_every_n_epochs=2)  # Uncomment to save checkpoints
        ]
        
        print("\nStarting fine-tuning (2 epochs for demo)...")
        
        # Fine-tune the model
        metrics = classifier.fine_tune(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            learning_rate=0.001,
            hooks=hooks
        )
        
        print(f"\n--- Fine-tuning Results ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Save the fine-tuned model
        model_path = "fine_tuned_model.pth"
        classifier.save_model(model_path)
        print(f"\nFine-tuned model saved to: {model_path}")
        
    except Exception as e:
        print(f"Error in fine-tuning demonstration: {e}")


def demonstrate_classification_models():
    """Demonstrate the ClassificationModel enum"""
    print("\n" + "="*60)
    print("6. SUPPORTED CLASSIFICATION MODELS")
    print("="*60)
    
    print("Available model architectures:")
    for model in ClassificationModel:
        print(f"- {model.value}")
    
    print("\nUsage examples:")
    print(f"classifier = ImageClassifier(model_name='{ClassificationModel.RESNET50.value}')")
    print(f"classifier = ImageClassifier(model_name='{ClassificationModel.MOBILENET_V3_LARGE.value}')")
    print(f"classifier = ImageClassifier(model_name='{ClassificationModel.MOBILENET_V3_SMALL.value}')")


async def main():
    """Run all demonstrations"""
    print("Image Classification Module Demonstration")
    print("=========================================")
    
    # Check dependencies
    missing_deps = []
    try:
        import torch
        import torchvision
    except ImportError:
        missing_deps.extend(["torch", "torchvision"])
    
    if not PIL_AVAILABLE:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them using:")
        print("pip install torch torchvision Pillow")
        return
    
    # Run demonstrations
    try:
        await demonstrate_basic_classification()
        await demonstrate_batch_classification()
        await demonstrate_different_models()
        demonstrate_convenience_functions()
        await demonstrate_fine_tuning()
        demonstrate_classification_models()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED")
        print("="*60)
        print("\nKey features demonstrated:")
        print("✓ Single image classification with top-k predictions")
        print("✓ Batch image processing")
        print("✓ Multiple model architectures (ResNet50, MobileNetV3)")
        print("✓ Convenience functions")
        print("✓ Fine-tuning with custom hooks")
        print("✓ Model saving and loading")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
