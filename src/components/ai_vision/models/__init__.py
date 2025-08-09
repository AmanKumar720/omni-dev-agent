# src/components/ai_vision/models/__init__.py

"""
AI Vision Models Module

This module contains various AI vision models and their implementations.
It provides interfaces for different vision tasks such as:
- Object detection
- Image classification  
- Semantic segmentation
- Optical character recognition (OCR)
- Face recognition
- Image generation
"""

from typing import Dict, Any, List

# Model registry for dynamic loading
MODEL_REGISTRY: Dict[str, Any] = {
    # Will be populated by specific model implementations
    'yolov8_object_detection': {
        'class': None,  # Will be lazy-loaded
        'capabilities': ['object_detection', 'real_time_detection', 'batch_processing']
    }
}

def register_model(name: str, model_class: Any, capabilities: List[str]) -> None:
    """
    Register a vision model in the global registry
    
    Args:
        name: Model name/identifier
        model_class: Model class
        capabilities: List of capabilities this model supports
    """
    MODEL_REGISTRY[name] = {
        'class': model_class,
        'capabilities': capabilities
    }

def get_model(name: str) -> Any:
    """
    Get a model from the registry
    
    Args:
        name: Model name
        
    Returns:
        Model class if found, None otherwise
    """
    return MODEL_REGISTRY.get(name, {}).get('class')

def list_models() -> Dict[str, List[str]]:
    """
    List all registered models and their capabilities
    
    Returns:
        Dictionary mapping model names to their capabilities
    """
    return {name: info['capabilities'] for name, info in MODEL_REGISTRY.items()}

__all__ = ['MODEL_REGISTRY', 'register_model', 'get_model', 'list_models']
