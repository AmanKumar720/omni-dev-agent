# src/components/component_registry/registry.py

COMPONENT_REGISTRY = {
    "Supabase": {
        "description": "Open-source Firebase alternative providing database, auth, and real-time.",
        "version": "latest",
        "capabilities": ["database", "authentication", "real-time", "storage"],
        "dependencies": ["supabase-py"],
        "documentation_url": "https://supabase.com/docs",
    },
    "Celery": {
        "description": "Asynchronous task queue/job queue based on distributed message passing.",
        "version": "latest",
        "capabilities": ["asynchronous_tasks", "background_processing", "scheduling"],
        "dependencies": ["celery", "redis"],  # Example dependency
        "documentation_url": "https://docs.celeryq.dev/en/stable/",
    },
    "boto3": {
        "description": "AWS SDK for Python, allows interaction with AWS services.",
        "version": "latest",
        "capabilities": ["aws_api_interaction", "cloud_resource_management"],
        "dependencies": [],
        "documentation_url": "https://boto3.amazonaws.com/v1/documentation/api/latest/index.html",
    },
    "Flask": {
        "description": "A lightweight WSGI web application framework.",
        "version": "latest",
        "capabilities": ["web_development", "api_creation"],
        "dependencies": [],
        "documentation_url": "https://flask.palletsprojects.com/",
    },
    "AIVision": {
        "description": "AI Vision component for computer vision and image processing tasks.",
        "version": "1.0.0",
        "capabilities": [
            "object_detection",
            "image_classification",
            "semantic_segmentation", 
            "ocr",
            "face_recognition",
            "image_enhancement",
            "video_processing",
            "pipeline_processing"
        ],
        "dependencies": ["opencv-python", "pillow", "numpy"],
        "documentation_url": "https://docs.opencv.org/",
        "component_path": "src.components.ai_vision",
        "main_class": "AIVisionAgent",
        "hot_pluggable": True
    },
    # Add more components as needed
}


def get_component_metadata(component_name: str) -> dict | None:
    """
    Retrieves metadata for a component from the registry.

    Args:
        component_name (str): The name of the component.

    Returns:
        dict | None: The component's metadata if found, otherwise None.
    """
    return COMPONENT_REGISTRY.get(component_name)


if __name__ == "__main__":
    print("Supabase Metadata:", get_component_metadata("Supabase"))
    print("NonExistent Metadata:", get_component_metadata("NonExistent"))
