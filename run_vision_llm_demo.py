import asyncio
import os
import logging
from components.llm_manager import LLMManager
from components.vision_llm_integrator import VisionLLMIntegrator

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_vision_llm_demonstration():
    logger.info("Starting Vision-LLM integration demonstration...")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Gemini functionality may be limited.")

    # Initialize LLMManager
    try:
        llm_manager = LLMManager(gemini_api_key=gemini_api_key)
        logger.info(f"Available LLMs: {llm_manager.get_available_models()}")
    except RuntimeError as e:
        logger.error(f"Error initializing LLMManager: {e}")
        logger.error("Please ensure you have installed 'google-generativeai' and 'ollama' packages,")
        logger.error("and that your GEMINI_API_KEY is set, and Ollama server is running if you intend to use it.")
        return

    # Initialize VisionLLMIntegrator
    integrator = VisionLLMIntegrator(llm_manager=llm_manager)

    # --- Simulate Vision Results ---
    # In a real application, these would come from your actual vision processing pipeline
    sample_vision_data_1 = {
        "objects": [
            {"label": "car", "confidence": 0.98},
            {"label": "traffic light", "confidence": 0.92},
            {"label": "person", "confidence": 0.75}
        ],
        "ocr_text": "STOP",
        "scene_description": "A busy street intersection."
    }

    sample_vision_data_2 = {
        "objects": [
            {"label": "dog", "confidence": 0.90},
            {"label": "ball", "confidence": 0.85}
        ],
        "scene_description": "A park with a dog playing."
    }

    sample_vision_data_3 = {
        "ocr_text": "Invoice #12345\nTotal: $150.00\nDate: 2023-07-26",
        "scene_description": "A scanned document."
    }

    # --- Analyze Vision Results with LLMs ---

    # Example 1: Analyze with Gemini (if available)
    if llm_manager.gemini_client:
        logger.info("\n--- Analyzing Vision Data 1 with Gemini ---")
        try:
            analysis_gemini = await integrator.analyze_vision_results(
                sample_vision_data_1, use_gemini=True, use_ollama=False,
                task_description="Describe the scene and identify any potential hazards based on the detected objects and text."
            )
            logger.info(f"Gemini's Vision Analysis: {analysis_gemini}")
        except Exception as e:
            logger.error(f"Gemini vision analysis failed: {e}")
    else:
        logger.warning("Gemini client not initialized. Skipping Gemini vision analysis.")

    # Example 2: Analyze with Ollama (if available)
    if llm_manager.ollama_client:
        logger.info("\n--- Analyzing Vision Data 2 with Ollama ---")
        try:
            analysis_ollama = await integrator.analyze_vision_results(
                sample_vision_data_2, use_gemini=False, use_ollama=True,
                task_description="Summarize the activity in the scene."
            )
            logger.info(f"Ollama's Vision Analysis: {analysis_ollama}")
        except Exception as e:
            logger.error(f"Ollama vision analysis failed: {e}")
    else:
        logger.warning("Ollama client not initialized. Skipping Ollama vision analysis.")

    # Example 3: Analyze OCR text with Gemini (if available)
    if llm_manager.gemini_client:
        logger.info("\n--- Analyzing Vision Data 3 (OCR) with Gemini ---")
        try:
            analysis_ocr_gemini = await integrator.analyze_vision_results(
                sample_vision_data_3, use_gemini=True, use_ollama=False,
                task_description="Extract key information from this document, such as invoice number, total amount, and date."
            )
            logger.info(f"Gemini's OCR Analysis: {analysis_ocr_gemini}")
        except Exception as e:
            logger.error(f"Gemini OCR analysis failed: {e}")
    else:
        logger.warning("Gemini client not initialized. Skipping Gemini OCR analysis.")

    logger.info("Vision-LLM integration demonstration completed.")

if __name__ == "__main__":
    asyncio.run(run_vision_llm_demonstration())