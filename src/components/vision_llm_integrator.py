import logging
from typing import List, Dict, Any, Optional

from components.llm_manager import LLMManager
# Assuming your vision components (e.g., ObjectDetector, ImageClassifier) output structured data
# For demonstration, we'll use a simplified structure.

logger = logging.getLogger(__name__)

class VisionLLMIntegrator:
    """
    Integrates vision capabilities with LLMs to generate insights from visual data.
    """
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        logger.info("VisionLLMIntegrator initialized.")

    async def analyze_vision_results(
        self,
        vision_results: Dict[str, Any],
        use_gemini: bool = True,
        use_ollama: bool = False,
        task_description: str = "Analyze the visual information and provide a summary or insights."
    ) -> str:
        """
        Analyzes structured vision results using an LLM.

        Args:
            vision_results: A dictionary containing structured vision data (e.g., object detections, OCR text).
                            Example: {"objects": [{"label": "car", "confidence": 0.95}, {"label": "person", "confidence": 0.8}], "ocr_text": "Hello World"}
            use_gemini: Whether to use Gemini for the analysis.
            use_ollama: Whether to use Ollama for the analysis.
            task_description: A description of the task for the LLM.

        Returns:
            The LLM's generated analysis or insights.
        """
        logger.info("Analyzing vision results with LLM...")

        # Construct a natural language prompt from vision results
        prompt_parts = [task_description]
        
        if "objects" in vision_results and vision_results["objects"]:
            objects_str = ", ".join([f"{obj['label']} (confidence: {obj['confidence']:.2f})" for obj in vision_results["objects"]])
            prompt_parts.append(f"Detected objects: {objects_str}.")
        
        if "ocr_text" in vision_results and vision_results["ocr_text"]:
            prompt_parts.append(f"Extracted text: \"{vision_results[\"ocr_text\"]}\".")

        if "scene_description" in vision_results and vision_results["scene_description"]:
            prompt_parts.append(f"Scene description: {vision_results[\"scene_description"]}.")

        full_prompt = "\n".join(prompt_parts)
        logger.debug(f"Full prompt for LLM: {full_prompt}")

        try:
            llm_response = await self.llm_manager.generate_content(
                prompt=full_prompt,
                use_gemini=use_gemini,
                use_ollama=use_ollama
            )
            logger.info("LLM analysis completed.")
            return llm_response
        except Exception as e:
            logger.error(f"Failed to get LLM analysis for vision results: {e}")
            raise

# Example Usage (for demonstration)
async def main():
    # This is a simplified example. In a real scenario, you'd get these from your vision pipeline.
    sample_vision_data = {
        "objects": [
            {"label": "car", "confidence": 0.98},
            {"label": "traffic light", "confidence": 0.92},
            {"label": "person", "confidence": 0.75}
        ],
        "ocr_text": "STOP",
        "scene_description": "A street intersection with vehicles and pedestrians."
    }

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    try:
        llm_manager = LLMManager(gemini_api_key=gemini_api_key)
        integrator = VisionLLMIntegrator(llm_manager=llm_manager)

        print("\n--- Vision Analysis with Gemini (if available) ---")
        if llm_manager.gemini_client:
            analysis_gemini = await integrator.analyze_vision_results(
                sample_vision_data, use_gemini=True, use_ollama=False,
                task_description="Describe the scene and identify any potential hazards based on the detected objects and text."
            )
            print(f"Gemini's Vision Analysis: {analysis_gemini}")
        else:
            print("Gemini client not initialized. Skipping Gemini vision analysis.")

        print("\n--- Vision Analysis with Ollama (if available) ---")
        if llm_manager.ollama_client:
            analysis_ollama = await integrator.analyze_vision_results(
                sample_vision_data, use_gemini=False, use_ollama=True,
                task_description="Summarize the key elements in the scene and their implications."
            )
            print(f"Ollama's Vision Analysis: {analysis_ollama}")
        else:
            print("Ollama client not initialized. Skipping Ollama vision analysis.")

    except Exception as e:
        print(f"Error during Vision-LLM integration demo: {e}")

if __name__ == "__main__":
    import os
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())