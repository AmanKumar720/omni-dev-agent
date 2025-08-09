import asyncio
import os
import logging
from components.llm_manager import LLMManager
from error_handling.llm_overload_handler import LLMOverloadHandler, with_llm_overload_protection

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_llm_demonstration():
    logger.info("Starting LLM demonstration...")

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

    # Initialize LLMOverloadHandler with the LLMManager
    overload_handler = LLMOverloadHandler(llm_manager=llm_manager)

    # --- Demonstrate Gemini Usage ---
    if llm_manager.gemini_client:
        logger.info("\n--- Demonstrating Gemini Usage ---")
        try:
            @with_llm_overload_protection(handler=overload_handler, use_gemini=True)
            async def protected_gemini_call(prompt: str):
                return await llm_manager.generate_content(prompt, use_gemini=True)

            prompt_gemini = "Explain the concept of quantum entanglement in simple terms."
            logger.info(f"Gemini Prompt: {prompt_gemini}")
            response_gemini = await protected_gemini_call(prompt=prompt_gemini)
            logger.info(f"Gemini Response: {response_gemini[:200]}...") # Print first 200 chars
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
    else:
        logger.warning("Gemini client not initialized. Skipping Gemini example.")

    # --- Demonstrate Ollama Usage ---
    if llm_manager.ollama_client:
        logger.info("\n--- Demonstrating Ollama Usage (Mistral) ---")
        try:
            @with_llm_overload_protection(handler=overload_handler, use_ollama=True, use_gemini=False)
            async def protected_ollama_call(prompt: str):
                return await llm_manager.generate_content(prompt, use_ollama=True, use_gemini=False)

            prompt_ollama = "Write a very short, funny haiku about a cat."
            logger.info(f"Ollama Prompt: {prompt_ollama}")
            response_ollama = await protected_ollama_call(prompt=prompt_ollama)
            logger.info(f"Ollama Response: {response_ollama}")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            logger.error("Ensure Ollama server is running and 'mistral' model is pulled (ollama run mistral).")
    else:
        logger.warning("Ollama client not initialized. Skipping Ollama example.")

    logger.info("LLM demonstration completed.")

if __name__ == "__main__":
    asyncio.run(run_llm_demonstration())