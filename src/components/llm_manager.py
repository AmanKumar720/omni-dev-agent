import os
import logging
from typing import Optional, Dict, Any, List

# Conditional imports for Gemini and Ollama
try:
    import google.generativeai as genai
    _has_gemini = True
except ImportError:
    _has_gemini = False
    logging.warning("google-generativeai not installed. Gemini functionality will be unavailable.")

try:
    import ollama
    _has_ollama = True
except ImportError:
    _has_ollama = False
    logging.warning("ollama not installed. Local LLM (Ollama) functionality will be unavailable.")

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GeminiClient:
    """Client for interacting with the Google Gemini API."""
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not _has_gemini:
            raise RuntimeError("google-generativeai package is not installed.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        logger.info(f"GeminiClient initialized with model: {self.model_name}")

    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generates content using the Gemini model."""
        try:
            response = await self.model.generate_content_async(prompt, **kwargs)
            return response.text
        except Exception as e:
            logger.error(f"Gemini content generation failed: {e}")
            raise

class OllamaClient:
    """Client for interacting with a local Ollama LLM."""
    def __init__(self, host: str = "http://localhost:11434", model_name: str = "mistral"):
        if not _has_ollama:
            raise RuntimeError("ollama package is not installed.")
        self.client = ollama.Client(host=host)
        self.model_name = model_name
        logger.info(f"OllamaClient initialized with model: {self.model_name} at {host}")

    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generates content using the local Ollama model."""
        try:
            response = await self.client.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], **kwargs)
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama content generation failed: {e}")
            raise

class LLMManager:
    """
    Manages interactions with multiple LLMs (Gemini and local Ollama)
    and routes requests based on specified preferences or capabilities.
    """
    def __init__(self, gemini_api_key: Optional[str] = None,
                 gemini_model_name: str = "gemini-1.5-flash",
                 ollama_host: str = "http://localhost:11434",
                 ollama_model_name: str = "mistral"):
        self.gemini_client: Optional[GeminiClient] = None
        self.ollama_client: Optional[OllamaClient] = None

        if gemini_api_key and _has_gemini:
            try:
                self.gemini_client = GeminiClient(api_key=gemini_api_key, model_name=gemini_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize GeminiClient: {e}")

        if _has_ollama:
            try:
                self.ollama_client = OllamaClient(host=ollama_host, model_name=ollama_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize OllamaClient: {e}")

        if not self.gemini_client and not self.ollama_client:
            logger.error("No LLM clients could be initialized. Please check API keys and Ollama server status.")
            raise RuntimeError("No LLM clients available.")

        logger.info("LLMManager initialized.")

    async def generate_content(self, prompt: str,
                               use_gemini: bool = True,
                               use_ollama: bool = False,
                               **kwargs) -> str:
        """
        Generates content using the specified LLM.
        Prioritizes Gemini if use_gemini is True and client is available.
        Falls back to Ollama if use_gemini is False or Gemini is unavailable and use_ollama is True.
        """
        if use_gemini and self.gemini_client:
            logger.debug("Using Gemini for content generation.")
            return await self.gemini_client.generate_content(prompt, **kwargs)
        elif use_ollama and self.ollama_client:
            logger.debug("Using Ollama for content generation.")
            return await self.ollama_client.generate_content(prompt, **kwargs)
        else:
            raise ValueError("No suitable LLM client available for the request.")

    def get_available_models(self) -> List[str]:
        """Returns a list of available LLM models."""
        models = []
        if self.gemini_client:
            models.append(f"Gemini ({self.gemini_client.model_name})")
        if self.ollama_client:
            models.append(f"Ollama ({self.ollama_client.model_name})")
        return models

# Example Usage (for demonstration and testing)
async def main():
    # Replace with your actual Gemini API key or set as environment variable
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Initialize LLMManager
    # It will try to initialize both clients if dependencies are met and API key is provided
    try:
        llm_manager = LLMManager(gemini_api_key=gemini_api_key)
    except RuntimeError as e:
        print(f"Error initializing LLMManager: {e}")
        print("Please ensure you have installed 'google-generativeai' and 'ollama' packages,")
        print("and that your GEMINI_API_KEY is set, and Ollama server is running if you intend to use it.")
        return

    print(f"Available LLMs: {llm_manager.get_available_models()}")

    # --- Demonstrate Gemini Usage ---
    if llm_manager.gemini_client:
        print("\n--- Gemini Example ---")
        try:
            gemini_response = await llm_manager.generate_content(
                "Write a short, creative slogan for a new AI development agent.",
                use_gemini=True
            )
            print(f"Gemini Response: {gemini_response}")
        except Exception as e:
            print(f"Gemini example failed: {e}")
    else:
        print("\n--- Gemini Client not initialized. Skipping Gemini example. ---")

    # --- Demonstrate Ollama Usage ---
    # Ensure Ollama server is running and 'mistral' model is pulled (ollama run mistral)
    if llm_manager.ollama_client:
        print("\n--- Ollama Example ---")
        try:
            ollama_response = await llm_manager.generate_content(
                "Explain the concept of 'free tier' in cloud computing in one sentence.",
                use_ollama=True,
                use_gemini=False # Explicitly tell it to use Ollama
            )
            print(f"Ollama Response: {ollama_response}")
        except Exception as e:
            print(f"Ollama example failed: {e}")
            print("Please ensure your Ollama server is running and the 'mistral' model is available (ollama run mistral).")
    else:
        print("\n--- Ollama Client not initialized. Skipping Ollama example. ---")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())