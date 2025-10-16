"""LLM provider abstraction layer supporting OpenAI and Ollama."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response and metadata
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., gpt-3.5-turbo, gpt-4)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized OpenAI provider with model: {model_name}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "content": response.choices[0].message.content,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }


class OllamaProvider(LLMProvider):
    """Ollama LLM provider (local)."""

    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.

        Args:
            model_name: Ollama model name (e.g., llama3:8b, qwen2.5:7b)
            base_url: Ollama server URL
        """
        # Use OpenAI client with Ollama's OpenAI-compatible endpoint
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama"  # Ollama doesn't need a real API key
        )
        self.model_name = model_name
        self.base_url = base_url
        logger.info(f"Initialized Ollama provider with model: {model_name} at {base_url}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate response using Ollama."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return {
                "content": response.choices[0].message.content,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            }
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to generate with Ollama. Make sure Ollama is running and "
                f"the model '{self.model_name}' is installed. Error: {str(e)}"
            )


def create_llm_provider(
    provider_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider_type: Type of provider ('openai' or 'ollama')
        model_name: Model name
        api_key: API key (for OpenAI)
        base_url: Base URL (for Ollama)

    Returns:
        LLMProvider instance
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires an API key")
        return OpenAIProvider(api_key=api_key, model_name=model_name)

    elif provider_type == "ollama":
        base_url = base_url or "http://localhost:11434"
        return OllamaProvider(model_name=model_name, base_url=base_url)

    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Use 'openai' or 'ollama'")
