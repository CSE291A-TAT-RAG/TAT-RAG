"""LLM provider abstraction layer supporting Ollama and AWS Bedrock."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import requests

try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

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


class OllamaProvider(LLMProvider):
    """Ollama LLM provider (local)."""

    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.

        Args:
            model_name: Ollama model name (e.g., llama3:8b, qwen2.5:7b)
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/chat"
        logger.info(f"Initialized Ollama provider with model: {model_name} at {base_url}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate response using Ollama."""
        try:
            # Prepare request payload for Ollama API
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "thinking": {
                        "enabled": False
                    }
                }
            }

            # Make request to Ollama (streaming)
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300,  # 5 minutes timeout
                stream=True
            )
            response.raise_for_status()

            content_parts = []
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            last_chunk: Dict[str, Any] = {}

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode Ollama chunk: %s", line)
                    continue

                last_chunk = chunk

                if "error" in chunk:
                    raise RuntimeError(chunk["error"])

                message = chunk.get("message", {})
                text = message.get("content", "")
                if text:
                    content_parts.append(text)

                if "prompt_eval_count" in chunk:
                    usage["prompt_tokens"] = chunk.get("prompt_eval_count", usage["prompt_tokens"])
                if "eval_count" in chunk:
                    usage["completion_tokens"] = chunk.get("eval_count", usage["completion_tokens"])
                if chunk.get("done"):
                    break

            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            answer = "".join(content_parts).strip()
            if not answer and last_chunk:
                answer = (last_chunk.get("response") or "").strip()

            return {
                "content": answer,
                "model": self.model_name,
                "usage": usage
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to generate with Ollama. Make sure Ollama is running and "
                f"the model '{self.model_name}' is installed. Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to generate with Ollama. Error: {str(e)}"
            )


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider."""

    def __init__(
        self,
        model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        profile_name: Optional[str] = None
    ):
        """
        Initialize AWS Bedrock provider.

        Args:
            model_name: Bedrock model ID (e.g., anthropic.claude-3-sonnet-20240229-v1:0)
            region_name: AWS region name
            aws_access_key_id: AWS access key (optional, can use IAM role or profile)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional, for temporary credentials)
            profile_name: AWS profile name (optional)
        """
        if not BEDROCK_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS Bedrock. Install it with: pip install boto3"
            )

        # Build session kwargs
        session_kwargs = {}
        if profile_name:
            session_kwargs['profile_name'] = profile_name
        if aws_access_key_id:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token

        # Create session and client
        if session_kwargs:
            session = boto3.Session(**session_kwargs)
            self.client = session.client('bedrock-runtime', region_name=region_name)
        else:
            # Use default credentials (IAM role, environment variables, or default profile)
            self.client = boto3.client('bedrock-runtime', region_name=region_name)

        self.model_name = model_name
        self.region_name = region_name
        logger.info(f"Initialized AWS Bedrock provider with model: {model_name} in region: {region_name}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate response using AWS Bedrock with Claude models."""
        try:
            # Convert messages to Bedrock Claude format
            if "anthropic.claude" in self.model_name:
                # Format for Claude models
                system_message = None
                conversation = []

                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "system":
                        system_message = content
                    else:
                        # Map 'assistant' and 'user' roles
                        conversation.append({
                            "role": role,
                            "content": content
                        })

                # Build request body for Claude
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": conversation,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

                if system_message:
                    body["system"] = system_message

            else:
                # Generic format for other models (adjust as needed)
                body = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

            # Call Bedrock API
            response = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract content based on model type
            if "anthropic.claude" in self.model_name:
                # Claude models format
                content = response_body.get("content", [{}])[0].get("text", "")
                usage = response_body.get("usage", {})

                return {
                    "content": content,
                    "model": self.model_name,
                    "usage": {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    }
                }
            else:
                # Generic response parsing (fallback)
                return {
                    "content": response_body.get("completion", ""),
                    "model": self.model_name,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }

        except Exception as e:
            logger.error(f"AWS Bedrock generation failed: {str(e)}")
            raise RuntimeError(
                f"Failed to generate with AWS Bedrock. Error: {str(e)}"
            )


class GeminiProvider(LLMProvider):
    """Placeholder Gemini provider (not implemented in this build)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GeminiProvider is not available in this configuration."
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "GeminiProvider is not available in this configuration."
        )


def create_llm_provider(
    provider_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    aws_profile_name: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider_type: Type of provider ('ollama' or 'bedrock')
        model_name: Model name
        api_key: API key (deprecated, not used)
        base_url: Base URL (for Ollama)
        region_name: AWS region (for Bedrock)
        aws_access_key_id: AWS access key (for Bedrock)
        aws_secret_access_key: AWS secret key (for Bedrock)
        aws_session_token: AWS session token (for Bedrock)
        aws_profile_name: AWS profile name (for Bedrock)

    Returns:
        LLMProvider instance
    """
    provider_type = provider_type.lower()

    if provider_type == "ollama":
        base_url = base_url or "http://localhost:11434"
        return OllamaProvider(model_name=model_name, base_url=base_url)

    elif provider_type == "bedrock":
        region_name = region_name or "us-east-1"
        return BedrockProvider(
            model_name=model_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            profile_name=aws_profile_name
        )

    elif provider_type == "gemini":
        return GeminiProvider()

    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Use 'ollama' or 'bedrock'")
