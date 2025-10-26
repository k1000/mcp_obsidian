"""
Embedding provider implementations.
"""

import logging
from typing import List

import httpx

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider for local embeddings."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: int = 30,
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            base_url: Ollama API base URL
            model: Embedding model name
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._dimension = None

        logger.info(f"Initialized OllamaEmbeddingProvider with model '{model}'")

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for text in texts:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                    )
                    response.raise_for_status()
                    data = response.json()

                    embedding = data.get("embedding", [])
                    if not embedding:
                        raise ValueError("No embedding returned from Ollama")

                    # Cache dimension on first call
                    if self._dimension is None:
                        self._dimension = len(embedding)

                    embeddings.append(embedding)

                except Exception as e:
                    logger.error(f"Error embedding text with Ollama: {e}")
                    raise RuntimeError(f"Failed to generate embedding: {e}") from e

        return embeddings

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        if self._dimension is None:
            # Return default dimension for common models
            # Will be updated on first embedding call
            model_dimensions = {
                "nomic-embed-text": 768,
                "mxbai-embed-large": 1024,
                "all-minilm": 384,
            }
            return model_dimensions.get(self.model, 768)

        return self._dimension

    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Name of the provider
        """
        return "ollama"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            dimensions: Optional embedding dimensions (for models that support it)
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._dimension = None

        logger.info(f"Initialized OpenAIEmbeddingProvider with model '{model}'")

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                payload = {
                    "model": self.model,
                    "input": texts,
                }

                if self.dimensions:
                    payload["dimensions"] = self.dimensions

                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("data", []):
                    embedding = item.get("embedding", [])
                    embeddings.append(embedding)

                    # Cache dimension on first call
                    if self._dimension is None:
                        self._dimension = len(embedding)

            except Exception as e:
                logger.error(f"Error embedding text with OpenAI: {e}")
                raise RuntimeError(f"Failed to generate embedding: {e}") from e

        return embeddings

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        if self._dimension is not None:
            return self._dimension

        if self.dimensions:
            return self.dimensions

        # Default dimensions for common models
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.model, 1536)

    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Name of the provider
        """
        return "openai"
