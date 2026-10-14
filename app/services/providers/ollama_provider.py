from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Local Ollama LLM provider.

    Responsibilities:
        - Communicate with the local Ollama server.
        - Send prompts to the configured model.
        - Handle HTTP/network errors.
        - Validate the Ollama response.
        - Return generated text.

    This class does not contain municipal insight business logic.
    """

    _instance: "OllamaProvider | None" = None

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

        logger.info(
            "Initializing OllamaProvider. "
            "base_url=%s model=%s timeout_seconds=%s",
            self.base_url,
            self.model,
            self.timeout_seconds,
        )

    @classmethod
    def get_instance(cls) -> "OllamaProvider":
        """
        Return the shared OllamaProvider instance.
        """

        if cls._instance is None:
            logger.info("Creating OllamaProvider singleton instance.")
            cls._instance = cls()

        else:
            logger.debug(
                "Reusing existing OllamaProvider singleton instance."
            )

        return cls._instance

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text from the configured local Ollama model.

        Args:
            prompt:
                Prompt sent to the LLM.

            temperature:
                Controls generation randomness.
                Lower values are preferred for factual municipal
                summaries and insights.

            max_tokens:
                Optional generation limit.

        Returns:
            Generated text.

        Raises:
            ModelInferenceError:
                If Ollama is unavailable, returns an invalid response,
                or generation fails.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ModelInferenceError(
                "Ollama prompt must not be empty."
            )

        if not 0.0 <= temperature <= 2.0:
            raise ModelInferenceError(
                "Ollama temperature must be between 0.0 and 2.0."
            )

        if max_tokens is not None and max_tokens <= 0:
            raise ModelInferenceError(
                "Ollama max_tokens must be greater than zero."
            )

        url = f"{self.base_url}/api/generate"

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        logger.info(
            "Sending generation request to Ollama. "
            "model=%s prompt_length=%d temperature=%.2f",
            self.model,
            len(prompt),
            temperature,
        )

        try:
            with httpx.Client(
                timeout=self.timeout_seconds
            ) as client:
                response = client.post(
                    url,
                    json=payload,
                )

        except httpx.ConnectError as exc:
            logger.exception(
                "Unable to connect to Ollama. "
                "url=%s model=%s",
                url,
                self.model,
            )
            raise ModelInferenceError(
                "Ollama server is unavailable."
            ) from exc

        except httpx.TimeoutException as exc:
            logger.exception(
                "Ollama request timed out. "
                "model=%s timeout_seconds=%s",
                self.model,
                self.timeout_seconds,
            )
            raise ModelInferenceError(
                "Ollama generation timed out."
            ) from exc

        except httpx.RequestError as exc:
            logger.exception(
                "Ollama HTTP request failed. "
                "model=%s error=%s",
                self.model,
                str(exc),
            )
            raise ModelInferenceError(
                "Ollama request failed."
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected error while communicating with Ollama. "
                "error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
            )
            raise ModelInferenceError(
                "Unexpected Ollama provider error."
            ) from exc

        if response.status_code != 200:
            logger.error(
                "Ollama returned an unsuccessful HTTP status. "
                "status_code=%d response=%s",
                response.status_code,
                response.text[:1000],
            )

            raise ModelInferenceError(
                f"Ollama returned HTTP {response.status_code}."
            )

        try:
            data: dict[str, Any] = response.json()

        except ValueError as exc:
            logger.exception(
                "Ollama returned invalid JSON."
            )
            raise ModelInferenceError(
                "Ollama returned an invalid response."
            ) from exc

        generated_text = data.get("response")

        if not isinstance(generated_text, str):
            logger.error(
                "Ollama response is missing the 'response' field."
            )
            raise ModelInferenceError(
                "Ollama response does not contain generated text."
            )

        generated_text = generated_text.strip()

        if not generated_text:
            logger.error(
                "Ollama returned an empty response."
            )
            raise ModelInferenceError(
                "Ollama returned an empty response."
            )

        logger.info(
            "Ollama generation completed successfully. "
            "model=%s response_length=%d",
            self.model,
            len(generated_text),
        )

        return generated_text