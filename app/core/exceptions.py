class AIServiceError(Exception):
    """Base exception for the AI service."""


class ModelLoadError(AIServiceError):
    """Raised when an AI model cannot be loaded."""


class ModelInferenceError(AIServiceError):
    """Raised when model inference fails."""


class InvalidTextError(AIServiceError):
    """Raised when input text is invalid."""


class UnsupportedLanguageError(AIServiceError):
    """Raised when a language is not supported."""


class ConfigurationError(AIServiceError):
    """Raised when application configuration is invalid."""