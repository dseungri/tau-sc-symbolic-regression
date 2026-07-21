"""Custom exceptions for clear failures."""


class TauSRError(RuntimeError):
    """Base project exception."""


class ConfigurationError(TauSRError):
    """Raised when configuration is missing or invalid."""


class ValidationError(TauSRError):
    """Raised when validation cannot safely continue."""


class SecurityError(TauSRError):
    """Raised when an operation violates repository security rules."""
