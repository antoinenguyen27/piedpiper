class PiedPiperError(Exception):
    """Base Pied Piper SDK error."""


class ConfigurationError(PiedPiperError):
    """Raised for missing client configuration."""


class AuthenticationError(PiedPiperError):
    """Raised for 401 responses."""


class RequestError(PiedPiperError):
    """Raised for invalid requests or transport failures."""


class ServerError(PiedPiperError):
    """Raised for 5xx responses."""

