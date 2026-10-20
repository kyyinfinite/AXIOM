from collections.abc import Sequence


class EurusError(Exception):
    pass


class ConfigError(EurusError):
    pass


class ModelRefError(EurusError, ValueError):
    pass


class PolicyError(EurusError):
    pass


class ToolError(EurusError):
    pass


class ProviderError(EurusError):
    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    pass


class AccessDeniedError(AuthenticationError):
    def __init__(
        self,
        message: str,
        *,
        reason: str = "forbidden",
        provider: str = "",
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            status=status,
            retryable=retryable,
            retry_after=retry_after,
        )
        self.reason = reason


class RateLimitError(ProviderError):
    pass


class InvalidRequestError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class AllProvidersFailedError(EurusError):
    def __init__(self, failures: Sequence[tuple[str, ProviderError]]) -> None:
        summary = "; ".join(f"{label}: {error}" for label, error in failures)
        super().__init__(f"all models failed: {summary}")
        self.failures = tuple(failures)
