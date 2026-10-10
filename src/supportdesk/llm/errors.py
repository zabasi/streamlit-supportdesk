"""Failure vocabulary for model calls. Callers decide what each one means for the case."""


class GatewayError(RuntimeError):
    """Base class."""


class ProviderError(GatewayError):
    """Transient provider/network failure. The only call failure a retry may fix."""


class BudgetExhausted(GatewayError):
    """The ceiling (ours) or the quota (the gateway's) is gone. Stop; never look for another route."""


class CallLimitReached(GatewayError):
    """The per-run cap on upstream calls was hit."""


class InputTooLarge(GatewayError):
    """The request exceeds the configured input size; it is refused before being sent."""


class ReplayMiss(GatewayError):
    """Replay mode was asked for a request that was never recorded."""
