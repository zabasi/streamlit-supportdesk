"""The controlled-routing v2 generator as a pipeline stage (pipeline.generator = controlled_v2).

The vendored package (supportdesk/generation/controlled_v2, unchanged from the handoff) builds one
Chat Completions request for a structured assessment, and Python routes the action from it (answer,
request_info or escalate). This stage gives that package a ``complete`` function backed by the project's
gateway, so the one model call is cached, recorded in the cost ledger and capped by the budget guard like
every other call, and converts the result with the package's own ``to_pipeline_answer``.

The package fixes its model (gpt-5.4-mini) and settings (reasoning_effort medium, 8,000 completion
tokens, a strict JSON schema). A refusal, a cut-off answer, invalid JSON or an assessment that does not
establish an action makes the package raise ValueError; this stage reports it as GenerationFailed with
the call's usage, so the case is recorded as an error and the run goes on, never relabelled as an
escalation (the package's contract). Budget and provider errors keep the gateway's own handling.
"""
from __future__ import annotations

import dataclasses

from ...config import Settings
from ...generation.controlled_v2 import generate_from_packet, to_pipeline_answer
from ...llm.factory import Clients, chat_client_for
from ...types import CaseInput
from ..packet import Packet
from ..parsing import ParsedAnswer
from .protocols import GenerationFailed

MODEL = "gpt-5.4-mini"  # the package's request builder accepts only this model
MAX_COMPLETION_TOKENS = 8000  # the package's request sets this; the gateway sends its own limit


class ControlledGenerator:
    """One structured assessment by the model; Python owns the action (controlled routing v2)."""

    def __init__(self, clients: Clients, settings: Settings):
        limit = dataclasses.replace(settings.generation, max_output_tokens=MAX_COMPLETION_TOKENS)
        self.chat = chat_client_for(clients, dataclasses.replace(settings, generation=limit), MODEL)

    def complete(self, request: dict) -> dict:
        """The package's transport: one call through the gateway, returned as a raw response dict."""
        if request.get("model") != MODEL:
            raise ValueError(f"controlled_v2 expects the model {MODEL}, got {request.get('model')!r}")
        extra = {"response_format": request["response_format"], "reasoning_effort": request["reasoning_effort"]}
        text, usage = self.chat.complete(request["messages"], stage="answer", json_mode=True, extra=extra)
        self.last_usage = usage
        return {"choices": [{"finish_reason": usage.get("finish_reason") or "stop",
                             "message": {"content": text, "refusal": usage.get("refusal")}}],
                "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                          "completion_tokens": usage.get("output_tokens", 0)}}

    def __call__(self, case: CaseInput, env: dict, packet: Packet) -> ParsedAnswer:
        self.last_usage = None
        if not case.environment and env:
            # a case typed into the app has no reported environment of its own: give the package the one
            # the pipeline extracted from the report (an evaluation case keeps its reported environment, as
            # in the package's verified run)
            case = dataclasses.replace(case, environment=dict(env))
        try:
            result = generate_from_packet(case, packet, self.complete)
        except ValueError as exc:  # no usable assessment: an error for this case, with the call's usage
            raise GenerationFailed(f"controlled_v2: {exc}", attempts=[
                {"error": str(exc), "usage": self.last_usage}]) from None
        answer = to_pipeline_answer(result)
        # the full controlled result (routing, assessment, reference map, maintainer summary) is kept for
        # diagnostics with the call's usage, as the package's README recommends
        answer.attempts = [{"controlled_v2": {k: v for k, v in result.items() if k != "assessment"},
                            "usage": result.get("usage", {})}]
        return answer


__all__ = ["MAX_COMPLETION_TOKENS", "MODEL", "ControlledGenerator"]
