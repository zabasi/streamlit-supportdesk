"""The policy-v3 generator as a pipeline stage (pipeline.generator = policy_v3).

supportdesk/generation/policy_v3 asks the model to choose the support action itself (answer, request_info
or escalate), guided by its system prompt, and returns the action with a structured record. The prompt is
prompts.policy: configs/prompts/policy_v3_system.txt (the original), policy_v3b_system.txt or
policy_v3c_system.txt. This stage gives it the retrieved packet as evidence and a chat client backed by the
project's gateway, so the call is cached, recorded in the cost ledger and capped by the budget guard like
every other call.

It runs with gpt-5.4-mini, reasoning_effort medium and 8,000 completion tokens, as in the
original policy-v3 runs. An outcome other than an action (invalid JSON, a schema violation, a withheld record)
is reported as GenerationFailed, so the case is an error and the run goes on, never relabelled as an
escalation, as with the controlled-routing v2 stage.
"""
from __future__ import annotations

import dataclasses

from ...config import Settings
from ...generation.policy_context import OUTCOMES
from ...generation.policy_v3 import run_policy_turn, sources_from_packet
from ...llm.factory import Clients, chat_client_for
from ...types import CaseInput
from ..packet import Packet
from ..parsing import ParsedAnswer
from .protocols import GenerationFailed

MODEL = "gpt-5.4-mini"
MAX_COMPLETION_TOKENS = 8000


class PolicyV3Generator:
    """The model chooses the action, guided by the policy-v3 prompt in prompts.policy."""

    def __init__(self, clients: Clients, settings: Settings):
        limit = dataclasses.replace(settings.generation, max_output_tokens=MAX_COMPLETION_TOKENS)
        self.chat = chat_client_for(clients, dataclasses.replace(settings, generation=limit), MODEL)
        self.prompt_path = str(settings.prompts.policy)

    def __call__(self, case: CaseInput, env: dict, packet: Packet) -> ParsedAnswer:
        if not case.environment and env:  # an app case: the environment extracted from the report
            case = dataclasses.replace(case, environment=dict(env))
        turn = run_policy_turn(case, sources_from_packet(packet), self.chat, packet=packet, model=MODEL,
                               prompt_path=self.prompt_path)
        record = {"policy_v3": turn.to_dict(), "prompt": self.prompt_path, "usage": turn.usage}
        if turn.outcome not in OUTCOMES:  # no usable action: an error for this case, not an escalation
            raise GenerationFailed(f"policy_v3: {turn.outcome}: {turn.violations}", attempts=[record])
        return ParsedAnswer(
            outcome=turn.outcome, user_reply=turn.user_reply or turn.answer or "",
            next_step=(turn.maintainer_summary or {}).get("next_step"), citations=list(turn.citations),
            attempts=[record], raw_output=turn.raw_output)


__all__ = ["MAX_COMPLETION_TOKENS", "MODEL", "PolicyV3Generator"]
