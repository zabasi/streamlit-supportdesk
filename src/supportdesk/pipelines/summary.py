"""The maintainer summary, assembled in code from validated fields (never free-form model text)."""
from __future__ import annotations

from ..data.environment import missing_environment
from ..types import CaseInput
from .packet import Packet
from .parsing import ParsedAnswer


def build_summary(case: CaseInput, env: dict, packet: Packet, answer: ParsedAnswer) -> dict:
    """The maintainer summary, built in code from the validated answer."""
    cited = {cid for cid in answer.citations}
    for fact in answer.known_facts:
        cited.update(fact["evidence"])
    by_id = {c.chunk_id: c for c in packet.chunks}
    return {
        "problem": case.title,
        "outcome": answer.outcome,
        "environment": {"known": {k: v for k, v in env.items() if v},
                        "unknown": missing_environment(env)},
        "known_facts": answer.known_facts,
        "open_questions": answer.open_questions,
        "next_step": answer.next_step,
        "sources": [{"chunk_id": cid, "type": by_id[cid].source_type, "url": by_id[cid].url,
                     "version": by_id[cid].version} for cid in sorted(cited) if cid in by_id],
    }
