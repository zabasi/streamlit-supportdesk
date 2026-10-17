"""One question message covering every material gap (pipeline.complete_questions).

When the controlled-routing package decides to ask, it renders the question of the first material gap
the user can supply, so a reply could never ask for two missing things at once, although the model's
assessment often lists two or three (and the maintainer summary already lists them all). With this on,
after the package has routed (the decision itself is never changed), a request_info reply asks for every
such gap in one message: each gap's reason with its references, then a numbered list of their questions.

Purely deterministic: it uses only the gaps the model's own assessment reported for this case.
"""
from __future__ import annotations

import copy

from ...generation.controlled_v2.generation import citations


def askable(assessment: dict) -> list[dict]:
    """The gaps the package would ask about: material, user-providable, not already supplied, with a question."""
    return [gap for gap in assessment.get("information_gaps") or []
            if gap.get("material_to_main_need") and gap.get("user_can_supply")
            and not gap.get("already_supplied_or_tested") and gap.get("question")]


def _reason(gap: dict) -> str:
    """A gap's reason with its references, leaving out any the reason already cites (no "[S3]. [S3]")."""
    text = gap["why_it_matters"].strip()
    missing = [ref for ref in gap["basis_refs"] if f"[{ref}]" not in text]
    return f"{text} {citations(missing)}".strip()


def combine(result: dict) -> dict:
    """The result with one message asking for every askable gap; unchanged unless it asks about two or more."""
    if result.get("outcome") != "request_info" or (result.get("routing") or {}).get("reason") != "material_user_providable_gap":
        return result
    gaps = askable(result.get("assessment") or {})
    if len(gaps) < 2:
        return result
    out = copy.deepcopy(result)
    message = " ".join(_reason(gap) for gap in gaps)
    question = "\n".join(f"{n}. {gap['question']}" for n, gap in enumerate(gaps, start=1))
    out["action"] = {**out["action"], "message": message, "question": question}
    out["user_reply"] = message + "\n\n" + question
    out["routing"] = {**out["routing"], "questions_combined": len(gaps)}
    return out


__all__ = ["askable", "combine"]
