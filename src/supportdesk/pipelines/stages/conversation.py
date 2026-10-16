"""Conversation state that only an interactive caller (the web UI) provides.

QUESTIONS_USED holds how many clarification questions the assistant has already asked in the current
conversation. Answer steps that enforce a follow-up limit (the controlled-routing v2 package) read it;
it is unset (None) everywhere else, so evaluation runs behave exactly as before.
"""
from __future__ import annotations

from contextvars import ContextVar

QUESTIONS_USED: ContextVar[int | None] = ContextVar("questions_used", default=None)

__all__ = ["QUESTIONS_USED"]
