"""The corrective loop that remembers: every chunk found in any try is kept with its score, and the
chunks the generator sees are chosen from that pool once the loop ends.

``retrieve_loop_best`` returns the highest-scoring chunks of the whole pool.
``retrieve_loop_mixed`` returns a third of the best, a third from the middle and a third of the
worst, so the model also sees weak evidence for contrast.

The score of a chunk is a number from 0 to 10 that the judge gives it, in the same call that
decides yes or no. A chunk the judge did not score gets its retrieval similarity times 10.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config import Settings
from ..llm.factory import Clients
from ..retrieval.retriever import RANK_DECIMALS, Retriever
from .retrieve_loop import RetrieveLoopPipeline, _passages
from .stages import default_stages

_BRACKETED = re.compile(r"\[([^\]\n]+)\]\s*[:=\-]?\s*(\d+(?:\.\d+)?)")


def parse_scores(text: str, known: set[str]) -> dict[str, float]:
    """The 0-10 score the judge gave each passage, by chunk id. Ids it did not mention are missing."""
    found: dict[str, float] = {}
    for chunk_id, value in _BRACKETED.findall(text):
        if chunk_id.strip() in known:
            found[chunk_id.strip()] = min(10.0, max(0.0, float(value)))
    for chunk_id in known - found.keys():  # "doc:w:1: 7" without the brackets
        match = re.search(re.escape(chunk_id) + r"\s*[:=\-]\s*(\d+(?:\.\d+)?)", text)
        if match:
            found[chunk_id] = min(10.0, max(0.0, float(match.group(1))))
    return found


def rank_pool(pool: dict) -> list[dict]:
    """The pool's entries, best score first. Ties go to the higher retrieval similarity, then to the
    earlier try, then to the chunk id, so the order is the same on every run."""
    return sorted(pool.values(), key=lambda e: (-round(e["score"], RANK_DECIMALS),
                                                -round(e["row"]["score"], RANK_DECIMALS),
                                                e["iteration"], e["row"]["chunk"]["chunk_id"]))


def select_best(ranked: list[dict], n: int) -> list[tuple[dict, str]]:
    """The ``n`` highest-scoring entries."""
    return [(entry, "best") for entry in ranked[:n]]


def split_sizes(n: int) -> tuple[int, int, int]:
    """(best, middle, worst) places for ``n`` chunks: thirds, and what does not divide goes to the best
    first and then to the middle (7 -> 3, 2, 2; 8 -> 3, 3, 2)."""
    base, extra = divmod(n, 3)
    return base + (1 if extra >= 1 else 0), base + (1 if extra == 2 else 0), base


def select_mixed(ranked: list[dict], n: int) -> list[tuple[dict, str]]:
    """A third of the best, a third from the middle of the pool, a third of the worst."""
    n = min(n, len(ranked))
    best, middle, worst = split_sizes(n)
    rest = ranked[best: len(ranked) - worst]
    start = (len(rest) - middle) // 2  # the middle chunks are the ones around the median
    return ([(e, "best") for e in ranked[:best]]
            + [(e, "middle") for e in rest[start: start + middle]]
            + [(e, "worst") for e in ranked[len(ranked) - worst:]])


class RetrieveLoopPoolPipeline(RetrieveLoopPipeline):
    """retrieve_loop that keeps every scored chunk and chooses the final ones after the loop."""

    mode = "best"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.judge_prompt = Path(self.settings.prompts.judge_scored).read_text(encoding="utf-8")

    def describe(self) -> dict:
        limit = self.settings.retrieval.max_loops
        return {**super().describe(), "loop": f"judge and score, then rewrite, at most {limit} searches",
                "selection": f"{self.mode} of the pool of scored chunks"}

    def _select(self, ranked: list[dict], n: int) -> list[tuple[dict, str]]:
        return select_best(ranked, n)

    def _judge(self, state) -> dict:
        extra = dict(state.get("extra") or {})
        question = extra.get("question") or state["queries"][0]
        rows = state.get("hits") or []
        messages = [{"role": "system", "content": self.judge_prompt},
                    {"role": "user", "content": f"Question:\n{question}\n\nPassages:\n{_passages(rows)}"}]
        text, _ = self.clients.chat.complete(messages, stage="judge", json_mode=False)
        extra["sufficient"] = "yes" if text.strip().lower().startswith("yes") else "no"
        scores = parse_scores(text, {row["chunk"]["chunk_id"] for row in rows})
        pool = dict(extra.get("pool") or {})
        for row in rows:
            chunk_id = row["chunk"]["chunk_id"]
            judged = chunk_id in scores
            value = scores[chunk_id] if judged else min(10.0, max(0.0, row["score"] * 10))
            kept = pool.get(chunk_id)
            if kept is None or value > kept["score"]:  # a chunk found twice keeps its best score
                pool[chunk_id] = {"row": row, "score": value, "iteration": int(extra.get("attempt") or 0),
                                  "score_source": "judge" if judged else "similarity"}
        extra["pool"] = pool
        return {"extra": extra}

    def _packet_node(self, state) -> dict:
        extra = state.get("extra") or {}
        sufficient = extra.get("sufficient") or "no"
        reason = ("passages were sufficient" if sufficient == "yes"
                  else f"reached maximum tries ({self.settings.retrieval.max_loops})")
        ranked = rank_pool(extra.get("pool") or {})
        chosen = self._select(ranked, self.settings.retrieval.top_k)
        rows = [{**entry["row"], "rank": rank, "source": f"loop_{group}"}
                for rank, (entry, group) in enumerate(chosen, start=1)]
        self._last_loop = {
            "attempt": int(extra.get("attempt") or 0), "sufficient": sufficient, "stop_reason": reason,
            "queries": list(extra.get("searched") or []),
            "selection": {"mode": self.mode, "pool_size": len(ranked),
                          "chunks": [{"chunk_id": entry["row"]["chunk"]["chunk_id"], "group": group,
                                      "score": round(entry["score"], 3), "score_source": entry["score_source"],
                                      "found_in_try": entry["iteration"]} for entry, group in chosen]}}
        return {**self._packet({**state, "hits": rows}), "hits": rows}


class RetrieveLoopMixedPipeline(RetrieveLoopPoolPipeline):
    """The pooled loop, returning a third best, a third middle and a third worst."""

    mode = "mixed"

    def _select(self, ranked: list[dict], n: int) -> list[tuple[dict, str]]:
        return select_mixed(ranked, n)


def build_retrieve_loop_best(retriever: Retriever, clients: Clients, settings: Settings) -> RetrieveLoopPoolPipeline:
    """retrieve_loop that keeps every scored chunk of every try and returns the best 6 once the loop ends."""
    return RetrieveLoopPoolPipeline(name="retrieve_loop_best", clients=clients, settings=settings,
                                    **default_stages(retriever, clients, settings))


def build_retrieve_loop_mixed(retriever: Retriever, clients: Clients, settings: Settings) -> RetrieveLoopMixedPipeline:
    """retrieve_loop that keeps every scored chunk of every try and returns a third best, middle and worst."""
    return RetrieveLoopMixedPipeline(name="retrieve_loop_mixed", clients=clients, settings=settings,
                                     **default_stages(retriever, clients, settings))
