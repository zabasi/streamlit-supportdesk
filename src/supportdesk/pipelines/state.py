"""Graph state: plain, JSON-like data only. LangGraph saves the state after every node (that is
what lets a run pause for a human and resume later), so objects are converted at node edges."""
from __future__ import annotations

from dataclasses import asdict
from typing import TypedDict

from ..retrieval.retriever import Hit
from ..types import CaseInput, Chunk
from .packet import Packet
from .parsing import ParsedAnswer


class SupportState(TypedDict, total=False):
    case: dict          # CaseInput
    env: dict           # environment read from the report
    queries: list[str]
    candidates: list[dict]  # retriever output (Hit), the whole pool
    hits: list[dict]        # after the reranker, top_k
    packet: dict            # Packet: what the model sees
    generated: dict         # ParsedAnswer as the generator returned it
    answer: dict            # ParsedAnswer after validation
    summary: dict           # maintainer summary
    extra: dict             # data written by a pipeline's extra nodes (see StagedPipeline.extra_nodes)


def case_to_dict(case: CaseInput) -> dict:
    return asdict(case)


def case_from_dict(row: dict) -> CaseInput:
    return CaseInput(**row)


def hit_to_dict(hit: Hit) -> dict:
    return {"chunk": hit.chunk.to_dict(), "score": float(hit.score), "rank": hit.rank, "source": hit.source}


def hit_from_dict(row: dict) -> Hit:
    return Hit(Chunk.from_dict(row["chunk"]), row["score"], row["rank"], row["source"])


def packet_to_dict(packet: Packet) -> dict:
    return {"chunks": [c.to_dict() for c in packet.chunks], "labels": dict(packet.labels),
            "token_count": packet.token_count, "dropped": list(packet.dropped), "sha256": packet.sha256}


def packet_from_dict(row: dict) -> Packet:
    return Packet([Chunk.from_dict(c) for c in row["chunks"]], dict(row["labels"]), row["token_count"],
                  list(row["dropped"]), row["sha256"])


def answer_to_dict(answer: ParsedAnswer) -> dict:
    return asdict(answer)


def answer_from_dict(row: dict) -> ParsedAnswer:
    return ParsedAnswer(**row)
