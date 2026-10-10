"""The evidence the model will see: the best chunks, labelled S1, S2, ..., limited to a maximum number
of chunks and tokens. Its hash identifies exactly what was sent."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..retrieval.retriever import Hit
from ..tokenizer import count_tokens
from ..types import Chunk


@dataclass
class Packet:
    chunks: list[Chunk] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)  # label (S1, S2, ...) to chunk_id
    token_count: int = 0
    dropped: list[dict] = field(default_factory=list)
    sha256: str = ""


def assemble_packet(hits: list[Hit], *, max_chunks: int, max_tokens: int) -> Packet:
    packet, seen = Packet(), set()
    for hit in hits:  # hits arrive best-first; keep that order
        chunk = hit.chunk
        if chunk.chunk_id in seen:
            packet.dropped.append({"chunk_id": chunk.chunk_id, "reason": "duplicate"})
            continue
        tokens = count_tokens(chunk.text)
        if len(packet.chunks) >= max_chunks:
            packet.dropped.append({"chunk_id": chunk.chunk_id, "reason": "chunk_limit"})
            continue
        if packet.token_count + tokens > max_tokens:
            packet.dropped.append({"chunk_id": chunk.chunk_id, "reason": "token_limit"})
            continue
        seen.add(chunk.chunk_id)
        packet.chunks.append(chunk)
        packet.labels[f"S{len(packet.chunks)}"] = chunk.chunk_id
        packet.token_count += tokens
    blob = "\n".join(f"{c.chunk_id}\x00{c.text}" for c in packet.chunks)
    packet.sha256 = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return packet
