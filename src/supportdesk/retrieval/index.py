"""The vector index. Vectors live in Qdrant (embedded mode: a local folder, no server; switching to
a Qdrant server is a client change). Next to it, chunks.jsonl and manifest.json record exactly what
was indexed and how, so a build is reproducible and BM25 can run over the same chunks."""
from __future__ import annotations

import atexit
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from ..config import Chunking
from ..data.chunking import Exclusions
from ..data.corpus import Corpus
from ..llm.protocols import EmbeddingClient
from ..types import Chunk
from .filters import parse_ts

COLLECTION = "chunks"
_CLIENTS: dict[str, QdrantClient] = {}


@atexit.register
def _close_clients() -> None:
    """Close embedded stores before interpreter teardown (avoids a noisy warning at exit)."""
    while _CLIENTS:
        _CLIENTS.popitem()[1].close()


def _disk_client(path: Path) -> QdrantClient:
    """Embedded Qdrant locks its folder, so the whole process shares one client per folder."""
    key = str(path.resolve())
    if key not in _CLIENTS:
        path.mkdir(parents=True, exist_ok=True)
        _CLIENTS[key] = QdrantClient(path=key)
    return _CLIENTS[key]


def _payload(position: int, chunk: Chunk) -> dict:
    stamp = parse_ts(chunk.as_of)
    payload = {"pos": position, "source_type": chunk.source_type}
    if stamp is not None:  # undated chunks (pinned documentation) carry no timestamp: always visible
        payload["as_of_ts"] = stamp.timestamp()
    return payload


def _fill(client: QdrantClient, matrix: np.ndarray, chunks: list[Chunk]) -> None:
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(
        size=int(matrix.shape[1]), distance=models.Distance.COSINE))
    for start in range(0, len(chunks), 256):
        client.upsert(COLLECTION, points=[
            models.PointStruct(id=i, vector=matrix[i].tolist(), payload=_payload(i, chunks[i]))
            for i in range(start, min(start + 256, len(chunks)))])


class IndexUnusable(RuntimeError):
    """The index on disk is missing, stale or built with a different embedding model."""


class VectorIndex:
    def __init__(self, chunks: list[Chunk], manifest: dict, client: QdrantClient,
                 matrix: np.ndarray | None = None):
        self.chunks, self.manifest, self.client = chunks, manifest, client
        self._matrix = matrix  # kept only until saved, so save() can write it to disk

    @classmethod
    def in_memory(cls, chunks: list[Chunk], matrix: np.ndarray, manifest: dict) -> VectorIndex:
        if len(chunks) != matrix.shape[0]:
            raise IndexUnusable(f"{len(chunks)} chunks but {matrix.shape[0]} vectors")
        client = QdrantClient(":memory:")
        _fill(client, matrix, chunks)
        return cls(chunks, manifest, client, matrix)

    @property
    def dimension(self) -> int:
        return int(self.client.get_collection(COLLECTION).config.params.vectors.size)

    def search(self, query_vector: list[float], *, limit: int, as_of: str | None) -> list[tuple[int, float]]:
        """(position in self.chunks, cosine score) for the best ``limit`` chunks visible at ``as_of``:
        undated, or dated strictly before it."""
        if len(query_vector) != self.dimension:
            raise IndexUnusable(f"query width {len(query_vector)} != index width {self.dimension}")
        cutoff = parse_ts(as_of)
        query_filter = None if cutoff is None else models.Filter(should=[
            models.FieldCondition(key="as_of_ts", range=models.Range(lt=cutoff.timestamp())),
            models.IsEmptyCondition(is_empty=models.PayloadField(key="as_of_ts"))])
        found = self.client.query_points(COLLECTION, query=list(map(float, query_vector)),
                                         query_filter=query_filter, limit=limit, with_payload=["pos"])
        return [(int(p.payload["pos"]), float(p.score)) for p in found.points]

    def save(self, directory: str | Path) -> None:
        if self._matrix is None:
            raise IndexUnusable("this index was loaded from disk; it is already saved")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        _fill(_disk_client(d / "qdrant"), self._matrix, self.chunks)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as h:
            for c in self.chunks:
                h.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (d / "manifest.json").write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str | Path, *, expect_model: str | None = None,
             expect_chunking: Chunking | None = None) -> VectorIndex:
        d = Path(directory)
        if not (d / "manifest.json").exists():
            raise IndexUnusable(f"no index at {d}; run `supportdesk index` first")
        manifest = json.loads((d / "manifest.json").read_text())
        if expect_model and manifest.get("embedding_model") != expect_model:
            raise IndexUnusable(f"index was built with {manifest.get('embedding_model')!r} but the "
                              f"configured embedding model is {expect_model!r}; rebuild it")
        built = manifest.get("chunking", {})
        want = None if expect_chunking is None else (
            expect_chunking.method, expect_chunking.max_tokens, expect_chunking.overlap_tokens)
        have = (built.get("method", "fixed"), built.get("max_tokens"), built.get("overlap_tokens"))
        if want is not None and have != want:
            raise IndexUnusable(
                f"index was built with chunking {have[0]} {have[1]}/{have[2]} (method max_tokens/overlap) "
                f"but the configuration says {want[0]} {want[1]}/{want[2]}; rebuild it with `supportdesk index`")
        if not (d / "qdrant").exists():
            raise IndexUnusable(f"index at {d} has no vector store; "
                                "rebuild it with `supportdesk index`")
        with (d / "chunks.jsonl").open(encoding="utf-8") as h:
            chunks = [Chunk.from_dict(json.loads(line)) for line in h if line.strip()]
        client = _disk_client(d / "qdrant")
        if not client.collection_exists(COLLECTION) or client.count(COLLECTION).count != len(chunks):
            raise IndexUnusable(f"vector store at {d} does not match chunks.jsonl; rebuild it")
        return cls(chunks, manifest, client)


def build_index(chunks: list[Chunk], embedder: EmbeddingClient, *, embedding_model: str,
                chunking: Chunking, corpus: Corpus | None = None,
                exclusions: Exclusions | None = None, mode: str = "",
                notes: dict | None = None) -> VectorIndex:
    """Embed and store in one call (the indexing graph runs these as two nodes)."""
    if not chunks:
        raise IndexUnusable("nothing to index")
    vectors, _ = embedder.embed([c.text for c in chunks], stage="index")
    return index_from_vectors(chunks, vectors, embedding_model=embedding_model, chunking=chunking,
                              corpus=corpus, exclusions=exclusions, mode=mode, notes=notes)


def index_from_vectors(chunks: list[Chunk], vectors: list, *, embedding_model: str, chunking: Chunking,
                       corpus: Corpus | None = None, exclusions: Exclusions | None = None,
                       mode: str = "", notes: dict | None = None) -> VectorIndex:
    if not chunks:
        raise IndexUnusable("nothing to index")
    if len(vectors) != len(chunks):
        raise IndexUnusable(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    ids_digest = hashlib.sha256("\n".join(c.chunk_id for c in chunks).encode()).hexdigest()
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": embedding_model, "mode": mode, "dimension": len(vectors[0]),
        "vector_store": "qdrant (embedded)",
        "chunk_count": len(chunks), "chunks_by_type": dict(Counter(c.source_type for c in chunks)),
        "chunk_ids_sha256": ids_digest,
        "chunking": {"method": chunking.method, "max_tokens": chunking.max_tokens,
                     "overlap_tokens": chunking.overlap_tokens},
        "exclusions": exclusions.digest() if exclusions else {},
        "corpus": corpus.manifest() if corpus else {},
        **(notes or {}),
    }
    return VectorIndex.in_memory(chunks, np.asarray(vectors, dtype=np.float32), manifest)
