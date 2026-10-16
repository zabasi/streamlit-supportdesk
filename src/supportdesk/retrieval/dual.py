"""Search with two embedding models and fuse their rankings (the dual_doc_slots retriever).

On the development cases the large embedding model ranked several hard documentation pages much higher
(pages that explain a mechanism the report does not name), while the small model found exact API
references the large one ranked low. This retriever searches both indexes, each with its own model and
the usual places for documentation, and fuses the two rankings with weighted RRF: the main model's ranking
counts 1, the second model's retrieval.second_weight.
"""
from __future__ import annotations

from .fusion import weighted_rrf
from .index import IndexUnusable, VectorIndex
from .retriever import DocSlotsRetriever, Hit


class DualDocSlotsRetriever:
    name = "dual_doc_slots"

    def __init__(self, main: DocSlotsRetriever, second: DocSlotsRetriever, *, weight: float, rrf_k: int):
        main_ids = {c.chunk_id for c in main.index.chunks}
        if main_ids != {c.chunk_id for c in second.index.chunks}:
            raise IndexUnusable("the second index holds different chunks: build it with the same chunking "
                                "(supportdesk index with paths.index_dir set to paths.second_index_dir)")
        self.main, self.second, self.weight, self.rrf_k = main, second, weight, rrf_k
        self.index = main.index  # for stages that read the chunks (merging, completion)

    def search(self, query, *, k, as_of=None):
        first = self.main.search(query, k=k, as_of=as_of)
        other = self.second.search(query, k=k, as_of=as_of)
        chunks = {h.chunk.chunk_id: h.chunk for h in first + other}  # the main index's chunk objects first
        for h in first:
            chunks[h.chunk.chunk_id] = h.chunk
        fused = weighted_rrf([[h.chunk.chunk_id for h in first], [h.chunk.chunk_id for h in other]],
                             [1.0, self.weight], k=self.rrf_k)[:k]
        return [Hit(chunks[cid], score, rank, "dual") for rank, (cid, score) in enumerate(fused, start=1)]


def build_dual(index: VectorIndex, clients, settings) -> DualDocSlotsRetriever:
    """The dual retriever from the settings: the main index with the main model, and paths.second_index_dir
    with models.second_embedding."""
    from ..llm.factory import second_embedding_client

    model = settings.models.second_embedding
    second_index = VectorIndex.load(settings.paths.second_index_dir, expect_model=model)
    r = settings.retrieval
    return DualDocSlotsRetriever(
        DocSlotsRetriever(index, clients.embedding, doc_slots=r.doc_slots),
        DocSlotsRetriever(second_index, second_embedding_client(clients, settings, model), doc_slots=r.doc_slots),
        weight=r.second_weight, rrf_k=r.rrf_k)


__all__ = ["DualDocSlotsRetriever", "build_dual"]
