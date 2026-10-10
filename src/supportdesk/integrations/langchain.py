"""Adapters that let LangChain and LangGraph use this project's clients and retrievers.

Model calls made through them still pass through the budget guard, the cache and the cost ledger.
LangGraph and LangChain core are required dependencies of the package.
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ..pipelines.staged import StagedPipeline
from ..types import CaseInput

ROLES = {HumanMessage: "user", SystemMessage: "system", AIMessage: "assistant"}


class ChatModel(BaseChatModel):
    """A LangChain chat model backed by a ``ChatClient`` (``clients.chat``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any
    stage: str = "langchain"
    json_mode: bool = False

    @property
    def _llm_type(self) -> str:
        return "supportdesk"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        try:
            wire = [{"role": ROLES[type(m)], "content": m.content} for m in messages]
        except KeyError as exc:
            raise ValueError(f"unsupported message type: {exc.args[0].__name__}") from None
        text, usage = self.client.complete(wire, stage=self.stage, json_mode=self.json_mode)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))],
                          llm_output={"usage": usage})


class EmbeddingModel(Embeddings):
    """LangChain embeddings backed by an ``EmbeddingClient`` (``clients.embedding``)."""

    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts, stage="langchain-documents")[0]

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed([text], stage="langchain-query")[0][0]


class Retriever(BaseRetriever):
    """A LangChain retriever over one of this project's retrievers (dense, bm25, hybrid)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    retriever: Any
    k: int
    as_of: str | None = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return [Document(page_content=h.chunk.text,
                         metadata={"chunk_id": h.chunk.chunk_id, "source_id": h.chunk.source_id,
                                   "source_type": h.chunk.source_type, "url": h.chunk.url,
                                   "version": h.chunk.version, "score": h.score, "rank": h.rank})
                for h in self.retriever.search(query, k=self.k, as_of=self.as_of)]


def pipeline_node(pipeline: StagedPipeline):
    """A LangGraph node: reads ``state["case"]`` (a case as a dict), writes ``state["result"]``.

    Both are plain JSON-compatible dicts, so any LangGraph checkpointer can store them.
    """
    def node(state: dict) -> dict:
        return {"result": pipeline.run(CaseInput.from_dict(state["case"])).to_dict()}
    return node
