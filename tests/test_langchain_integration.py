"""LangChain and LangGraph can use the project's clients, retrievers and pipelines, and every model
call still reaches the cost ledger. Skipped when the ``agent`` extra is not installed."""
from typing import TypedDict

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from supportdesk.integrations.langchain import (
    ChatModel,
    EmbeddingModel,
    Retriever,
    pipeline_node,
)
from supportdesk.pipelines.baseline import build_baseline
from supportdesk.retrieval.retriever import DenseRetriever

CASE = {"case_id": "g1", "title": "Session state is empty after browser refresh",
        "body": "st.session_state is lost after F5. Streamlit 1.31.1, Chrome.",
        "created_at": "2024-04-20T10:00:00Z"}


def rows_for(clients, stage):
    return [r for r in clients.ledger.rows if r["stage"] == stage]


def test_chat_model_goes_through_the_clients_and_the_ledger(clients):
    model = ChatModel(client=clients.chat)
    reply = model.invoke([SystemMessage("be brief"), HumanMessage("hi"), AIMessage("hello"),
                          HumanMessage("again")])
    assert isinstance(reply, AIMessage) and reply.content
    assert len(rows_for(clients, "langchain")) == 1  # counted, so the budget guard sees it
    assert clients.chat.seen[-1] == [
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}, {"role": "user", "content": "again"}]


def test_chat_model_works_inside_a_langchain_chain_and_can_ask_for_json(clients):
    chain = ChatModel(client=clients.chat, stage="chain", json_mode=True) | StrOutputParser()
    assert isinstance(chain.invoke("hello"), str) and len(rows_for(clients, "chain")) == 1
    seen = []

    class Spy:
        def complete(self, messages, *, stage, json_mode):
            seen.append((stage, json_mode))
            return "x", {}
    ChatModel(client=Spy(), json_mode=True).invoke("hi")
    ChatModel(client=Spy(), json_mode=False, stage="other").invoke("hi")
    assert seen == [("langchain", True), ("other", False)]  # each value reaches the client


def test_chat_model_rejects_message_types_it_cannot_map(clients):
    from langchain_core.messages import ToolMessage
    with pytest.raises(ValueError, match="unsupported message type"):
        ChatModel(client=clients.chat).invoke([ToolMessage(content="x", tool_call_id="1")])


def test_embedding_model_uses_the_embedding_client(clients):
    emb = EmbeddingModel(clients.embedding)
    vectors = emb.embed_documents(["a", "b"])
    assert len(vectors) == 2 and len(emb.embed_query("a")) == len(vectors[0])
    assert rows_for(clients, "langchain-documents") and rows_for(clients, "langchain-query")


def test_retriever_returns_documents_with_provenance(index, clients):
    retriever = Retriever(retriever=DenseRetriever(index, clients.embedding), k=3)
    docs = retriever.invoke("session state lost after browser refresh")
    assert len(docs) == 3 and docs[0].page_content
    assert {"chunk_id", "source_id", "source_type", "url", "score", "rank"} <= set(docs[0].metadata)
    assert "fixture-doc-session-state" in {d.metadata["source_id"] for d in docs}
    early = Retriever(retriever=DenseRetriever(index, clients.embedding), k=20,
                      as_of="2024-03-11T11:00:00Z").invoke("widget state release")
    assert "fixture-release-1.33.0" not in {d.metadata["source_id"] for d in early}


def test_a_pipeline_runs_as_a_langgraph_node(index, clients, settings):
    pipeline = build_baseline(DenseRetriever(index, clients.embedding), clients, settings)

    class State(TypedDict, total=False):
        case: dict
        result: dict
    graph = StateGraph(State)
    graph.add_node("analyze", pipeline_node(pipeline))
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", END)
    result = graph.compile().invoke({"case": CASE})["result"]
    assert result["outcome"] == "answer" and result["citations"] and result["pipeline"] == "baseline"
    assert rows_for(clients, "answer")  # the model call inside the node is on the ledger


def test_langchain_pieces_compose_in_a_langgraph_flow(index, clients):
    retriever = Retriever(retriever=DenseRetriever(index, clients.embedding), k=2)
    model = ChatModel(client=clients.chat, stage="summarise")

    class State(TypedDict, total=False):
        question: str
        docs: list
        answer: str
    graph = StateGraph(State)
    graph.add_node("retrieve", lambda s: {"docs": retriever.invoke(s["question"])})
    graph.add_node("answer", lambda s: {"answer": model.invoke(
        "Summarise: " + " ".join(d.page_content for d in s["docs"])).content})
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    out = graph.compile().invoke({"question": "session state after refresh"})
    assert len(out["docs"]) == 2 and out["answer"]
    assert rows_for(clients, "query") and rows_for(clients, "summarise")  # embedding and chat both counted


def test_a_graph_can_pause_for_human_approval_and_resume(index, clients, settings):
    """Human-in-the-loop shape: analyse, wait for a maintainer, then act (or not)."""
    pipeline = build_baseline(DenseRetriever(index, clients.embedding), clients, settings)

    class State(TypedDict, total=False):
        case: dict
        result: dict
        approved: bool
        action: str

    def approval(state):
        decision = interrupt({"proposal": state["result"]["user_reply"]})
        return {"approved": decision["approve"]}

    def act(state):
        return {"action": "commented" if state["approved"] else "nothing"}
    graph = StateGraph(State)
    graph.add_node("analyze", pipeline_node(pipeline))
    graph.add_node("approval", approval)
    graph.add_node("act", act)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "approval")
    graph.add_edge("approval", "act")
    graph.add_edge("act", END)
    app = graph.compile(checkpointer=InMemorySaver())

    for thread, approve, expected in (("t1", True, "commented"), ("t2", False, "nothing")):
        config = {"configurable": {"thread_id": thread}}
        paused = app.invoke({"case": CASE}, config)
        assert "__interrupt__" in paused and "action" not in paused  # nothing happened yet
        assert paused["__interrupt__"][0].value["proposal"]  # the maintainer sees the proposal
        assert app.invoke(Command(resume={"approve": approve}), config)["action"] == expected
    assert len(rows_for(clients, "answer")) == 2  # each case analysed once; resuming did not re-run it
