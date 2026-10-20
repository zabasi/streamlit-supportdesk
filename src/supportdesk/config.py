"""Configuration: one typed, validated object; every tunable lives here, none elsewhere.

Precedence, lowest to highest: built-in defaults, ``configs/default.toml``, the pipeline profile
(``configs/pipelines/<pipeline.name>.toml``, or under ``configs/pipelines/experiments/``), ``$SUPPORTDESK_CONFIG``, each ``--config`` file,
``SUPPORTDESK__SECTION__KEY`` environment variables, then ``--set section.key=value``.

Each value remembers which layer set it (``supportdesk config`` prints it). Unknown keys, wrong
types and inconsistent values fail at load time naming the key. Secrets are not configuration: the
API key comes from the environment and a config file containing one is rejected. Relative paths
resolve against the working directory, so run commands from the project root.
"""
# ruff: noqa: RUF009  (_opt() returns dataclasses.field(); it is not a shared mutable default)
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import types
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10
    import tomli as tomllib

MODES = ("offline", "replay", "live")
OUTCOMES = ("answer", "request_info", "escalate")
FAILURE_OUTCOMES = ("error", "budget_stopped")
#: answerable / ambiguous / escalation: the base issue cases. history_variant and source_usage are
#: reported as their own groups and scored by the rubric (judge), not by outcome.
CATEGORIES = ("answerable", "ambiguous", "escalation", "history_variant", "source_usage")

BASE_URL_ENV = "OPENAI_BASE_URL"
CONFIG_ENV = "SUPPORTDESK_CONFIG"
ENV_PREFIX = "SUPPORTDESK__"
DEFAULT_CONFIG_FILE = Path("configs/default.toml")
PROFILE_DIR = Path("configs/pipelines")

#: What a pipeline profile (configs/pipelines/<name>.toml) may set: the settings that define how that
#: pipeline retrieves and builds its prompt. Models, budget, generation and paths stay shared by every
#: pipeline, so comparisons are fair and one client and cost ledger serve them all.
#: ``None`` means any key in the section.
PROFILE_ALLOWED: dict[str, set[str] | None] = {
    "chunking": None, "retrieval": None, "context": None, "prompts": None,
    "pipeline": {"retriever", "scope_gate", "generator", "version_notes", "answer_model", "complete_questions", "controlled_version"}, "paths": {"index_dir"},
}

SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_\-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._\-]{20,})"
)


def redact(text: object) -> str:
    """Mask anything key-shaped. Applied to every error message before it is stored."""
    return SECRET_PATTERN.sub("<redacted>", str(text))


class ConfigError(RuntimeError):
    """The configuration is invalid, or the environment cannot support the requested mode."""


def _opt(help_text: str, default=None):
    return field(default=default, metadata={"help": help_text})


#: Chunkers the indexing graph can run (see data/chunking.py: CHUNKERS).
CHUNKING_METHODS = ("fixed", "structure", "token_structure", "langchain_rule")


@dataclass(frozen=True)
class Run:
    mode: str = _opt("offline: free fake models; replay: recorded responses only; live: real model calls, which cost money", "offline")


@dataclass(frozen=True)
class Models:
    chat: str = _opt("the chat model's name at the provider; it must have a price in pricing.json", "gpt-4o-mini")
    rerank: str = _opt("the chat model for the reranking call only (it must have a price in pricing.json); empty "
                       "uses models.chat", "")
    embedding: str = _opt("the embedding model's name; changing it means building the index again", "text-embedding-3-small")
    second_embedding: str = _opt("the second embedding model of the dual_doc_slots retriever (its own index: "
                                 "paths.second_index_dir)", "text-embedding-3-large")
    base_url: str | None = _opt("the provider's URL; if unset, it is read from the OPENAI_BASE_URL environment variable")
    api_key_env: str = _opt("the name of the environment variable that holds the API key (never the key itself)", "OPENAI_API_KEY")


@dataclass(frozen=True)
class Prompts:
    system: Path = _opt("system prompt of the answer call", Path("configs/prompts/answer_system.txt"))
    output_format: Path = _opt("JSON shape the answer must follow", Path("configs/prompts/answer_output_format.txt"))
    rewrite: Path = _opt("system prompt of the query-rewriting call (QueryRewriter)",
                         Path("configs/prompts/rewrite_system.txt"))
    rerank: Path = _opt("system prompt of the reranking call (LLMReranker)", Path("configs/prompts/rerank_system.txt"))
    policy: str = _opt("pipeline.generator = policy_v3: its system prompt (configs/prompts/policy_v3_system.txt, "
                       "policy_v3b_system.txt or policy_v3c_system.txt)", "configs/prompts/policy_v3_system.txt")
    rerank_second: str = _opt("a second reranking prompt (for example configs/prompts/rerank_explain_system.txt): "
                              "both rankings are made and merged by retrieval.rerank_merge; empty: one ranking", "")
    scope: Path = _opt("system prompt of the pre-retrieval scope classifier (scope gate on advanced)",
                       Path("configs/prompts/scope_system.txt"))
    section_questions: Path = _opt("system prompt that writes user-style problem descriptions for one "
                                   "documentation section", Path("configs/prompts/section_questions_system.txt"))
    topic_discover: Path = _opt("system prompt that grows a list of routing topics from the corpus, batch by batch "
                                "(scripts/discover_topics.py)", Path("configs/prompts/topic_discover_system.txt"))
    topic_refine: Path = _opt("system prompt that merges the discovered topics into the target range",
                              Path("configs/prompts/topic_refine_system.txt"))
    topic_classify: Path = _opt("system prompt that labels chunks and reports with a fixed list of Streamlit "
                                "topics (class_doc_slots)", Path("configs/prompts/topic_classify_system.txt"))
    label_cluster: Path = _opt("system prompt that names a topic cluster and lists its keywords "
                               "(scripts/build_clusters.py)", Path("configs/prompts/label_cluster_system.txt"))
    scope_rewrite: Path = _opt("system prompt of the combined scope check and concept queries, one call "
                               "(pipeline.scope_gate = combined)", Path("configs/prompts/scope_rewrite_system.txt"))
    summarize_section: Path = _opt("system prompt that summarizes one documentation section "
                                   "(scripts/summarize_sections.py)", Path("configs/prompts/summarize_section_system.txt"))
    judge_coverage: Path = _opt("system prompt of the coverage judge (CorrectiveReranker)",
                                Path("configs/prompts/judge_coverage_system.txt"))
    rewrite_missing: Path = _opt("system prompt of the query for a missing mechanism (CorrectiveReranker)",
                                 Path("configs/prompts/rewrite_missing_system.txt"))
    extract: Path = _opt("system prompt of the call that keeps the relevant parts of the chosen chunks "
                         "(ExtractingReranker)", Path("configs/prompts/extract_system.txt"))
    judge: Path = _opt("system prompt of the retrieve_loop sufficiency check (yes or no)",
                       Path("configs/prompts/judge_system.txt"))
    loop_rewrite: Path = _opt("system prompt of the retrieve_loop query rewrite",
                              Path("configs/prompts/loop_rewrite_system.txt"))
    judge_scored: Path = _opt("system prompt of the retrieve_loop_best and retrieve_loop_mixed judge: yes or no, "
                              "then a 0-10 score for every passage", Path("configs/prompts/judge_scored_system.txt"))


@dataclass(frozen=True)
class Paths:
    data_dir: Path = _opt("the folder that holds corpus.jsonl (its format is described in data/corpus.py)", Path("data/processed"))
    index_dir: Path = _opt("where the index is written (qdrant/ vector store, chunks.jsonl, manifest.json)", Path("index"))
    artifacts_dir: Path = _opt("where runs, reports, the tracker and the cache are written", Path("artifacts"))
    cache_path: Path = _opt("SQLite cache of model responses (makes re-runs free, enables replay)", Path("artifacts/cache.sqlite"))
    ledger_path: Path = _opt("record of every live call's cost; it counts toward the budget across runs", Path("artifacts/cost_ledger.jsonl"))
    pricing_path: Path = _opt("per-model prices used by the budget guard", Path("configs/pricing.json"))
    summaries: Path = _opt("section summaries written by scripts/summarize_sections.py (SummaryReranker)",
                           Path("data/summaries/sections.jsonl"))
    second_index_dir: Path = _opt("the dual_doc_slots retriever's second index, built with models.second_embedding",
                                  Path("index/large"))
    section_questions: Path = _opt("symptom questions per documentation section, with their vectors, for the "
                                   "question_doc_slots retriever (scripts/build_section_questions.py)",
                                   Path("data/questions"))
    class_labels: Path = _opt("topic labels per chunk for the class_doc_slots retriever "
                              "(scripts/build_class_labels.py)", Path("data/classes/llm.jsonl"))
    clusters: Path = _opt("topic clusters written by scripts/build_clusters.py (cluster_doc_slots retriever)",
                          Path("data/clusters/clusters.jsonl"))


@dataclass(frozen=True)
class Budget:
    ceiling_usd: float = _opt("hard stop for live spending on this machine, all runs together", 5.0)
    reserve_usd: float = _opt("kept unspent: a call is refused if it could eat into this", 0.25)
    max_live_calls: int = _opt("the most real model calls one command may make", 5000)


@dataclass(frozen=True)
class Chunking:
    method: str = _opt("which chunker the indexing graph's chunk node runs: " + " | ".join(CHUNKING_METHODS),
                       "fixed")
    max_tokens: int = _opt("chunk size in tokens (the project's tokenizer); changing it means building the index again", 512)
    overlap_tokens: int = _opt("tokens shared by consecutive chunks; must be smaller than max_tokens", 128)
    context_header: bool = _opt("embed each documentation chunk with its page title and heading path in front "
                                "(the stored text is unchanged); needs its own index", False)
    context_header_scope: str = _opt("which headings the context header names: start (the path where the chunk "
                                     "starts) or all (that path plus every section heading inside the chunk)", "start")
    structure_min_chars: int = _opt("structure chunker: sections shorter than this many characters are merged "
                                    "into the next", 150)
    structure_max_chars: int = _opt("structure chunker: sections longer than this many characters are split at "
                                    "block boundaries", 2000)
    token_target: int = _opt("token_structure chunker: the chunk size it aims for, in tokens", 500)
    token_soft_max: int = _opt("token_structure chunker: the most tokens a chunk may grow to before it is cut", 700)
    token_min_section: int = _opt("token_structure chunker: a section of at least this many tokens starts a new "
                                  "chunk; smaller ones are packed together", 150)
    langchain_chunk_chars: int = _opt("langchain_rule chunker: the most characters in a chunk (LangChain's "
                                      "recursive splitter, inside each heading section)", 1600)
    langchain_overlap_chars: int = _opt("langchain_rule chunker: characters shared by neighbouring chunks", 200)

    def extras(self) -> dict:
        """The chunking settings beyond method, sizes and the header switch that change an index: they are
        recorded in it, and an index built with other values is refused."""
        found = {}
        if self.method == "structure":
            found.update(structure_min_chars=self.structure_min_chars, structure_max_chars=self.structure_max_chars)
        if self.method == "token_structure":
            found.update(token_target=self.token_target, token_soft_max=self.token_soft_max,
                         token_min_section=self.token_min_section)
        if self.method == "langchain_rule":
            found.update(langchain_chunk_chars=self.langchain_chunk_chars,
                         langchain_overlap_chars=self.langchain_overlap_chars)
        if self.context_header:
            found["context_header_scope"] = self.context_header_scope
        return found


@dataclass(frozen=True)
class Retrieval:
    top_k: int = _opt("how many chunks the final retrieval step returns", 6)
    candidate_k: int | None = _opt("candidates fetched before reranking; if unset, the same as top_k")
    historical_filter: bool = _opt("hide material dated after a case was filed", True)
    embed_batch_size: int = _opt("texts per embedding request", 64)
    bm25_k1: float = _opt("BM25: how quickly repeating a word stops raising the score", 1.5)
    bm25_b: float = _opt("BM25: how much longer chunks are penalised, from 0 (not at all) to 1", 0.75)
    hybrid_pool: int = _opt("in hybrid search, how many candidates dense and BM25 search each contribute", 30)
    doc_slots: int = _opt("places among the candidates kept for documentation chunks, used by the "
                          "dense_doc_slots retriever; 0 keeps none", 0)
    max_queries: int = _opt("most search queries a query stage may return; their results are merged by "
                            "rank fusion", 1)
    keep_report_query: bool = _opt("query stages that write several queries (QueryRewriter, SentenceSplitQuery) "
                                    "also search with the cleaned report itself", False)
    rerank_snippet_chars: int = _opt("characters of each candidate shown to the model by the LLMReranker", 600)
    rerank_snippet: str = _opt("which part of each candidate the LLMReranker shows: start (its first "
                               "rerank_snippet_chars) | focused (the window that best matches the report, "
                               "with its section heading)", "start")
    rerank_keep_docs: int | None = _opt("documentation chunks the LLMReranker keeps in the final top_k; "
                                        "unset: retrieval.doc_slots", None)
    min_chunks: int = _opt("fewest chunks the GradedReranker keeps, even when fewer are graded relevant", 2)
    extract_mode: str = _opt("ExtractingReranker: the model lists the parts to keep, or the parts to drop", "keep")
    extract_show_concepts: bool = _opt("ExtractingReranker: show the model the concept queries the query stage wrote",
                                       False)
    extract_protect_top: int = _opt("ExtractingReranker: never drop parts of this many top-ranked chunks", 0)
    rerank_show_concepts: bool = _opt("the LLMReranker also sees the concept queries the query stage wrote "
                                      "(\"Mechanisms involved\")", False)
    corrective_turns: int = _opt("CorrectiveReranker: after reranking, a judge checks the kept chunks; if they "
                                 "are not enough, rewrite the query, search again and rerank old and new "
                                 "candidates together, at most this many times (0 = off)", 0)
    corrective_judge: str = _opt("CorrectiveReranker: relevance (yes or no: is there relevant evidence) or coverage "
                                 "(which concept queries no passage explains; one new query per missing one)",
                                 "relevance")
    corrective_max_missing: int = _opt("CorrectiveReranker with the coverage judge: most missing mechanisms "
                                       "searched for in one turn", 2)
    summary_sections: int = _opt("SummaryReranker: before reranking, add the chunks of this many documentation "
                                 "sections whose summaries best match the queries (0 = off)", 0)
    hier_sections: int = _opt("hierarchical retrievers: how many documentation sections, chosen by comparing the "
                              "query with the section summaries, the chunk search looks inside", 3)
    hier_places: int = _opt("hierarchical retrievers: how many of the returned chunks come from inside those "
                            "sections; the rest come from the ordinary search", 3)
    second_weight: float = _opt("dual_doc_slots: how much the second embedding model's ranking counts against "
                                "the main model's in the fusion", 0.5)
    report_views: str = _opt("advanced: extra searches besides the queries, comma-separated: title (the report's "
                             "title alone), code (its st.* commands, their parameter names and its error lines), or "
                             "none; fused with the report's ranking at view_weight", "none")
    view_weight: float = _opt("report_views: how much each view's ranking counts against the report's", 0.5)
    rerank_mode: str = _opt("llm (one reranking call) | score (no call: the candidates' similarity to the report, "
                            "for a question-found one its question's) | rrf (no call: each candidate's rank among "
                            "the candidates of its own source)", "llm")
    rerank_merge: str = _opt("prompts.rerank_second: merge the two rankings by rrf (a candidate's ranks in both) | "
                             "alternate (the best not yet taken of each in turn) | split (the first ranking's best "
                             "rerank_split, then the second's best not yet taken up to top_k)", "rrf")
    rerank_split: int = _opt("rerank_merge = split: pieces taken from the first ranking; the rest of top_k "
                             "comes from the second", 4)
    rerank_merge_k: int = _opt("rerank_merge = rrf: the RRF constant", 10)
    bm25_extra: int = _opt("question_doc_slots: add the best BM25 documentation chunks for the report that are not "
                           "already candidates (exact identifiers); 0 adds none", 0)
    rerank_fuse_weight: float = _opt("blend the reranker's order with each candidate's own search rank (among "
                                     "the candidates of its source) by weighted RRF; 0 uses the reranker's order "
                                     "alone", 0.0)
    rerank_fuse_k: int = _opt("rerank_fuse_weight: the RRF constant (smaller makes the top ranks count more)", 10)
    rerank_versions: bool = _opt("reranking: state the reporter's Streamlit version and tag each candidate with the "
                                 "version it describes, and add a version rule to the prompt "
                                 "(configs/prompts/rerank_version_rule.txt)", False)
    release_slots: int = _opt("retrieval: add up to this many release notes for versions newer than the reported "
                              "one, best matching the query, to the candidates (retrieval/release_slots.py); 0 = off",
                              0)
    rerank_show_question: bool = _opt("the reranker also sees, for a candidate found through a documentation "
                                      "section's symptom question (question_doc_slots), the question that matched "
                                      "the report", False)
    rerank_show_sections: bool = _opt("the reranker also sees the headings of the sections each candidate "
                                      "contains, which its window may not show", False)
    pool_mode: str = _opt("advanced: how the reranker's candidates are chosen: fuse (every query's ranking merged "
                          "with RRF, the best candidate_k kept) | quota (fixed shares: the report's best dense and "
                          "BM25 chunks, and the concept queries' best dense chunks) | union (no fusion: the report's "
                          "own candidates all kept, each concept query adding its best new ones)", "fuse")
    union_report: int = _opt("pool_mode = union: the cleaned report's own candidates, all kept", 15)
    union_per_concept: int = _opt("pool_mode = union: new candidates each concept query adds (its best ones not "
                                  "already in the pool)", 12)
    union_max: int = _opt("pool_mode = union: the most candidates the reranker gets", 40)
    quota_report_dense: int = _opt("pool_mode = quota: dense chunks from the cleaned report", 5)
    quota_report_bm25: int = _opt("pool_mode = quota: keyword (BM25) chunks from the cleaned report", 5)
    quota_concepts_dense: int = _opt("pool_mode = quota: dense chunks from the concept queries, taken in turns", 10)
    concept_weight: float = _opt("with concept queries after the cleaned report (keep_report_query), how much each "
                                 "concept query's ranking counts in the fusion against the report's (1 = equal, plain "
                                 "RRF; below 1, the report's matches keep the lead)", 1.0)
    decompose: bool = _opt("advanced: write concept queries (the decompose step) besides the cleaned report; false "
                           "searches with the cleaned report alone and reranks without the mechanisms list", True)
    complete: str = _opt("after reranking, repair each kept documentation chunk's edges: none | neighbors (add "
                         "the chunk before and after it) | section (grow to the whole top-level sections it "
                         "touches) | subsection (grow each edge only to the innermost section around it) | "
                         "trim (drop a short leading fragment, so it starts at a heading) | "
                         "section_trim (trim the start, complete the last section)", "none")
    complete_max_tokens: int = _opt("retrieval.complete: the most tokens a repaired chunk may grow to", 900)
    cluster_size: int = _opt("cluster routing: about how many chunks scripts/build_clusters.py puts in each "
                             "topic cluster (k-means on the chunk embeddings)", 100)
    cluster_method: str = _opt("cluster routing: how scripts/build_clusters.py groups the chunks: kmeans (the "
                               "embeddings as they are) | centered (each source type's average removed first, so "
                               "clusters follow topic rather than writing style) | doc_anchored (the documentation "
                               "clustered alone, centered; every other chunk joins its nearest documentation cluster)",
                               "kmeans")
    cluster_top: int = _opt("cluster routing: how many clusters each query is routed to", 2)
    cluster_route: str = _opt("cluster routing: how a query is routed: keywords (its words among each "
                              "cluster's keywords) | embedding (its meaning against each cluster's description) | "
                              "centroid (its meaning against the average of each cluster's own chunks)", "keywords")
    cluster_mode: str = _opt("cluster routing: filter (search only the routed clusters) | boost (search "
                             "everything, the routed clusters' chunks ranked higher by retrieval.cluster_boost)",
                             "filter")
    question_filter_top: int = _opt("question_doc_slots, Doc2Query--: keep a symptom question only if one of its "
                                    "this many nearest chunks in the index belongs to its own section; 0 keeps all", 0)
    question_chunk: str = _opt("question_doc_slots: the chunk added for a matched section: overlap (the one "
                               "holding most of the section) | question (the one most similar to the matched "
                               "question)", "overlap")
    question_snippet: str = _opt("question_doc_slots: the window the reranker sees for a candidate found through a "
                                 "symptom question: report (chosen by the report's words) | question (by the "
                                 "matched question's words) | both", "report")
    question_order: str = _opt("question_doc_slots: where the question-found candidates appear in the reranking "
                               "list: append (after the base's) | interleave (one of each in turn) | shuffle "
                               "(an order that says nothing about their source)", "append")
    question_keep: int = _opt("question_doc_slots: at least this many of the kept pieces are question-found "
                              "candidates (like rerank_keep_docs); 0 leaves it to the reranker", 0)
    question_sections: int = _opt("question_doc_slots: sections found through their symptom questions that add "
                                  "a candidate beside the base retriever's", 5)
    class_route: str = _opt("class_doc_slots: how a query picks its topic classes: llm (one model call with "
                            "prompts.topic_classify) | centroid (nearest class centres, no call)", "llm")
    class_top: int = _opt("class_doc_slots: topic classes each query is routed to", 2)
    class_mode: str = _opt("class_doc_slots: filter (search only the routed classes) | boost (search everything, "
                           "the routed classes' chunks ranked higher by retrieval.class_boost)", "filter")
    class_boost: float = _opt("class_doc_slots with class_mode = boost: added to the similarity of chunks in the "
                              "routed classes", 0.05)
    cluster_boost: float = _opt("cluster routing with cluster_mode = boost: added to the similarity of chunks "
                                "in the routed clusters", 0.05)
    expand: str = _opt("ExpandingReranker: add more chunks of the documentation pages among the candidates "
                       "before reranking: none | page (all their other chunks) | section (only chunks sharing a "
                       "section with a found chunk) | branch (for structure chunks: the found chunk's ## parent "
                       "and all its children, from the chunks' heading paths)", "none")
    expand_max_candidates: int = _opt("ExpandingReranker: the most candidates after expansion", 30)
    section_candidates: int = _opt("most sections and other candidates the SectionReranker ranks", 30)
    rerank_doc_places: int = _opt("documentation places filled by the model's choice among the documentation "
                                  "candidates (DocPlacesReranker)", 0)
    bm25_slots: int = _opt("places in the results filled by BM25 (keyword) search, the rest by dense "
                           "search; used by the hybrid_split retriever", 0)
    rrf_k: int = _opt("Reciprocal Rank Fusion constant; higher values make rank differences count less", 60)
    max_loops: int = _opt("retrieve-and-judge tries in the retrieve_loop pipeline; the last chunks "
                          "are used once this many tries have run", 3)


@dataclass(frozen=True)
class Context:
    max_chunks: int | None = _opt("most chunks shown to the model; if unset, retrieval.top_k (it cannot be more)")
    max_tokens: int = _opt("token budget for all evidence in the prompt", 3200)
    max_query_chars: int = _opt("longest search query built from a report", 4000)
    max_body_chars: int = _opt("longest report body shown to the model", 4000)


@dataclass(frozen=True)
class Generation:
    max_output_tokens: int = _opt("cap on tokens the model may generate per call", 900)
    max_retries: int = _opt("retries after a provider error or a reply that is not valid JSON, and nothing else", 1)
    temperature: float = _opt("sampling temperature; 0 gives the most repeatable answers", 0.0)
    timeout_seconds: float = _opt("per-request timeout to the gateway", 60.0)


#: How sure the model says it is; an "answer" with low confidence is escalated in code.
CONFIDENCE = ("high", "medium", "low")

#: Which proposals wait for a maintainer in `supportdesk case`.
REVIEW_POLICIES = ("escalations", "all")


@dataclass(frozen=True)
class Case:
    review: str = _opt("which proposals wait for a maintainer before they are posted: escalations "
                       "(answers and questions are posted directly) or all", "escalations")


@dataclass(frozen=True)
class Limits:
    max_input_chars: int = _opt("a request larger than this is refused before it is sent", 60_000)


@dataclass(frozen=True)
class PipelineCfg:
    name: str = _opt("registered pipeline to run (see registry.py)", "baseline")
    retriever: str = _opt("registered retriever: dense | bm25 | hybrid", "dense")
    scope_gate: str = _opt("advanced only: the scope check before retrieval: llm (a model call), rules (free "
                           "keyword rules), combined (the same call that writes the concept queries) or off "
                           "(no check); advanced's profile turns it on (llm)", "off")
    answer_model: str = _opt("controlled_v2 answer step: the model the package's request is sent to (the package "
                             "builds it for gpt-5.4-mini; gpt-5-mini and gpt-4.1-mini are cheaper)", "gpt-5.4-mini")
    controlled_version: str = _opt("controlled answer step's package: v2 (as delivered), v3 (v2 + an investigation rule) "
                                   "or v4 (v2 + already-supplied facts, escalation with complete controls and length "
                                   "limits); generation/controlled_v3 and _v4", "v2")
    complete_questions: bool = _opt("controlled_v2 answer step: a question asks for every material gap the user "
                                    "can supply in one message, not only the first; the decision is unchanged "
                                    "(pipelines/stages/complete_questions.py)", False)
    version_notes: bool = _opt("controlled_v2 answer step: add version and usage notes to each evidence piece "
                               "(what version it describes, compared with the reporter's; how it may be used); "
                               "pipelines/stages/version_notes.py", False)
    generator: str = _opt("advanced-based pipelines only: the answer step: single_call (the baseline's prompt "
                          "and parser) | controlled_v2 (one structured assessment by gpt-5.4-mini, the action "
                          "routed by Python; supportdesk/generation/controlled_v2) | policy_v3 (gpt-5.4-mini "
                          "chooses the action itself, guided by prompts.policy)", "single_call")


@dataclass(frozen=True)
class EvalCfg:
    case_files: tuple[Path, ...] = _opt("a folder of evaluation files, or case .jsonl files; their source issues are kept out of the index", (Path("data/eval"),))
    test_fraction: float = _opt("share of families assigned to the test split by `supportdesk split`", 0.5)
    split_seed: int = _opt("seed for `supportdesk split`", 7)
    recall_ks: tuple[int, ...] = _opt("k values reported for Recall@k", (1, 3, 6, 10))
    passage_found_fraction: float = _opt("a gold passage counts as found when the retrieved chunks cover at "
                                         "least this fraction of it", 0.8)
    leakage_overlap_threshold: float = _opt("warn when a chunk repeats this fraction of an evaluation case's text", 0.6)
    leakage_shingle_tokens: int = _opt("length, in words, of the word sequences the leakage check compares", 8)


@dataclass(frozen=True)
class Web:
    port: int = _opt("port of the local web page", 8765)
    max_request_bytes: int = _opt("largest accepted request body", 64_000)
    max_title_chars: int = _opt("longest title the UI accepts", 300)
    max_body_chars: int = _opt("longest description the UI accepts", 8000)
    max_created_at_chars: int = _opt("longest date the web page accepts", 40)
    max_history_items: int = _opt("most earlier messages the UI accepts", 20)
    max_history_chars: int = _opt("longest earlier message", 2000)


SECTIONS: dict[str, type] = {
    "run": Run, "models": Models, "prompts": Prompts, "paths": Paths, "budget": Budget, "chunking": Chunking,
    "retrieval": Retrieval, "context": Context, "generation": Generation, "limits": Limits,
    "pipeline": PipelineCfg, "eval": EvalCfg, "web": Web, "case": Case,
}
HINTS = {name: typing.get_type_hints(cls) for name, cls in SECTIONS.items()}

#: Short environment variable names, accepted as well as SUPPORTDESK__SECTION__KEY.
ALIASES = {
    "SUPPORT_MODE": ("run", "mode"), "CHAT_MODEL": ("models", "chat"),
    "EMBEDDING_MODEL": ("models", "embedding"), "SUPPORT_DATA_DIR": ("paths", "data_dir"),
    "SUPPORT_INDEX_DIR": ("paths", "index_dir"), "BUDGET_CEILING_USD": ("budget", "ceiling_usd"),
}


def _is_optional(hint) -> bool:
    return typing.get_origin(hint) in (typing.Union, types.UnionType) and type(None) in typing.get_args(hint)


def _base(hint):
    if _is_optional(hint):
        return next(a for a in typing.get_args(hint) if a is not type(None))
    return hint


def _type_name(hint) -> str:
    base = _base(hint)
    return "a list" if typing.get_origin(base) is tuple else getattr(base, "__name__", str(base))


def _coerce(value: Any, hint, where: str) -> Any:
    if value is None:
        if _is_optional(hint):
            return None
        raise ConfigError(f"{where}: a value is required")
    base = _base(hint)

    def bad():
        raise ConfigError(f"{where}: expected {_type_name(hint)}, got {value!r}")
    if base is bool:
        return value if isinstance(value, bool) else bad()
    if base is int:
        return value if isinstance(value, int) and not isinstance(value, bool) else bad()
    if base is float:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else bad()
    if base is str:
        return value if isinstance(value, str) else bad()
    if base is Path:
        return Path(value) if isinstance(value, (str, Path)) else bad()
    if typing.get_origin(base) is tuple:
        if not isinstance(value, (list, tuple)):
            bad()
        element = typing.get_args(base)[0]
        return tuple(_coerce(v, element, where) for v in value)
    raise ConfigError(f"{where}: unsupported configuration type {base!r}")  # pragma: no cover


def _parse_text(raw: str, hint, where: str) -> Any:
    """Read a string from the environment or command line as the type the key expects."""
    if _is_optional(hint) and raw.strip().lower() in ("none", "null", ""):
        return None
    if _base(hint) in (str, Path):
        return raw
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        raise ConfigError(f"{where}: cannot read {raw!r} as {_type_name(hint)}") from None


def _suggest(word: str, options) -> str:
    close = difflib.get_close_matches(word, list(options), n=1)
    return f" Did you mean {close[0]!r}?" if close else f" Valid: {', '.join(sorted(options))}."


def _apply(layer: dict, origin: str, raw: dict, sources: dict, *, from_text: bool) -> None:
    for section, body in layer.items():
        if section not in SECTIONS:
            raise ConfigError(f"[{origin}] unknown section {section!r}." + _suggest(section, SECTIONS))
        if not isinstance(body, dict):
            raise ConfigError(f"[{origin}] {section!r} must be a table of key = value pairs")
        for key, value in body.items():
            where = f"{section}.{key} ({origin})"
            if key not in HINTS[section]:
                raise ConfigError(f"{where}: unknown key." + _suggest(key, HINTS[section]))
            hint = HINTS[section][key]
            if from_text and isinstance(value, str):
                value = _parse_text(value, hint, where)
            raw.setdefault(section, {})[key] = _coerce(value, hint, where)
            sources[f"{section}.{key}"] = origin


def _split_override(item: str) -> tuple[str, str, str]:
    if "=" not in item:
        raise ConfigError(f"--set expects section.key=value, got {item!r}")
    dotted, value = item.split("=", 1)
    parts = dotted.strip().split(".")
    if len(parts) != 2:
        raise ConfigError(f"--set expects section.key=value, got {item!r}")
    return parts[0], parts[1], value.strip()


def _read_layer(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if SECRET_PATTERN.search(text):
        raise ConfigError(f"{path} contains something that looks like an API key. Keys "
                          "belong in the environment, never in a config file")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML ({exc})") from None


def _check_profile(layer: dict, path: Path) -> None:
    for section, body in layer.items():
        if section not in PROFILE_ALLOWED:
            raise ConfigError(f"{path}: a pipeline profile may only set "
                              f"{sorted(PROFILE_ALLOWED)}; found [{section}]")
        allowed = PROFILE_ALLOWED[section]
        if allowed is not None and isinstance(body, dict):
            for key in body:
                if key not in allowed:
                    raise ConfigError(f"{path}: a pipeline profile may only set "
                                      f"{section}.{sorted(allowed)}; found {section}.{key}")


def _profile_path(name: str) -> Path | None:
    """The profile for a pipeline name: ``configs/pipelines/<name>.toml`` for the main pipelines, or
    the one file of that name under ``configs/pipelines/experiments/`` (grouped by what they test)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return None
    if (PROFILE_DIR / f"{name}.toml").is_file():
        return PROFILE_DIR / f"{name}.toml"
    found = sorted(PROFILE_DIR.glob(f"experiments/**/{name}.toml"))
    if len(found) > 1:
        raise ConfigError(f"pipeline {name!r} has more than one profile: {[str(p) for p in found]}")
    return found[0] if found else None


@dataclass(frozen=True)
class Settings:
    run: Run = field(default_factory=Run)
    models: Models = field(default_factory=Models)
    prompts: Prompts = field(default_factory=Prompts)
    paths: Paths = field(default_factory=Paths)
    budget: Budget = field(default_factory=Budget)
    chunking: Chunking = field(default_factory=Chunking)
    retrieval: Retrieval = field(default_factory=Retrieval)
    context: Context = field(default_factory=Context)
    generation: Generation = field(default_factory=Generation)
    limits: Limits = field(default_factory=Limits)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    web: Web = field(default_factory=Web)
    case: Case = field(default_factory=Case)
    sources: dict = field(default_factory=dict, compare=False, repr=False)
    config_files: tuple = field(default=(), compare=False, repr=False)
    profile: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        problems = self._problems()
        if problems:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))

    @property
    def mode(self) -> str:
        return self.run.mode

    @property
    def pool_size(self) -> int:
        """Candidates fetched before reranking."""
        return self.retrieval.candidate_k or self.retrieval.top_k

    @property
    def prompt_chunks(self) -> int:
        """Chunks allowed into the prompt. Follows top_k unless set explicitly."""
        return self.context.max_chunks or self.retrieval.top_k

    def _problems(self) -> list[str]:
        r, c, ch, g, b, e, w = (self.retrieval, self.context, self.chunking, self.generation,
                                self.budget, self.eval, self.web)
        checks = [
            (self.run.mode in MODES, f"run.mode must be one of {MODES}, got {self.run.mode!r}"),
            (self.case.review in REVIEW_POLICIES,
             f"case.review must be one of {REVIEW_POLICIES}, got {self.case.review!r}"),
            (r.top_k >= 1, "retrieval.top_k must be >= 1"),
            (1 <= r.max_queries <= 8, f"retrieval.max_queries ({r.max_queries}) must be between 1 and 8"),
            (r.rerank_snippet_chars >= 100, "retrieval.rerank_snippet_chars must be >= 100"),
            (r.rerank_snippet in ("start", "focused"),
             f"retrieval.rerank_snippet must be start or focused, got {r.rerank_snippet!r}"),
            (r.rerank_keep_docs is None or 0 <= r.rerank_keep_docs <= r.top_k,
             f"retrieval.rerank_keep_docs ({r.rerank_keep_docs}) must be between 0 and retrieval.top_k"),
            (r.section_candidates >= 1, "retrieval.section_candidates must be >= 1"),
            (r.min_chunks >= 1, "retrieval.min_chunks must be >= 1"),
            (r.expand in ("none", "page", "section", "branch"),
             f"retrieval.expand must be none, page, section or branch, got {r.expand!r}"),
            (r.expand_max_candidates >= 1, "retrieval.expand_max_candidates must be >= 1"),
            (0 <= r.corrective_turns <= 3, "retrieval.corrective_turns must be between 0 and 3"),
            (r.corrective_judge in ("relevance", "coverage"),
             f"retrieval.corrective_judge must be relevance or coverage, got {r.corrective_judge!r}"),
            (r.corrective_max_missing >= 1, "retrieval.corrective_max_missing must be >= 1"),
            (0 <= r.summary_sections <= 10, "retrieval.summary_sections must be between 0 and 10"),
            (1 <= r.hier_sections <= 10, "retrieval.hier_sections must be between 1 and 10"),
            (r.hier_places >= 0, "retrieval.hier_places must be >= 0 (a search never takes more than it returns)"),
            (r.cluster_size >= 1, "retrieval.cluster_size must be >= 1"),
            (r.complete in ("none", "neighbors", "section", "subsection", "trim", "section_trim"),
             f"retrieval.complete must be one of the completion modes, got {r.complete!r}"),
            (r.complete_max_tokens >= 1, "retrieval.complete_max_tokens must be >= 1"),
            (0.0 <= r.concept_weight <= 1.0, "retrieval.concept_weight must be between 0 and 1"),
            (all(v.strip() in ("none", "title", "code") for v in r.report_views.split(",")),
             f"retrieval.report_views must be none, or title and/or code (comma-separated), got {r.report_views!r}"),
            (0.0 <= r.view_weight <= 1.0, "retrieval.view_weight must be between 0 and 1"),
            (0.0 <= r.second_weight <= 1.0, "retrieval.second_weight must be between 0 and 1"),
            (r.pool_mode in ("fuse", "quota", "union"),
             f"retrieval.pool_mode must be fuse, quota or union, got {r.pool_mode!r}"),
            (min(r.union_report, r.union_per_concept) >= 0 and r.union_max >= max(1, r.union_report),
             "retrieval.union_* must be >= 0, and union_max at least union_report"),
            (min(r.quota_report_dense, r.quota_report_bm25, r.quota_concepts_dense) >= 0,
             "retrieval.quota_* must be >= 0"),
            (r.cluster_top >= 1, "retrieval.cluster_top must be >= 1"),
            (r.cluster_route in ("keywords", "embedding", "centroid"),
             f"retrieval.cluster_route must be keywords, embedding or centroid, got {r.cluster_route!r}"),
            (r.cluster_method in ("kmeans", "centered", "doc_anchored"),
             f"retrieval.cluster_method must be kmeans, centered or doc_anchored, got {r.cluster_method!r}"),
            (r.cluster_mode in ("filter", "boost"),
             f"retrieval.cluster_mode must be filter or boost, got {r.cluster_mode!r}"),
            (r.cluster_boost >= 0, "retrieval.cluster_boost must be >= 0"),
            (r.question_sections >= 0, "retrieval.question_sections must be >= 0"),
            (r.question_snippet in ("report", "question", "both"),
             f"retrieval.question_snippet must be report, question or both, got {r.question_snippet!r}"),
            (r.question_order in ("append", "interleave", "shuffle"),
             f"retrieval.question_order must be append, interleave or shuffle, got {r.question_order!r}"),
            (0 <= r.question_keep <= r.top_k, "retrieval.question_keep must be between 0 and retrieval.top_k"),
            (r.question_filter_top >= 0, "retrieval.question_filter_top must be >= 0"),
            (r.question_chunk in ("overlap", "question"),
             f"retrieval.question_chunk must be overlap or question, got {r.question_chunk!r}"),
            (r.rerank_mode in ("llm", "score", "rrf"), f"retrieval.rerank_mode must be llm, score or rrf, got {r.rerank_mode!r}"),
            (r.rerank_merge in ("rrf", "alternate", "split"),
             f"retrieval.rerank_merge must be rrf, alternate or split, got {r.rerank_merge!r}"),
            (r.rerank_split >= 0, "retrieval.rerank_split must be >= 0 (it is capped at retrieval.top_k)"),
            (r.rerank_merge_k >= 1 and r.bm25_extra >= 0, "retrieval.rerank_merge_k must be >= 1 and bm25_extra >= 0"),
            (r.rerank_fuse_weight >= 0 and r.rerank_fuse_k >= 1,
             "retrieval.rerank_fuse_weight must be >= 0 and rerank_fuse_k >= 1"),
            (r.class_route in ("llm", "centroid"), f"retrieval.class_route must be llm or centroid, got {r.class_route!r}"),
            (r.class_mode in ("filter", "boost"), f"retrieval.class_mode must be filter or boost, got {r.class_mode!r}"),
            (r.class_top >= 1 and r.class_boost >= 0, "retrieval.class_top must be >= 1 and class_boost >= 0"),
            (r.extract_mode in ("keep", "drop"), f"retrieval.extract_mode must be keep or drop, got {r.extract_mode!r}"),
            (r.extract_protect_top >= 0, "retrieval.extract_protect_top must be >= 0"),
            (0 <= r.rerank_doc_places <= r.top_k,
             f"retrieval.rerank_doc_places ({r.rerank_doc_places}) must be between 0 and retrieval.top_k"),
            (0 <= r.bm25_slots <= r.top_k,
             f"retrieval.bm25_slots ({r.bm25_slots}) must be between 0 and retrieval.top_k ({r.top_k})"),
            (0 <= r.doc_slots <= (r.candidate_k or r.top_k),
             (f"retrieval.doc_slots ({r.doc_slots}) must be between 0 and the candidates fetched "
              f"(retrieval.candidate_k, else retrieval.top_k: {r.candidate_k or r.top_k})")),
            (r.candidate_k is None or r.candidate_k >= r.top_k,
             f"retrieval.candidate_k ({r.candidate_k}) must be >= retrieval.top_k ({r.top_k})"),
            (r.embed_batch_size >= 1, "retrieval.embed_batch_size must be >= 1"),
            (r.bm25_k1 > 0, "retrieval.bm25_k1 must be > 0"),
            (0 <= r.bm25_b <= 1, "retrieval.bm25_b must be between 0 and 1"),
            (r.hybrid_pool >= r.top_k,
             f"retrieval.hybrid_pool ({r.hybrid_pool}) must be >= retrieval.top_k ({r.top_k})"),
            (r.rrf_k >= 1, "retrieval.rrf_k must be >= 1"),
            (1 <= r.max_loops <= 3,
             f"retrieval.max_loops ({r.max_loops}) must be between 1 and 3"),
            (c.max_chunks is None or 1 <= c.max_chunks <= r.top_k,
             (f"context.max_chunks ({c.max_chunks}) must be between 1 and retrieval.top_k "
              f"({r.top_k}): only top_k chunks are ever retrieved, so a larger value could never "
              "be filled. Raise retrieval.top_k instead")),
            (c.max_tokens >= 1, "context.max_tokens must be >= 1"),
            (c.max_query_chars >= 1 and c.max_body_chars >= 1, "context.max_*_chars must be >= 1"),
            (ch.max_tokens >= 1, "chunking.max_tokens must be >= 1"),
            (ch.context_header_scope in ("start", "all"),
             f"chunking.context_header_scope must be start or all, got {ch.context_header_scope!r}"),
            (0 <= ch.structure_min_chars < ch.structure_max_chars,
             "chunking.structure_min_chars must be >= 0 and smaller than chunking.structure_max_chars"),
            (1 <= ch.token_target <= ch.token_soft_max and 0 <= ch.token_min_section,
             "chunking.token_target must be >= 1 and <= token_soft_max, and token_min_section >= 0"),
            (ch.method in CHUNKING_METHODS,
             f"chunking.method {ch.method!r} is unknown; choose from {', '.join(CHUNKING_METHODS)}"),
            (0 <= ch.overlap_tokens < ch.max_tokens,
             (f"chunking.overlap_tokens ({ch.overlap_tokens}) must be >= 0 and < "
              f"chunking.max_tokens ({ch.max_tokens})")),
            (g.max_output_tokens >= 1, "generation.max_output_tokens must be >= 1"),
            (g.max_retries >= 0, "generation.max_retries must be >= 0"),
            (0 <= g.temperature <= 2, "generation.temperature must be between 0 and 2"),
            (g.timeout_seconds > 0, "generation.timeout_seconds must be > 0"),
            (b.ceiling_usd > 0, "budget.ceiling_usd must be > 0"),
            (0 <= b.reserve_usd < b.ceiling_usd,
             f"budget.reserve_usd ({b.reserve_usd}) must be >= 0 and < budget.ceiling_usd ({b.ceiling_usd})"),
            (b.max_live_calls >= 1, "budget.max_live_calls must be >= 1"),
            (self.limits.max_input_chars >= 1000, "limits.max_input_chars must be >= 1000"),
            (bool(self.pipeline.name) and bool(self.pipeline.retriever),
             "pipeline.name and pipeline.retriever must not be empty"),
            (self.pipeline.scope_gate in ("llm", "rules", "combined", "off"),
             f"pipeline.scope_gate must be llm, rules, combined or off, got {self.pipeline.scope_gate!r}"),
            (self.pipeline.controlled_version in ("v2", "v3", "v4"),
             f"pipeline.controlled_version must be v2, v3 or v4, got {self.pipeline.controlled_version!r}"),
            (self.pipeline.answer_model in ("gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini"),
             f"pipeline.answer_model must be gpt-5.4-mini, gpt-5-mini or gpt-4.1-mini, got {self.pipeline.answer_model!r}"),
            (self.pipeline.generator in ("single_call", "controlled_v2", "policy_v3"),
             f"pipeline.generator must be single_call, controlled_v2 or policy_v3, got {self.pipeline.generator!r}"),
            (len(e.case_files) >= 1, "eval.case_files must list at least one file"),
            (0 < e.test_fraction < 1, "eval.test_fraction must be between 0 and 1 (exclusive)"),
            (0 < e.passage_found_fraction <= 1, "eval.passage_found_fraction must be in (0, 1]"),
            (len(e.recall_ks) >= 1 and all(k >= 1 for k in e.recall_ks),
             "eval.recall_ks must be a non-empty list of integers >= 1"),
            (0 < e.leakage_overlap_threshold <= 1, "eval.leakage_overlap_threshold must be in (0, 1]"),
            (e.leakage_shingle_tokens >= 2, "eval.leakage_shingle_tokens must be >= 2"),
            (0 <= w.port <= 65535, "web.port must be between 0 and 65535"),
            (min(w.max_request_bytes, w.max_title_chars, w.max_body_chars, w.max_created_at_chars,
                 w.max_history_items, w.max_history_chars) >= 1, "web.max_* limits must be >= 1"),
        ]
        return [message for ok, message in checks if not ok]

    @classmethod
    def from_dict(cls, data: dict | None = None, *, sources: dict | None = None,
                  config_files: tuple = (), profile: str | None = None) -> Settings:
        """Build from a nested ``{section: {key: value}}`` dict. Unknown keys are errors."""
        raw: dict = {}
        origin_map = {f"{s}.{f.name}": "default" for s, k in SECTIONS.items() for f in fields(k)}
        origin_map.update(sources or {})
        _apply(data or {}, "given", raw, origin_map, from_text=False)
        built = {name: SECTIONS[name](**raw.get(name, {})) for name in SECTIONS}
        return cls(**built, sources=origin_map, config_files=tuple(config_files), profile=profile)

    @classmethod
    def load(cls, *, config_files=(), overrides=None, env=None) -> Settings:
        """defaults < default.toml < pipeline profile < $SUPPORTDESK_CONFIG < --config files
        < environment < overrides.

        The pipeline profile (``configs/pipelines/<pipeline.name>.toml``) is chosen by the pipeline
        the other layers select, so ``--pipeline advanced`` automatically brings advanced's settings.
        ``overrides`` is a list of ``"section.key=value"`` strings (the CLI) or a dict
        ``{"section.key": value}`` (code); both rank highest.
        """
        env = os.environ if env is None else env
        base = [DEFAULT_CONFIG_FILE] if DEFAULT_CONFIG_FILE.exists() else []
        extra = ([Path(env[CONFIG_ENV])] if env.get(CONFIG_ENV) else []) + [Path(f) for f in config_files]

        def resolve(profile: Path | None):
            files = [*base, *([profile] if profile else []), *extra]
            raw: dict = {}
            sources = {f"{s}.{f.name}": "default" for s, k in SECTIONS.items() for f in fields(k)}
            for path in files:
                layer = _read_layer(path)
                if path == profile:
                    _check_profile(layer, path)
                _apply(layer, f"file:{path}", raw, sources, from_text=False)
            for name, (section, key) in ALIASES.items():
                if env.get(name):
                    _apply({section: {key: env[name]}}, f"env:{name}", raw, sources, from_text=True)
            for name in sorted(env):
                if not name.startswith(ENV_PREFIX) or env[name] == "":
                    continue
                parts = name[len(ENV_PREFIX):].lower().split("__")
                if len(parts) != 2:
                    raise ConfigError(f"{name}: expected {ENV_PREFIX}SECTION__KEY")
                _apply({parts[0]: {parts[1]: env[name]}}, f"env:{name}", raw, sources, from_text=True)
            if isinstance(overrides, dict):
                for dotted, value in overrides.items():
                    section, key = dotted.split(".", 1)
                    _apply({section: {key: value}}, "override", raw, sources, from_text=True)
            else:
                for item in overrides or []:
                    section, key, value = _split_override(item)
                    _apply({section: {key: value}}, "cli", raw, sources, from_text=True)
            return raw, sources, files

        raw, sources, files = resolve(None)  # first pass: which pipeline do the other layers pick?
        profile = _profile_path(raw.get("pipeline", {}).get("name", PipelineCfg().name))
        if profile is not None:
            raw, sources, files = resolve(profile)
        built = {name: SECTIONS[name](**raw.get(name, {})) for name in SECTIONS}
        return cls(**built, sources=sources, config_files=tuple(str(f) for f in files),
                   profile=str(profile) if profile else None)

    def to_dict(self) -> dict:
        """The fully resolved configuration as JSON-safe nested data. Contains no secrets."""
        out: dict = {}
        for section, section_cls in SECTIONS.items():
            body = {}
            for f in fields(section_cls):
                value = getattr(getattr(self, section), f.name)
                body[f.name] = ([str(v) if isinstance(v, Path) else v for v in value] if isinstance(value, tuple)
                                else str(value) if isinstance(value, Path) else value)
            out[section] = body
        return out

    def own_retriever(self) -> str | None:
        """The retriever this pipeline's own definition sets (built-in defaults, configs/default.toml
        and its profile), ignoring everything layered on top: --retriever, --set, extra config files,
        $SUPPORTDESK_CONFIG and environment variables. None if the pipeline has no such definition."""
        try:
            return Settings.load(env={}, overrides=[f"pipeline.name={self.pipeline.name}"]).pipeline.retriever
        except ConfigError:
            return None

    def retriever_override(self) -> str | None:
        """The pipeline's own retriever when this configuration runs a different one, else None."""
        own = self.own_retriever()
        return own if own is not None and own != self.pipeline.retriever else None

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def rows(self):
        """(section, key, value, source, help) for every setting, for ``supportdesk config``."""
        for section, section_cls in SECTIONS.items():
            for f in fields(section_cls):
                dotted = f"{section}.{f.name}"
                yield (section, f.name, getattr(getattr(self, section), f.name),
                       self.sources.get(dotted, "default"), f.metadata.get("help", ""))

    def public(self) -> dict:
        """Safe to write into an artifact or a log: the API key is reduced to a boolean."""
        base = self.models.base_url or os.environ.get(BASE_URL_ENV)
        return {
            "mode": self.mode,
            "api_key_present": bool(os.environ.get(self.models.api_key_env)),
            "gateway_host": urlparse(base).hostname if base else None,
            "config": self.to_dict(),
            "config_sha256": self.fingerprint(),
            "config_sources": dict(self.sources),
            "config_files": list(self.config_files),
            "config_profile": self.profile,
        }
