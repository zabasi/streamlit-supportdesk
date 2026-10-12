"""Configuration: one typed, validated object; every tunable lives here, none elsewhere.

Precedence, lowest to highest: built-in defaults, ``configs/default.toml``, the pipeline profile
(``configs/pipelines/<pipeline.name>.toml``), ``$SUPPORTDESK_CONFIG``, each ``--config`` file,
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
    "pipeline": {"retriever"}, "paths": {"index_dir"},
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
CHUNKING_METHODS = ("fixed", "structure", "token_structure")


@dataclass(frozen=True)
class Run:
    mode: str = _opt("offline: free fake models; replay: recorded responses only; live: real model calls, which cost money", "offline")


@dataclass(frozen=True)
class Models:
    chat: str = _opt("the chat model's name at the provider; it must have a price in pricing.json", "gpt-4o-mini")
    embedding: str = _opt("the embedding model's name; changing it means building the index again", "text-embedding-3-small")
    base_url: str | None = _opt("the provider's URL; if unset, it is read from the OPENAI_BASE_URL environment variable")
    api_key_env: str = _opt("the name of the environment variable that holds the API key (never the key itself)", "OPENAI_API_KEY")


@dataclass(frozen=True)
class Prompts:
    system: Path = _opt("system prompt of the answer call", Path("configs/prompts/answer_system.txt"))
    output_format: Path = _opt("JSON shape the answer must follow", Path("configs/prompts/answer_output_format.txt"))
    rewrite: Path = _opt("system prompt of the query-rewriting call (QueryRewriter)",
                         Path("configs/prompts/rewrite_system.txt"))
    rerank: Path = _opt("system prompt of the reranking call (LLMReranker)", Path("configs/prompts/rerank_system.txt"))
    judge: Path = _opt("system prompt of the retrieve_loop sufficiency check (yes or no)",
                       Path("configs/prompts/judge_system.txt"))
    loop_rewrite: Path = _opt("system prompt of the retrieve_loop query rewrite",
                              Path("configs/prompts/loop_rewrite_system.txt"))


@dataclass(frozen=True)
class Paths:
    data_dir: Path = _opt("the folder that holds corpus.jsonl (its format is described in data/corpus.py)", Path("data/processed"))
    index_dir: Path = _opt("where the index is written (qdrant/ vector store, chunks.jsonl, manifest.json)", Path("index"))
    artifacts_dir: Path = _opt("where runs, reports, the tracker and the cache are written", Path("artifacts"))
    cache_path: Path = _opt("SQLite cache of model responses (makes re-runs free, enables replay)", Path("artifacts/cache.sqlite"))
    ledger_path: Path = _opt("record of every live call's cost; it counts toward the budget across runs", Path("artifacts/cost_ledger.jsonl"))
    pricing_path: Path = _opt("per-model prices used by the budget guard", Path("configs/pricing.json"))


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
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) and (PROFILE_DIR / f"{name}.toml").is_file():
        return PROFILE_DIR / f"{name}.toml"
    return None


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
