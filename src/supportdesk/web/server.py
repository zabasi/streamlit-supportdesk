"""A local web page, served with Python's standard library only. Because it can spend money in live
mode, it is locked down:

- it only listens on this computer, and the mode is fixed when it starts;
- requests must come from the page itself, so other websites (and DNS rebinding) are refused;
- requests must be JSON, sizes are limited, and errors never reveal secrets."""
from __future__ import annotations

import dataclasses
import json
import re
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from ..config import ConfigError, Settings, redact
from ..llm.factory import Clients, build_clients
from ..pipelines.stages.conversation import QUESTIONS_USED
from ..registry import PIPELINES, RETRIEVERS, make_pipeline
from ..retrieval.filters import parse_ts
from ..retrieval.index import VectorIndex
from ..types import CaseInput
from .conversations import (
    MAX_FOLLOWUPS,
    MAX_TURNS,
    OPEN,
    ConversationStore,
    now,
    report_html,
    summary_markdown,
)

LOOPBACK = {"localhost", "127.0.0.1", "[::1]"}
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
}


class BadRequest(ValueError):
    """The request is malformed or out of bounds (HTTP 400)."""


class App:
    """Everything a request needs, built once. Runs are serialised: the ledger and the SQLite cache
    are shared state, and the budget is not something to race on."""

    def __init__(self, settings: Settings, clients: Clients | None = None,
                 index: VectorIndex | None = None, settings_for=None, store_path=None):
        """``settings_for(pipeline_name)`` returns that pipeline's resolved settings (its profile);
        without it every pipeline uses ``settings``."""
        self.settings = settings
        self._settings_for = settings_for or (lambda name: settings)
        self._resolved: dict[str, Settings] = {}
        w = settings.web
        self.limits = {"title": w.max_title_chars, "body": w.max_body_chars,
                       "created_at": w.max_created_at_chars, "history_items": w.max_history_items,
                       "history_chars": w.max_history_chars}
        self.clients = clients or build_clients(settings)
        self.index = index or VectorIndex.load(settings.paths.index_dir,
                                               expect_model=self.clients.embedding_model,
                                               expect_chunking=settings.chunking)
        self.chunks = {c.chunk_id: c for c in self.index.chunks}
        self._pages = None  # documentation page texts, to rebuild completed and merged pieces
        self._pipelines: dict = {}
        self.lock = threading.Lock()
        # saved conversations: local, like the cache and the ledger
        self.store = ConversationStore(store_path or settings.paths.artifacts_dir / "web" / "conversations.sqlite")
        self._dev_cases = None

    def settings_for(self, pipeline: str) -> Settings:
        if pipeline not in self._resolved:
            self._resolved[pipeline] = self._settings_for(pipeline)
        return self._resolved[pipeline]

    def status(self) -> dict:
        m = self.index.manifest
        return {
            "settings": self.settings.public(),
            "pipelines": sorted(PIPELINES), "retrievers": sorted(RETRIEVERS),
            "defaults": {"pipeline": self.settings.pipeline.name,
                         "retriever": self.settings.pipeline.retriever},
            "retriever_by_pipeline": {name: self.settings_for(name).pipeline.retriever
                                      for name in sorted(PIPELINES)},
            # the pipeline's own retriever, where this server's configuration overrides it
            "retriever_override_of": {name: self.settings_for(name).retriever_override()
                                      for name in sorted(PIPELINES)},
            "index": {k: m.get(k) for k in ("chunk_count", "chunks_by_type", "embedding_model",
                                            "built_at", "exclusions")},
            "budget": {"spent_usd": self.clients.ledger.total_cost_usd,
                       "remaining_usd": self.clients.ledger.remaining_usd(),
                       "ceiling_usd": self.clients.ledger.ceiling_usd},
            "limits": self.limits,
        }

    def ask(self, payload: object) -> dict:
        case, pipeline_name, retriever_name = self._validate(payload)
        return self._run(case, pipeline_name, retriever_name)

    def _run(self, case: CaseInput, pipeline_name: str, retriever_name: str, *, questions_used=None,
             keep_result: bool = False) -> dict:
        settings = self.settings_for(pipeline_name)  # this pipeline's own profile
        built = self.index.manifest.get("chunking", {})
        if (built.get("max_tokens"), built.get("overlap_tokens")) != (
                settings.chunking.max_tokens, settings.chunking.overlap_tokens):
            raise BadRequest(
                f"pipeline {pipeline_name!r} is configured for chunking "
                f"{settings.chunking.max_tokens}/{settings.chunking.overlap_tokens} but the loaded index "
                f"was built with {built.get('max_tokens')}/{built.get('overlap_tokens')}. Index it with "
                f"`supportdesk index --pipeline {pipeline_name}` and start the UI with "
                f"`--pipeline {pipeline_name}`")
        key = (pipeline_name, retriever_name)
        with self.lock:
            if key not in self._pipelines:
                self._pipelines[key] = make_pipeline(pipeline_name, retriever_name, self.index,
                                                     self.clients, settings)
            token = QUESTIONS_USED.set(questions_used)  # only a conversation sets it
            try:
                result = self._pipelines[key].run(case)
            finally:
                QUESTIONS_USED.reset(token)
            budget = {"spent_usd": self.clients.ledger.total_cost_usd,
                      "remaining_usd": self.clients.ledger.remaining_usd()}
        return {"result": result.to_dict(), "evidence": self._evidence(result),
                "budget": budget, "pipeline": pipeline_name, "retriever": retriever_name,
                "applied": {"profile": settings.profile, "top_k": settings.retrieval.top_k,
                            "prompt_chunks": settings.prompt_chunks,
                            "candidate_k": settings.pool_size}}

    # --- development cases (inputs only: never rubrics, labels or gold evidence) ----------------
    def dev_cases(self) -> list[dict]:
        if self._dev_cases is None:
            try:
                from ..eval.cases import load_cases

                cases = load_cases(*self.settings.eval.case_files, split="dev")
            except Exception:  # noqa: BLE001 - no evaluation data on this computer: just none to offer
                cases = []
            self._dev_cases = {c.case_id: c.input for c in cases}
        return [{"id": cid, "title": inp.title} for cid, inp in self._dev_cases.items()]

    def dev_case(self, case_id: str) -> dict:
        self.dev_cases()
        inp = (self._dev_cases or {}).get(case_id)
        if inp is None:
            raise BadRequest("unknown development case")
        return {"id": case_id, "title": inp.title, "body": inp.body, "created_at": inp.created_at,
                "environment": dict(inp.environment or {}),
                "history": [{"role": h.get("role", "user"), "body": h.get("body", "")} for h in inp.history or []]}

    # --- conversations ------------------------------------------------------------------------
    def conversation(self, conv_id: str) -> dict:
        conv = self.store.get(conv_id or "")
        if conv is None:
            raise BadRequest("unknown conversation")
        return conv

    def conversation_new(self, payload: object) -> dict:
        """Create a conversation from a report; no model call."""
        case, pipeline, _ = self._validate(payload)
        env = payload.get("environment") or {}
        if not isinstance(env, dict) or len(env) > 20 or any(
                not isinstance(k, str) or not isinstance(v, str) or len(k) > 60 or len(v) > 200 for k, v in env.items()):
            raise BadRequest("environment must be an object of at most 20 short text fields")
        source = payload.get("source") or {}
        kind = source.get("kind") if isinstance(source, dict) else None
        if kind not in ("typed", "example", "dev"):
            kind = "typed"
        src = {"kind": kind}
        if kind == "dev":
            src["case_id"] = self.dev_case(str(source.get("case_id") or ""))["id"]
        report = {"title": case.title, "body": case.body, "created_at": case.created_at, "environment": env}
        return self.store.create(pipeline=pipeline, source=src, report=report, history=case.history)

    def conversation_turn(self, payload: object) -> dict:
        """One pipeline turn: with the user's reply when the assistant asked for information."""
        if not isinstance(payload, dict):
            raise BadRequest("body must be a JSON object")
        conv = self.conversation(str(payload.get("id") or ""))
        if conv["status"] not in OPEN:
            raise BadRequest(f"this conversation is finished ({conv['status']}); start a new one")
        if len(conv["turns"]) >= MAX_TURNS:
            conv["status"] = "limit_reached"
            self.store.save(conv)
            return conv
        if conv["status"] == "awaiting_reply":
            reply = payload.get("reply")
            if not isinstance(reply, str) or not reply.strip():
                raise BadRequest("write a reply to the assistant's question first")
            if len(reply) > self.limits["body"]:
                raise BadRequest(f"reply is longer than {self.limits['body']} characters")
            conv["messages"].append({"role": "user", "body": reply.strip(), "at": now()})
        r = conv["report"]
        case = CaseInput(case_id=f"web-{conv['id']}", title=r["title"], body=r["body"], created_at=r["created_at"],
                         environment=dict(r.get("environment") or {}),
                         history=[{"role": m["role"], "body": m["body"]} for m in conv["initial_history"] + conv["messages"]])
        retriever = self.settings_for(conv["pipeline"]).pipeline.retriever
        data = self._run(case, conv["pipeline"], retriever, questions_used=conv["questions_used"])
        result = data["result"]
        outcome = result.get("outcome")
        conv["turns"].append({"n": len(conv["turns"]) + 1, "at": now(), "outcome": outcome, "result": result,
                              "evidence": data["evidence"], "retriever": data["retriever"],
                              "applied": data.get("applied"), "usage": result.get("usage") or {}})
        reply_text = result.get("user_reply")
        if reply_text and outcome in ("answer", "request_info", "escalate"):
            conv["messages"].append({"role": "assistant", "body": reply_text, "at": now(), "outcome": outcome})
        if outcome == "request_info":
            conv["questions_used"] += 1
            conv["status"] = "awaiting_reply" if conv["questions_used"] < MAX_FOLLOWUPS else "limit_reached"
        elif outcome in ("answer", "escalate"):
            conv["status"] = outcome
        elif result.get("error") or outcome in ("error", "budget_stopped"):
            conv["status"] = "error"  # a turn that can be retried by hand
        else:
            conv["status"] = outcome or "error"  # for example out of scope: finished
        self.store.save(conv)
        return conv

    def conversation_delete(self, payload: object) -> dict:
        if not isinstance(payload, dict) or not self.store.delete(str(payload.get("id") or "")):
            raise BadRequest("unknown conversation")
        return {"deleted": payload["id"]}

    def _evidence(self, result) -> list[dict]:
        scores = {r["chunk_id"]: r["score"] for r in result.retrieved}
        out = []
        for item in result.trace.get("packet", []):
            chunk = self.chunks.get(item["chunk_id"]) or self._piece(item["chunk_id"])
            if chunk is None:
                continue
            out.append({"label": item["label"], "chunk_id": chunk.chunk_id,
                        "source_id": chunk.source_id, "type": chunk.source_type,
                        "title": chunk.title, "url": chunk.url, "version": chunk.version,
                        "state": chunk.meta.get("state"), "as_of": chunk.as_of,
                        "score": scores.get(chunk.chunk_id), "text": chunk.text,
                        "cited": chunk.chunk_id in result.citations,
                        # a whole section or merged neighbours, not a chunk of the index as it is
                        "completed": chunk.chunk_id not in self.chunks})
        return out

    def _piece(self, chunk_id: str):
        """A completed or merged piece (``source_id#start-end``): the same span of the page the pipeline
        cut it from, so the page shows exactly the text the model saw."""
        match = re.fullmatch(r"(.+)#(\d+)-(\d+)", chunk_id)
        if not match:
            return None
        source, a, b = match.group(1), int(match.group(2)), int(match.group(3))
        if self._pages is None:
            from ..pipelines.stages.sections import page_texts

            self._pages = page_texts(list(self.index.chunks))
        page = self._pages.get(source)
        base = next((c for c in self.index.chunks if c.source_id == source), None)
        if page is None or base is None or not 0 <= a < b <= len(page):
            return None
        return dataclasses.replace(base, chunk_id=chunk_id, text=page[a:b], start_char=a, end_char=b)

    def _validate(self, payload: object) -> tuple[CaseInput, str, str]:
        if not isinstance(payload, dict):
            raise BadRequest("body must be a JSON object")

        def text(name: str, *, required: bool) -> str:
            value = payload.get(name, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise BadRequest(f"{name} must be a string")
            if len(value) > self.limits[name]:
                raise BadRequest(f"{name} is longer than {self.limits[name]} characters")
            if required and not value.strip():
                raise BadRequest(f"{name} is required")
            return value.strip()

        title, body, created = text("title", required=True), text("body", required=False), \
            text("created_at", required=False)
        if created:
            try:
                parse_ts(created)
            except ValueError:
                raise BadRequest("created_at must be an ISO timestamp, for example 2024-05-01T10:00:00Z") \
                    from None
        history = payload.get("history", [])
        # earlier messages: plain strings (from the user), or {role: user | assistant, body} objects
        items = [h if isinstance(h, dict) else {"role": "user", "body": h} for h in history] \
            if isinstance(history, list) else None
        if items is None or len(items) > self.limits["history_items"] or any(
                not isinstance(h.get("body"), str) or len(h["body"]) > self.limits["history_chars"]
                or h.get("role", "user") not in ("user", "assistant") for h in items):
            raise BadRequest(f"history must be a list of at most {self.limits['history_items']} "
                             f"messages of at most {self.limits['history_chars']} characters")
        pipeline = payload.get("pipeline") or self.settings.pipeline.name
        if pipeline not in PIPELINES:
            raise BadRequest(f"unknown pipeline; choose from {sorted(PIPELINES)}")
        # The page runs each pipeline exactly as configured; swapping retrievers is an experiment,
        # done on the command line (--retriever), where the run is marked as an override.
        retriever = self.settings_for(pipeline).pipeline.retriever
        asked = payload.get("retriever")
        if asked not in (None, "", retriever):
            raise BadRequest(f"this page runs each pipeline with its own retriever ({pipeline}: "
                             f"{retriever}); to try another, use --retriever on the command line")
        case = CaseInput(case_id=f"web-{uuid.uuid4().hex[:8]}", title=title, body=body,
                         created_at=created or None,
                         history=[{"role": h.get("role", "user"), "body": h["body"].strip()}
                                  for h in items if h["body"].strip()])
        return case, pipeline, retriever


def _hostname(host_header: str) -> str:
    if host_header.startswith("["):  # IPv6 literal, for example [::1]:8765
        end = host_header.find("]")
        return host_header[: end + 1].lower() if end != -1 else ""
    return host_header.split(":")[0].lower()


def _host_ok(host_header: str | None) -> bool:
    return bool(host_header) and _hostname(host_header) in LOOPBACK


def _origin_ok(origin: str | None, host_header: str) -> bool:
    if origin is None:
        return True  # same-origin GETs and non-browser clients do not send one
    parsed = urlparse(origin)
    return parsed.scheme == "http" and parsed.netloc.lower() == host_header.lower()


GET_ROUTES = {
    "/api/cases": lambda app, q: ("json", {"cases": app.dev_cases()}),
    "/api/case": lambda app, q: ("json", app.dev_case(q.get("id", ""))),
    "/api/conversations": lambda app, q: ("json", {"conversations": app.store.listing(),
                                                   "limits": {"followups": MAX_FOLLOWUPS, "turns": MAX_TURNS}}),
    "/api/conversation": lambda app, q: ("json", app.conversation(q.get("id", ""))),
    "/api/conversation/summary": lambda app, q: ("text/markdown; charset=utf-8",
                                                 summary_markdown(app.conversation(q.get("id", "")))),
    "/api/report": lambda app, q: ("text/html; charset=utf-8", report_html(app.store.all())),
}
POST_ROUTES = {
    "/api/ask": "ask",  # method names, looked up on the app (so it stays the single source of truth)
    "/api/conversations/new": "conversation_new",
    "/api/conversations/turn": "conversation_turn",
    "/api/conversations/delete": "conversation_delete",
}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "supportdesk"
        sys_version = ""

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _guard(self) -> bool:
            host = self.headers.get("Host")
            if not _host_ok(host):
                self._json(403, {"error": "forbidden host"})
                return False
            if not _origin_ok(self.headers.get("Origin"), host):
                self._json(403, {"error": "forbidden origin"})
                return False
            return True

        def do_GET(self) -> None:
            if not self._guard():
                return
            path = urlparse(self.path).path
            if path in STATIC:
                name, ctype = STATIC[path]
                data = resources.files("supportdesk.web").joinpath("static", name).read_bytes()
                self._send(200, data, ctype)
            elif path == "/api/status":
                self._json(200, app.status())
            elif path in GET_ROUTES:
                query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                try:
                    kind, value = GET_ROUTES[path](app, query)
                except BadRequest as exc:
                    return self._json(400, {"error": redact(exc)})
                if kind == "json":
                    self._json(200, value)
                else:
                    self._send(200, value.encode("utf-8"), kind)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._guard():
                return
            action = POST_ROUTES.get(urlparse(self.path).path)
            if action is None:
                return self._json(404, {"error": "not found"})
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "Content-Type must be application/json"})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0:
                return self._json(411, {"error": "Content-Length required"})
            limit = app.settings.web.max_request_bytes
            if length > limit:
                return self._json(413, {"error": f"request larger than {limit} bytes"})
            try:
                payload = json.loads(self.rfile.read(length) or b"null")
                self._json(200, getattr(app, action)(payload))
            except (BadRequest, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._json(400, {"error": redact(exc) if isinstance(exc, BadRequest)
                                 else "body is not valid JSON"})
            except ConfigError as exc:
                self._json(400, {"error": redact(exc)})
            except Exception as exc:  # noqa: BLE001 - last line of defence; never leak internals
                print(f"internal error: {type(exc).__name__}: {redact(exc)}", file=sys.stderr)
                self._json(500, {"error": "internal error; see the server log"})

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write(f"{self.address_string()} - {redact(fmt % args)}\n")

    return Handler


def create_server(app: App, *, host: str = "127.0.0.1", port: int | None = None) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ConfigError("the web UI only binds to loopback; it can spend money and has no login")
    server = ThreadingHTTPServer((host, app.settings.web.port if port is None else port),
                                 make_handler(app))
    server.daemon_threads = True
    return server


def serve(settings: Settings, *, host: str = "127.0.0.1", port: int | None = None,
          settings_for=None) -> int:
    app = App(settings, settings_for=settings_for)
    server = create_server(app, host=host, port=port)
    bound = server.server_address[1]
    print(f"supportdesk web UI  http://{host}:{bound}   mode={settings.mode}   (Ctrl+C to stop)")
    if settings.mode == "live":
        print(f"LIVE mode: real calls, ${app.clients.ledger.remaining_usd():.2f} of "
              f"${app.clients.ledger.ceiling_usd:.2f} remaining")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0

