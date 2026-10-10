"""A local web page, served with Python's standard library only. Because it can spend money in live
mode, it is locked down:

- it only listens on this computer, and the mode is fixed when it starts;
- requests must come from the page itself, so other websites (and DNS rebinding) are refused;
- requests must be JSON, sizes are limited, and errors never reveal secrets."""
from __future__ import annotations

import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import urlparse

from ..config import ConfigError, Settings, redact
from ..llm.factory import Clients, build_clients
from ..registry import PIPELINES, RETRIEVERS, make_pipeline
from ..retrieval.filters import parse_ts
from ..retrieval.index import VectorIndex
from ..types import CaseInput

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
                 index: VectorIndex | None = None, settings_for=None):
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
        self._pipelines: dict = {}
        self.lock = threading.Lock()

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
            result = self._pipelines[key].run(case)
            budget = {"spent_usd": self.clients.ledger.total_cost_usd,
                      "remaining_usd": self.clients.ledger.remaining_usd()}
        return {"result": result.to_dict(), "evidence": self._evidence(result),
                "budget": budget, "pipeline": pipeline_name, "retriever": retriever_name,
                "applied": {"profile": settings.profile, "top_k": settings.retrieval.top_k,
                            "prompt_chunks": settings.prompt_chunks,
                            "candidate_k": settings.pool_size}}

    def _evidence(self, result) -> list[dict]:
        scores = {r["chunk_id"]: r["score"] for r in result.retrieved}
        out = []
        for item in result.trace.get("packet", []):
            chunk = self.chunks.get(item["chunk_id"])
            if chunk is None:
                continue
            out.append({"label": item["label"], "chunk_id": chunk.chunk_id,
                        "source_id": chunk.source_id, "type": chunk.source_type,
                        "title": chunk.title, "url": chunk.url, "version": chunk.version,
                        "state": chunk.meta.get("state"), "as_of": chunk.as_of,
                        "score": scores.get(chunk.chunk_id), "text": chunk.text,
                        "cited": chunk.chunk_id in result.citations})
        return out

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
        if not isinstance(history, list) or len(history) > self.limits["history_items"] or any(
                not isinstance(h, str) or len(h) > self.limits["history_chars"] for h in history):
            raise BadRequest(f"history must be a list of at most {self.limits['history_items']} "
                             f"strings of at most {self.limits['history_chars']} characters")
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
                         history=[{"role": "user", "body": h.strip()} for h in history if h.strip()])
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
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._guard():
                return
            if urlparse(self.path).path != "/api/ask":
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
                self._json(200, app.ask(payload))
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

