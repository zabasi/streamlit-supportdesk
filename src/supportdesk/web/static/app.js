/* supportdesk web UI. No framework, no external requests.
 * Every piece of data (report text, evidence, model output) is inserted with textContent, never
 * innerHTML, so nothing in a report or a retrieved document can inject markup or script. */
"use strict";

const EXAMPLES = [
  { title: "Session state is empty after browser refresh",
    body: "I store the logged-in user in st.session_state. After pressing F5 the value is gone.\nStreamlit 1.31.1, Chrome.",
    history: "", created: "" },
  { title: "App stuck on loading on my server but works locally",
    body: "It opens fine on my machine, but on the server the page stays on the loading screen.\nReinstalling did not help.",
    history: "", created: "" },
];

const OUTCOMES = {
  answer: ["Answer", "ok"], request_info: ["More information needed", "warn"],
  escalate: ["Escalate to a maintainer", "warn"], error: ["Error", "bad"],
  budget_stopped: ["Stopped: budget", "bad"],
};

const $ = (id) => document.getElementById(id);

/** Build an element. children may be strings (as text), nodes, arrays, or null. */
function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "href") { if (/^https?:\/\//i.test(value)) { node.setAttribute("href", value);
      node.setAttribute("target", "_blank"); node.setAttribute("rel", "noopener noreferrer"); } }
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  const add = (child) => {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) child.forEach(add);
    else if (typeof child === "object") node.appendChild(child);
    else node.appendChild(document.createTextNode(String(child)));
  };
  children.forEach(add);
  return node;
}

const pill = (text, kind) => h("span", { class: "pill" + (kind ? " " + kind : "") }, text);
const usd = (n) => "$" + Number(n || 0).toFixed(6);

function showError(message) {
  const box = $("error");
  box.textContent = message || "";
  box.hidden = !message;
}

async function api(path, options) {
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch (_) { /* non-JSON error body */ }
  if (!response.ok) throw new Error((data && data.error) || `request failed (${response.status})`);
  return data;
}

function renderStatus(status) {
  const bar = $("status");
  bar.replaceChildren();
  const mode = status.settings.mode;
  bar.append(pill("mode: " + mode, mode === "live" ? "warn" : "ok"));
  bar.append(pill(`${status.index.chunk_count} chunks`));
  bar.append(pill(status.index.embedding_model));
  if (mode === "live") {
    bar.append(pill(`budget left ${usd(status.budget.remaining_usd)} of $${status.budget.ceiling_usd.toFixed(2)}`, "warn"));
  }
}

function fillSelect(select, values) {
  select.replaceChildren(...values.map((v) => h("option", { value: v }, v)));
}

let retrieverByPipeline = {};
let overrideOf = {};

// The retriever is part of the pipeline: shown, not chosen. Overrides are marked so a run is never
// mistaken for the pipeline as defined.
function showRetriever() {
  const name = $("pipeline").value;
  const own = overrideOf[name];
  $("retriever").textContent = (retrieverByPipeline[name] || "") +
    (own ? ` (override; ${name}'s own: ${own})` : "");
}

async function loadStatus() {
  try {
    const status = await api("/api/status");
    const keep = $("pipeline").value;
    fillSelect($("pipeline"), status.pipelines);
    retrieverByPipeline = status.retriever_by_pipeline || {};
    overrideOf = status.retriever_override_of || {};
    // The default pipeline comes from the server's configuration (never "whatever sorts first").
    $("pipeline").value = keep || (status.defaults || {}).pipeline || status.pipelines[0];
    showRetriever();
    if (status.limits) {  // input limits come from the server's configuration, not from this file
      $("title").maxLength = status.limits.title;
      $("body").maxLength = status.limits.body;
      $("created").maxLength = status.limits.created_at;
    }
    renderStatus(status);
  } catch (error) {
    $("status").replaceChildren(pill("server unreachable", "bad"));
    showError(error.message);
  }
}

function labelOf(evidence) {
  const map = {};
  evidence.forEach((e) => { map[e.chunk_id] = e.label; });
  return (chunkId) => map[chunkId] || chunkId;
}

function renderEvidence(evidence) {
  if (!evidence.length) return h("p", { class: "unknown" }, "No evidence was retrieved.");
  return evidence.map((e) => h("details", { class: e.cited ? "cited" : "" },
    h("summary", null, h("span", { class: "chip" }, e.label), " ",
      h("strong", null, e.title || e.source_id), " ",
      h("span", { class: "ev-meta" }, `${e.type}${e.score === null || e.score === undefined ? "" : " · score " + Number(e.score).toFixed(3)}`),
      e.cited ? [" ", pill("cited", "ok")] : null),
    h("div", { class: "inner" },
      h("div", { class: "ev-meta" },
        e.url ? h("a", { href: e.url }, e.url) : null,
        e.version ? ` · version ${e.version}` : null,
        e.state ? ` · state ${e.state}` : null,
        e.as_of ? ` · dated ${e.as_of}` : " · undated (documentation)"),
      h("pre", null, e.text))));
}

function renderSummary(result, evidence) {
  const s = result.maintainer_summary || {};
  if (!s.problem) return null;
  const label = labelOf(evidence);
  const known = Object.entries((s.environment || {}).known || {});
  const unknown = (s.environment || {}).unknown || [];
  return h("div", null,
    h("h3", null, "Maintainer summary"),
    h("dl", { class: "kv" },
      h("dt", null, "Problem"), h("dd", null, s.problem),
      h("dt", null, "Environment"), h("dd", null,
        known.length ? known.map(([k, v]) => h("div", null, `${k}: ${v}`)) : h("span", { class: "unknown" }, "nothing stated"),
        unknown.length ? h("div", { class: "unknown" }, "not stated: " + unknown.join(", ")) : null),
      h("dt", null, "Next step"), h("dd", null, s.next_step || h("span", { class: "unknown" }, "none given"))),
    h("h3", null, "Facts and where they come from"),
    (s.known_facts || []).length
      ? h("ul", { class: "plain" }, s.known_facts.map((f) => h("li", null, f.statement,
          f.evidence.length ? f.evidence.map((c) => h("span", { class: "chip" }, label(c)))
                            : h("span", { class: "chip" }, "no source"))))
      : h("p", { class: "unknown" }, "No supported facts."),
    h("h3", null, "Open questions"),
    (s.open_questions || []).length
      ? h("ul", { class: "plain" }, s.open_questions.map((q) => h("li", null, q)))
      : h("p", { class: "unknown" }, "None."),
    (s.sources || []).length ? [h("h3", null, "Sources used"),
      h("ul", { class: "plain" }, s.sources.map((src) => h("li", null,
        h("span", { class: "chip" }, label(src.chunk_id)), " ", src.type, " ",
        src.url ? h("a", { href: src.url }, src.url) : null)))] : null);
}

function renderResult(data) {
  const r = data.result, ev = data.evidence || [];
  const [text, kind] = OUTCOMES[r.outcome] || [r.outcome, ""];
  const u = r.usage || {};
  const root = $("result");
  // Everything goes through h(): replaceChildren() would render null as the text "null"
  // and an array as "[object ...]" in a real browser.
  root.replaceChildren(h("div", null,
    h("div", null, pill(text, kind), " ", pill(data.pipeline), " ", pill(data.retriever),
      " ", pill(r.mode),
      data.applied ? [" ", pill(`top_k ${data.applied.top_k}`), " ",
        pill(`${data.applied.prompt_chunks} chunks in prompt`)] : null),
    r.error ? h("div", { class: "banner bad" }, r.error) : null,
    (r.violations || []).length
      ? h("div", { class: "banner" }, "Grounding problems: " + r.violations.join(", ")) : null,
    r.user_reply ? [h("h3", null, "Reply to the reporter"), h("div", { class: "reply" }, r.user_reply)] : null,
    renderSummary(r, ev),
    h("h3", null, `Evidence shown to the model (${ev.length})`),
    renderEvidence(ev),
    h("div", { class: "usage" },
      h("span", null, "cost ", h("b", null, usd(u.cost_usd))),
      h("span", null, "upstream calls ", h("b", null, u.calls || 0)),
      h("span", null, "cache hits ", h("b", null, u.cache_hits || 0)),
      h("span", null, "tokens in/out ", h("b", null, `${u.input_tokens || 0}/${u.output_tokens || 0}`)),
      h("span", null, "time ", h("b", null, `${Number(u.wall_seconds || 0).toFixed(2)}s`)),
      u.replayed_cost_usd ? h("span", null, "replayed (not spent) ", h("b", null, usd(u.replayed_cost_usd))) : null),
    h("details", null, h("summary", null, "Raw record (for debugging and the report)"),
      h("div", { class: "inner" }, h("pre", null, JSON.stringify(r, null, 2))))));
  $("empty").hidden = true;
  root.hidden = false;
}

async function ask() {
  const button = $("go");
  showError("");
  const history = $("history").value.split("\n").map((l) => l.trim()).filter(Boolean);
  const payload = {
    title: $("title").value, body: $("body").value, history,
    created_at: $("created").value.trim() || null,
    pipeline: $("pipeline").value,
  };
  if (!payload.title.trim()) { showError("Please enter a title."); return; }
  button.disabled = true;
  button.firstChild.nodeValue = "Analyzing… ";
  try {
    const data = await api("/api/ask", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    renderResult(data);
    loadStatus();  // refresh the budget readout
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.firstChild.nodeValue = "Analyze ";
  }
}

function init() {
  $("go").addEventListener("click", ask);
  $("pipeline").addEventListener("change", showRetriever);  // each pipeline brings its own retriever
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); ask(); }
  });
  document.querySelectorAll("[data-example]").forEach((b) => b.addEventListener("click", () => {
    const ex = EXAMPLES[Number(b.dataset.example)];
    $("title").value = ex.title; $("body").value = ex.body;
    $("history").value = ex.history; $("created").value = ex.created;
  }));
  loadStatus();
}

if (typeof window !== "undefined") window.__supportdesk = { renderResult, h, init, loadStatus };
if (typeof document !== "undefined" && document.getElementById) init();
