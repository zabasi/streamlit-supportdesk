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
      e.completed ? [" ", pill("whole section")] : null,
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
        pill(`up to ${data.applied.prompt_chunks} pieces`)] : null),
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

/* ---------- conversations ---------- */

const STATUS = {
  new: ["Ready for the first turn", ""], awaiting_reply: ["The assistant asked a question: your reply", "warn"],
  answer: ["Answered", "ok"], escalate: ["Escalation proposed", "warn"],
  limit_reached: ["Clarification limit reached", "bad"], error: ["The last turn failed: retry by hand", "bad"],
};
let conv = null;        // the conversation on screen
let devHistory = null;  // a development case's earlier messages, with their roles
let devEnv = {};        // and its reported environment
let limits = { followups: 5, turns: 6 };
let busy = false;
let picked = "";        // the saved conversation the user chose in the list (the list never changes it by itself)

const post = (path, body) => api(path, { method: "POST",
  headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function setBusy(value, label) {
  busy = value;
  ["go", "send", "retry", "delete", "saved", "source"].forEach((id) => { $(id).disabled = value; });
  if (!value) $("delete").disabled = !conv;
  if (label) $("go").firstChild.nodeValue = label;
}

async function loadCases() {
  const select = $("source");
  const options = [h("option", { value: "typed" }, "New question (type it below)"),
    h("optgroup", { label: "Examples" }, EXAMPLES.map((ex, i) => h("option", { value: "example:" + i }, ex.title)))];
  try {
    const data = await api("/api/cases");
    for (const [split, label] of [["dev", "Development cases"], ["test", "Test cases"]]) {
      const group = data.cases.filter((c) => (c.split || "dev") === split);
      if (group.length) {
        options.push(h("optgroup", { label }, group.map((c) => h("option", { value: "dev:" + c.id }, `${c.id} — ${c.title}`))));
      }
    }
  } catch (_) { /* no evaluation cases on this computer */ }
  select.replaceChildren(...options);
}

async function loadSaved() {
  const data = await api("/api/conversations");
  limits = data.limits || limits;

  $("saved").replaceChildren(h("option", { value: "" }, "— choose a saved conversation —"),
    ...data.conversations.map((c) => h("option", { value: c.id },
      `${c.title} · ${(STATUS[c.status] || [c.status])[0]} · ${c.pipeline} · ` +
      (c.source && c.source.run_id ? `from run ${c.source.run_id} · ` : "") +
      `${new Date(c.updated_at).toLocaleString()}`)));
  if ($("follow").checked) {  // ticked: select, and open, the latest conversation
    const latest = data.conversations.length ? data.conversations[0].id : "";
    $("saved").value = latest;
    picked = latest;
    if (latest && (!conv || conv.id !== latest)) await resume();
  } else {
    $("saved").value = data.conversations.some((c) => c.id === picked) ? picked : "";
  }
}

async function chooseSource() {
  const value = $("source").value;
  devHistory = null; devEnv = {};
  $("envline").hidden = true;
  if (value === "typed") {
    ["title", "body", "history", "created"].forEach((id) => { $(id).value = ""; });
  } else if (value.startsWith("example:")) {
    const ex = EXAMPLES[Number(value.slice(8))];
    $("title").value = ex.title; $("body").value = ex.body;
    $("history").value = ex.history; $("created").value = ex.created;
  } else if (value.startsWith("dev:")) {
    try {
      const c = await api("/api/case?id=" + encodeURIComponent(value.slice(4)));
      $("title").value = c.title; $("body").value = c.body; $("created").value = c.created_at || "";
      devHistory = c.history; devEnv = c.environment || {};
      $("history").value = c.history.map((m) => `${m.role}: ${m.body}`).join("\n");
      const env = Object.entries(devEnv);
      $("envline").textContent = env.length ? "Reported environment: " + env.map(([k, v]) => `${k}=${v}`).join(", ")
                                            : "No reported environment.";
      $("envline").hidden = false;
    } catch (error) { showError(error.message); }
  }
}

function historyItems() {
  // a development case keeps its messages with their roles unless the text was edited
  const typed = $("history").value.split("\n").map((l) => l.trim()).filter(Boolean);
  if (devHistory && typed.join("\n") === devHistory.map((m) => `${m.role}: ${m.body}`).join("\n")) return devHistory;
  return typed;
}

function renderConversation(c) {
  conv = c;
  const [label, kind] = STATUS[c.status] || [c.status, ""];
  const lastTurn = c.turns[c.turns.length - 1];
  // replaceChildren() takes nodes only: an array or null would be shown as text
  const badges = [pill(label, kind), " ", pill(c.pipeline), " ",
    pill(`questions ${c.questions_used} of ${limits.followups}`), " ", pill(`turns ${c.turns.length} of ${limits.turns}`)];
  if (c.source && c.source.case_id) badges.push(" ", pill(c.source.case_id));
  if (c.label && c.label.expected) {  // the evaluation label: shown here only, never sent to the model
    badges.push(" ", pill(`label: ${c.label.category_name} → ${c.label.expected}`));
    if (lastTurn) {
      const match = lastTurn.outcome === c.label.expected;
      badges.push(" ", pill(`decided: ${lastTurn.outcome} ${match ? "✓" : "✗"}`, match ? "ok" : "bad"));
    }
  }
  $("convhead").replaceChildren(...badges);
  const r = c.report;
  $("thread").replaceChildren(
    // the report collapses, so the assistant's messages and the replies stay in view
    h("details", { class: "msg report", open: !c.turns.length },
      h("summary", null, h("strong", null, r.title), " ", h("span", { class: "hint" }, "the report")),
      r.body ? h("div", null, r.body) : null,
      Object.keys(r.environment || {}).length
        ? h("div", { class: "ev-meta" }, Object.entries(r.environment).map(([k, v]) => `${k}=${v}`).join(" · ")) : null),
    ...c.initial_history.concat(c.messages).map((m) => h("article", { class: "msg " + m.role },
      h("strong", null, m.role === "assistant" ? "Assistant" : "User"),
      m.outcome ? [" ", pill((OUTCOMES[m.outcome] || [m.outcome])[0], (OUTCOMES[m.outcome] || [])[1])] : null,
      h("div", null, m.body))));
  $("replybox").hidden = c.status !== "awaiting_reply";
  $("retry").hidden = c.status !== "error";
  $("convo").hidden = false;
  $("empty").hidden = true;
  $("delete").disabled = false;
  if (lastTurn) renderResult({ result: lastTurn.result, evidence: lastTurn.evidence, pipeline: c.pipeline,
                               retriever: lastTurn.retriever, applied: lastTurn.applied });
  else { $("result").hidden = true; }
}

async function runTurn(reply) {
  if (busy || !conv) return;
  showError("");
  setBusy(true);
  try {
    renderConversation(await post("/api/conversations/turn", { id: conv.id, reply }));
    $("reply").value = "";
  } catch (error) {
    showError(error.message);
    try { renderConversation(await api("/api/conversation?id=" + encodeURIComponent(conv.id))); } catch (_) { /* gone */ }
  } finally {
    setBusy(false);
    loadSaved().catch(() => {});
    loadStatus();  // refresh the budget readout
  }
}

async function start() {
  if (busy) return;
  showError("");
  const value = $("source").value;
  const payload = {
    title: $("title").value, body: $("body").value, history: historyItems(),
    created_at: $("created").value.trim() || null, pipeline: $("pipeline").value,
    environment: value.startsWith("dev:") ? devEnv : {},
    source: value.startsWith("dev:") ? { kind: "dev", case_id: value.slice(4) }
          : { kind: value.startsWith("example:") ? "example" : "typed" },
  };
  if (!payload.title.trim()) { showError("Please enter a title."); return; }
  setBusy(true, "Starting… ");
  picked = "";  // a new conversation is not one the user picked in the list
  try {
    renderConversation(await post("/api/conversations/new", payload));
  } catch (error) {
    showError(error.message);
    setBusy(false, "Start conversation ");
    return;
  }
  setBusy(false, "Start conversation ");
  await runTurn("");
}

async function resume() {
  const id = $("saved").value;
  picked = id;
  if (!id) return;
  showError("");
  try { renderConversation(await api("/api/conversation?id=" + encodeURIComponent(id))); }
  catch (error) { showError(error.message); }
}

async function remove() {
  if (!conv || busy || !window.confirm(`Delete the saved conversation “${conv.report.title}”?`)) return;
  try {
    await post("/api/conversations/delete", { id: conv.id });
    conv = null;
    picked = "";
    $("convo").hidden = true; $("result").hidden = true; $("empty").hidden = false;
    $("delete").disabled = true;
    await loadSaved();
  } catch (error) { showError(error.message); }
}

async function loadRuns() {
  try {
    const data = await api("/api/runs");
    $("runs").replaceChildren(h("option", { value: "" }, "— choose a finished run —"),
      ...data.runs.map((r) => h("option", { value: r.run_id },
        `${r.run_id} · ${r.pipeline} · ${r.mode}` + (r.cases ? ` · ${r.cases} cases` : ""))));
  } catch (_) { /* no runs on this computer */ }
}

async function importRun() {
  const runId = $("runs").value;
  if (!runId || busy) return;
  showError("");
  $("importrun").disabled = true;
  try {
    const r = await post("/api/runs/import", { run_id: runId });
    $("importmsg").textContent = `${r.imported} case(s) imported` +
      (r.already_imported ? `, ${r.already_imported} already imported` : "") +
      (r.without_case_input ? `, ${r.without_case_input} without case input` : "") + ".";
    await loadSaved();  // the list grows; what is open, or not, stays as it was
  } catch (error) { showError(error.message); }
  finally { $("importrun").disabled = !$("runs").value; }
}

function download(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = h("a", null);
  link.setAttribute("href", url); link.setAttribute("download", name);
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadSummary() {
  if (!conv) return;
  try {
    const response = await fetch("/api/conversation/summary?id=" + encodeURIComponent(conv.id));
    if (!response.ok) throw new Error(`request failed (${response.status})`);
    download(await response.text(), `supportdesk-${conv.id}-summary.md`, "text/markdown");
  } catch (error) { showError(error.message); }
}

function init() {
  $("go").addEventListener("click", start);
  $("send").addEventListener("click", () => {
    if (!$("reply").value.trim()) { showError("Write your reply first."); return; }
    runTurn($("reply").value);
  });
  $("retry").addEventListener("click", () => runTurn(""));
  $("saved").addEventListener("change", resume);
  $("delete").addEventListener("click", remove);
  $("source").addEventListener("change", chooseSource);
  $("follow").addEventListener("change", () => loadSaved().catch((error) => showError(error.message)));
  $("runs").addEventListener("change", () => { $("importrun").disabled = !$("runs").value; $("importmsg").textContent = ""; });
  $("importrun").addEventListener("click", importRun);
  $("dljson").addEventListener("click", () => conv && download(JSON.stringify(conv, null, 2),
    `supportdesk-${conv.id}.json`, "application/json"));
  $("dlsum").addEventListener("click", downloadSummary);
  $("pipeline").addEventListener("change", showRetriever);  // each pipeline brings its own retriever
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (conv && conv.status === "awaiting_reply" && document.activeElement === $("reply")) $("send").click();
      else start();
    }
  });
  loadStatus();
  loadCases();
  loadRuns();
  loadSaved().catch((error) => showError(error.message));
}

if (typeof window !== "undefined") window.__supportdesk = { renderResult, renderConversation, h, init, loadStatus };
if (typeof document !== "undefined" && document.getElementById) init();
