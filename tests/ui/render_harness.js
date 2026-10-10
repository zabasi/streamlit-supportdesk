// Runs the real app.js against a minimal stub DOM.
//
//   node render_harness.js app.js response.json            render a /api/ask response, print what a person would read
//   node render_harness.js app.js status.json status NAME  feed /api/status to loadStatus(), then switch the
//                                                          pipeline select to NAME; print the selected options
const fs = require("fs");
const [appPath, dataPath, mode, switchTo] = process.argv.slice(2);

class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.className = "";
    this.hidden = false; this.value = ""; this._text = ""; this.dataset = {}; this.listeners = {}; }
  appendChild(c) { this.children.push(c); return c; }
  // Like a real DOM: anything that is not a Node is converted to a string and becomes text.
  _coerce(c) { return c instanceof Node || c instanceof Text ? c : new Text(String(c)); }
  append(...cs) { cs.forEach((c) => this.children.push(this._coerce(c))); }
  replaceChildren(...cs) { this.children = cs.map((c) => this._coerce(c)); }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(t, fn) { this.listeners[t] = fn; }
  get firstChild() { return this.children[0] || { nodeValue: "" }; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
}
class Text { constructor(t) { this.nodeValue = String(t); } get textContent() { return this.nodeValue; } }

const registry = {};
const get = (id) => (registry[id] = registry[id] || new Node("#" + id));
const data = JSON.parse(fs.readFileSync(dataPath, "utf8"));

global.window = {};
global.document = { createElement: (t) => new Node(t), createTextNode: (t) => new Text(t),
  getElementById: get, addEventListener() {}, querySelectorAll: () => [] };
global.fetch = async () => ({ ok: true, json: async () => ({ settings: { mode: "offline" },
  index: { chunk_count: 1, embedding_model: "m" }, budget: {}, pipelines: ["baseline"], retrievers: ["dense"] }) });
// Fail loudly if app.js ever writes HTML.
Object.defineProperty(Node.prototype, "innerHTML", { set() { throw new Error("innerHTML used"); } });

eval(fs.readFileSync(appPath, "utf8"));

async function statusMode() {
  await new Promise((resolve) => setTimeout(resolve, 0));  // let the page's own startup load finish
  get("pipeline").value = "";                              // then behave like a freshly opened page
  global.fetch = async () => ({ ok: true, json: async () => data });
  await window.__supportdesk.loadStatus();
  const first = { pipeline: get("pipeline").value, retriever: get("retriever").textContent };
  get("pipeline").value = switchTo;          // the user picks another pipeline
  get("pipeline").listeners.change();
  console.log(JSON.stringify({ first, afterSwitch: get("retriever").textContent }));
}

function renderMode() {
  window.__supportdesk.renderResult(data);
  const text = get("result").textContent;
  // Text a person reads, excluding the raw JSON record (which legitimately contains `null`).
  const visibleText = (n) => (n.tag === "details" && n.textContent.startsWith("Raw record")) ? ""
    : (n._text || "") + (n.nodeValue || "") + (n.children || []).map(visibleText).join("");
  const visible = visibleText(get("result"));
  const hrefs = [];
  (function walk(n) { if (n.attrs && n.attrs.href) hrefs.push(n.attrs.href); (n.children || []).forEach(walk); })(get("result"));
  console.log(JSON.stringify({ text, visible, hrefs, hidden: get("result").hidden, emptyHidden: get("empty").hidden }));
}

if (mode === "status") statusMode(); else renderMode();
