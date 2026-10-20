# Grounded generation on hier-class retrieval

> Archived results of the original runs are in `docs/experiments/hier_class/generation-results/`. Running the scripts again writes new results to `artifacts/hier_class_rag/generation/` (not committed).

This is the layer **above** the hierarchical-class retrieval experiment. Retrieval finds
passages. Generation must answer only from that packet, cite it, and **refuse** when the
check fails.

It follows a standard grounded-generation design (cite, validate, refuse when the check fails), adapted
to this Streamlit corpus and the annotated dev gold. Retrieval stays frozen: each method reuses its stored packet. Only the answer
call is new.

Stay on `--split dev`. Do not commit `.env`.

A second generator, **policy v3**, writes a support record with literal report quotes
instead of claim-support refusal. See `docs/experiments/policy_v3/README.md`. Do not mix
its 9/15 gold-passage score with the packet numbers below.

```powershell
.\.venv\Scripts\Activate.ps1
# extractive stand-in (free, same metrics code, not a real model score)
python scripts/hier_class_generate.py --mode offline
# real answers (gpt-4o-mini)
python scripts/hier_class_generate.py --mode live
python scripts/hier_class_generate.py --mode live --methods advanced,llm_llm
```

Tables: `docs/experiments/hier_class/generation-results/comparison.md` and `results.json`.
Per-method turns: `docs/experiments/hier_class/generation-results/<method>/turns.jsonl`.

## What one turn does

```text
stored retrieval packet
  → assemble evidence (S1…, chunk ids intact)
  → ask (JSON schema + <question> / <evidence>)
  → parse (never repair)
  → validate citations, claim support, instruction-echo
  → deliver the answer, or refuse with a typed reason
```

Guiding rule: **a grounded assistant is not one that cites; it is one that can
be checked, and refuses when the check fails.**

Two limits of this design:

- The model sees only this turn's packet. Instructions live in the system message. The
  packet is data.
- At most **one retry**, and only for a provider error or non-JSON. A validation failure
  is never retried, and a rejected payload is never shown (`answer` is null).

Offline mode uses an extractive stand-in (quote a packet sentence that shares
query words). It does not reason. Live mode uses `gpt-4o-mini` through the existing
gateway.

## Default methods

| method | why it is in the default set |
|---|---|
| `advanced` | current system retrieve (best recall@6 besides one ablation cell) |
| `llm_llm` | winner of the simple 2×2 |
| `advanced_rule_kmeans` | only 3×3 cell that beat `advanced` on recall@6 |
| `advanced_llmchunk_llmfilter` | the combo you asked to attach to advanced |

`--methods` can list any finished retrieval run.

## Official support outcomes

The model must pick **exactly one**:

| outcome | gold when the case is | meaning |
|---|---|---|
| `answer` | `answerable` | packet supports concrete guidance |
| `request_info` | `ambiguous` | one missing fact would tell the causes apart |
| `escalate` | `escalation` | a maintainer must look |

`outcome accuracy` is the share of cases whose delivered decision is in the eval
file's `acceptable_outcomes`. A `validation_failed` turn never counts as a match:
the checker blocked an ungrounded `answer`.

The **confusion matrix** (gold × predicted) is in
`docs/experiments/hier_class/generation-results/comparison.md` under each method. Rows are
what gold wanted; columns are what the turn delivered. The diagonal is correct.

## Metrics (how you can tell if it works)

These are standard grounded-generation measures, mapped onto selected_spans gold. There is no
fact index here, so two of them are defined on this gold instead of aliases.

| measure | what it asks | usual bar | good on this project if |
|---|---|---:|---|
| **claim support rate** | of claims that assert a fact, how many are supported by the *cited* packet text (content-word overlap ≥ 0.5) | ≥ 0.90 | close to 1.0; a drop means the model wrote words the citation does not contain |
| **citation precision** | of cited ids, how many are in this packet | ≥ 0.95 | 1.0 |
| **citations outside packet** | invented ids | 0 | 0 |
| **answered after failure** | delivered an answer after validation failed | 0 | 0 |
| **abstention P / R / F1** | should refuse when the packet has **no gold source**; should answer when it does | F1 ≥ 0.85 | F1 high. Precision-only means it refused everything; recall-only means it answered everything |
| **answer when packet had gold** | of cases where retrieval actually put gold in the packet, how many produced a delivered answer | — | this is the generation layer, not recall@6 |
| **completeness vs gold spans** | of reviewer-selected words, how many appear in the delivered answer | — | higher is better; retrieval can be fine and this still low if the model hedges |
| **answers cite gold** | a delivered answer cites a chunk from a gold source | — | high means citations point at the reviewer pages, not only at decoys |
| **instruction echo** | answer repeats a packet instruction ("ignore previous instructions", "system prompt", …) | pass rate 1.0 | must stay 0 |

`stale_answers` is reported as 0: this corpus does not mark `superseded` versions.

### A numerical example

Suppose the packet is three chunks, `S1` is gold (`session_state.md`), `S2` and `S3` are not.

1. The model returns two claims. Claim 1 quotes `S1` ("Session state keeps widget values")
   → supported. Claim 2 says "refunds take 14 days" and cites `S1` → **not supported**.
   Claim support = 1/2 = **0.50**.
2. Citations `["S1", "S9"]`. `S9` is not in the packet → citation precision 1/2, outside
   packet = 1. The turn is **refused** (`validation_failed`). `answered after failure` stays
   0 because the customer never sees that text.
3. Packet has gold, so the turn *should* have answered. A validation failure here is a
   **false refusal** (hurts abstention precision? no — it refused when it should have
   answered, so it is a false positive for abstention / hurts answer-when-gold).
4. Completeness: gold span text is "Session state keeps widget values". The delivered
   answer (if any) must use those words. If the turn was refused, completeness is not
   counted for that case.

## Links

- What “grounded” means: <https://www.pinecone.io/learn/series/rag/rag-evaluation/>
- Why a citation is not enough (RARR): <https://arxiv.org/abs/2305.14627>
- Structured JSON: <https://platform.openai.com/docs/guides/structured-outputs>
- Prompt injection / packet is data: <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- Precision / recall / F1: <https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)>


## Live numbers (15 annotated dev cases, `gpt-4o-mini`)

Same packets as the retrieval experiment. The model now returns `answer`, `request_info`,
or `escalate`. Gold: 6 answerable, 6 ambiguous, 3 escalation.

| method | answer | request_info | escalate | validation_failed | outcome accuracy | answerable (n=6) | ambiguous (n=6) | escalation (n=3) | claim support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `advanced` | 7 | 5 | 3 | 0 | **0.40** | 0.50 | 0.33 | 0.33 | 1.00 |
| `llm_llm` | 3 | 5 | 6 | 1 | 0.27 | 0.17 | 0.17 | **0.67** | 0.75 |
| `advanced_rule_kmeans` | 3 | 3 | 6 | 3 | 0.27 | 0.17 | 0.17 | **0.67** | 0.60 |
| `advanced_llmchunk_llmfilter` | 7 | 3 | 4 | 1 | 0.27 | **0.67** | 0.00 | 0.00 | 0.87 |

What that means:

- **Decision quality is the weak point**, not citations. Best overall match to gold is
  `advanced` at 0.40. The model often `answer`s when gold wanted `request_info`, or
  `escalate`s an answerable case.
- **`advanced_llmchunk_llmfilter` is best on answerable cases (0.67)** but never hit
  `request_info` or `escalate` when gold required them (0.00 / 0.00). It is willing to
  give guidance; it does not know when to stop.
- **`llm_llm` and `advanced_rule_kmeans` over-escalate** (6 escalations, 2 of 3 gold
  escalation cases correct) and almost never answer the answerable ones.
- **When it does `answer`, citations stay clean** (precision 1.0). Claim support is 1.00
  on `advanced` and 0.87 on `advanced_llmchunk_llmfilter`.
- Open `docs/experiments/hier_class/generation-results/comparison.md` for the table, and each
  `turns.jsonl` for per-case `outcome` vs `expected_outcomes`.
