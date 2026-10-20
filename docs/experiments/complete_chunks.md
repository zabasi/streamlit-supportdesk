# Repairing kept chunks at their edges (experiment)

Fixed windows start and stop anywhere, so a chunk the reranker keeps can begin with the end of the
previous section, or stop in the middle of the section that holds the answer. `retrieval.complete` repairs
the kept documentation chunks after reranking. The reranker's choice is not changed, only the text of the
chosen pieces, so the reranker is never confused by extra candidates (which is why expanding the
candidates before reranking lost gold). Repaired pieces that now overlap are merged as before.

| `retrieval.complete` | What happens to each kept documentation chunk |
|---|---|
| `none` (default) | nothing: advanced as it is |
| `neighbors` | the chunk before and the chunk after it on the page are added |
| `section` | it grows to the whole top-level sections it touches |
| `trim` | a short leading fragment of the previous section is dropped, so it starts at a heading ("fragment / heading / text" becomes "heading / text"); never the chunk's main part |
| `section_trim` | trim the start, then complete the last section |

A repaired piece may grow to `retrieval.complete_max_tokens` (900). When the full repair is longer, the
forward-only, then the backward-only repair is tried; otherwise the chunk is kept as it is. Issues,
comments and release notes are never changed: their headings are templates, not topics.

## Comparing at the same budget

Completion adds text, and the packet drops pieces that no longer fit the token budget, so compare at the
same larger budget as the 8-chunk option. All of this happens after reranking: the rewrite and reranking
calls replay from the cache, so these runs cost nothing.

```bash
T=$(date +%H%M)
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set context.max_tokens=4300 --run-id adv-4300-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set context.max_tokens=4300 --set retrieval.complete=section --run-id comp-section-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set context.max_tokens=4300 --set retrieval.complete=section_trim --run-id comp-sectrim-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set context.max_tokens=4300 --set retrieval.complete=neighbors --run-id comp-neighbors-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set retrieval.complete=trim --run-id comp-trim-$T
supportdesk eval-retrieval --pipeline advanced --split dev --mode live --set pipeline.scope_gate=off \
  --set retrieval.top_k=8 --set context.max_tokens=4300 --run-id adv-top8-$T
```

`trim` only removes text, so it runs at the usual budget. `adv-top8` is the other way to spend the same
budget: two more chunks instead of repaired ones.
