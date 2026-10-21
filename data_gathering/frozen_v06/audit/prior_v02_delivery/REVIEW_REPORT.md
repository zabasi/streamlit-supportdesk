# Streamlit Session State: Dataset Review and Selective Merge

**Version:** 2.0  
**Review date:** 2026-10-07  
**Supplied snapshot date:** 2026-10-06  
**Status:** Source preparation complete; comment review, rubric signoff, chunking, embeddings, and agent evaluation remain pending.

## 1. Decision

Use a selective merge focused on Session State and its interactions with widget lifecycle, callbacks, reruns, forms, and multipage apps. Preserve the previously selected 30 evaluation cases and their case-specific rubrics. Add relevant evidence from the colleague's collection, retain the supplied source snapshots for audit, and require explicit review before indexing community comments.

The colleague's package is a useful collection starting point. Its original corpus and evaluation should not be used unchanged: evaluation threads are present in the retrieval corpus, evidence selection is largely generic, and broad search terms collect unrelated topics. Larger counts alone do not improve a support agent.

This delivery is a reviewed source package and an evaluation specification. It is not a completed agent, a truth-certified knowledge base, or a measured RAG result.

## 2. Inputs and count reconciliation

| Collection | Issues | Comments | Documentation | Release material |
|---|---:|---:|---|---|
| Original focused collection | 220 | 954 | 28 representations of 20 pages | 81 version records |
| Previous prepared evaluation package | 154 retrieval issues | 580 retrieval comments | 20 canonical pages | 269 sections across 67 versions |
| Colleague's collection | 500 | 2,384 | 50 Markdown files | 9 annual/index pages; these are not 9 versions |
| Merged source archive | 690 unique issues | 3,181 unique comments | 70 source representations | Original 81 versions and supplied annual snapshots retained |
| Current retrieval input | 164 | **0 approved comments** | 21 canonical pages | 263 sections across 67 versions |
| Separate comment review queue | — | **343 candidates** | — | — |

The issue union is 220 + 500 − 30 overlapping issues = 690. The comment union removes 157 duplicate comment IDs. Of 690 issues, 249 match the initial focused date/topic rules. Additional scope review, incomplete-thread checks, media dependence, held-out families, and references to blocked sources reduce the current retrieval set to 164 issues.

The previous retrieval set increases from 154 to 164 issues: 12 issues are added and 2 previous retrieval issues are excluded by the expanded pilot/reference protections. One narrative caching concept page is added to the 20 canonical documentation pages. Six additional release sections are withheld because they refer to blocked issue/PR numbers.

Counts describe **source records**, not embedding chunks. A release section is a selected changelog unit; 263 sections do not mean 263 releases. No embeddings or final chunks have been generated.

## 3. Review of the colleague's work

| Finding | Assessment | Final treatment and evidence |
|---|---|---|
| GitHub pagination follows Link headers; pull requests are filtered out | Useful collection engineering | Original scripts retained under `audit/colleague_scripts/`; no recollection or execution required for this review |
| Comments retain parent issue references and timestamps | Valuable context and provenance | Merge by issue number and comment ID; preserve complete archived thread context |
| Documentation is tied to commit `695e6ce2fb1b23afc6df66b40420ab0013ad72fd` | Good reproducibility for pinned Markdown | Retain commit/path metadata and source hashes; do not assign this commit to independently rendered live HTML |
| Reusable scripts and source snapshots are provided | Useful for inspection and later work | Archive as non-runtime material; configuration files and bytecode are omitted |
| Broad keywords and updated-descending collection order | Topic drift and recency bias | Narrow to Session State interactions; preserve rejected records for audit; do not describe this as an unbiased sample of all issues |
| Original corpus contains all 30 evaluation roots and 181 comments attached to them | Direct evaluation leakage | Colleague cases are not adopted as gold; their threads and related candidates are quarantined |
| 29 cases cite the 2020 annual release page; 1 cites the 2019 page | Token overlap is not version-specific evidence | Retain previous case-specific evidence; colleague release references are audit evidence of a flaw, not answer keys |
| The same Widget Behavior page appears in 28 case evidence lists, including unrelated cases | Existence of a page does not establish support for a claim | Require an exact supporting span and applicability check; do not accept generic page matches as sufficient evidence |
| All 30 cases are marked multi-turn, but the second message is a generic request for more information | Does not test changing facts, state transitions, or operational behavior | Preserve the existing 10 specific scenario specifications; future turns are evaluator-only |
| `family_id` is derived only from issue number | Prevents same-ID overlap but misses related/duplicate reports | Preserve inherited related-source quarantine and add conservative relation rules |
| Issues 15892 and 15893 appear on opposite splits and describe a similar 1.59.0 tabs regression | Concrete cross-split duplicate candidate; equivalence is not fully human-verified | Quarantine both conservatively; record the relation and its uncertainty |
| Two comment totals disagree with downloaded counts | Source-completeness concern | Issue 2068: expected 10, received 9. Issue 12323: expected 4, received 3. Neither enters the focused retrieval set. Source deletion/editing is also possible; a fetch bug is not established |
| Document identifiers use Python `abs(hash(path))` | Process-dependent IDs are unstable | Use canonical path-based source IDs, commit/hash metadata, and an alias mapping |
| Many Markdown API pages contain unresolved automatic API components | A placeholder is not a full rendered API definition | Exclude 30 unresolved representations from retrieval; keep meaningful narrative text and existing rendered snapshots |
| Quality checks report PASS using presence/overlap checks | Useful structural checks, insufficient semantic review | Report structural PASS separately from semantic truth, rubric approval, and agent performance |

Only 11 of the colleague's 30 selected case roots pass the focused keyword/date rules. Passing that rule is not a full semantic relevance decision. Their case selection is not substituted for the previously fixed evaluation set.

## 4. Retrieval, review, and evaluator boundaries

| Area | Contents | Permitted use |
|---|---|---|
| `retrieval_corpus/` | Four explicit JSONL inputs: issues, approved comments, canonical documents, release sections | Candidate source text for retrieval/chunking |
| `review_pending/` | 343 comment candidates and context cards | Review only; never embed or expose as trusted agent evidence |
| `agent_inputs/` | Initial reports for 15 development and 15 test cases | Introduce one assigned case to the agent |
| `evaluator_only/` | Rubrics, full selected threads, future turns, withheld source material | Evaluation runner and reviewers only |
| `audit/` | Source snapshots, provenance, merge decisions, exclusions, media references, historical scripts | Reproducibility and inspection only |
| `config/` | Explicit review decisions and example review form | Control approved selections |

The loader enumerates four permitted filenames. It never recursively indexes the package. This is an operational boundary, not an operating-system access restriction: the runner must give the agent access only to the allowed retrieval output and its current case input. The full ZIP is a reviewer/developer package and must not be mounted as unrestricted agent knowledge.

Source reports are retrieved as reports, not as confirmed causes or fixes. Metadata records this distinction, including a warning that proposed APIs in feature requests do not establish implemented behavior. Official documentation and changelogs provide stronger product evidence, but still require a match to the user's version and situation.

## 5. Media policy

No image or video payload was downloaded, embedded, or interpreted. Images, video embeds, attachment references, and recognized media URLs are removed from retrieval text outside code fences. Their references remain in `audit/media_manifest.jsonl`, with `downloaded: false` and `interpreted: false`. Code fences remain intact, including URLs used as code.

Do not discard an entire thread merely because it contains a screenshot. Preserve useful standalone text and keep the original provided body in the audit inputs. Media-only or very short low-information comments are excluded from retrieval candidates. Mixed comments go to review when the text may depend on the attachment.

Fourteen of the 343 comment candidates carry a media-independence review flag. A reviewer must decide whether the selected excerpt is understandable without the media. If the missing image is needed to establish a fact, keep that fact unresolved or ask for a textual reproduction. No OCR or guessed description is supplied. Retained issues with media are still unverified reports; the presence of enough text is a relevance rule, not proof of complete media independence.

## 6. Incorrect, misleading, or obsolete comments

The default approved-comment file is intentionally empty. **343 candidates remain pending; this does not mean that 343 comments are false.** Every merged comment remains available in the source archive, and exclusions have reasons.

Review comments at the claim/excerpt level:

1. Read the parent report, the preceding and following messages, and the complete thread when needed. Check for a later correction or a failed attempt.
2. Separate observed behavior, reproduction steps, hypotheses, suggested workarounds, confirmed explanations, and proposals. Do not turn a suggestion into an established fix.
3. Check the Streamlit version and relevant environment. Unknown values stay `null`; no current-version behavior is assumed for historical issues.
4. Compare an alleged product fact with an exact official documentation/release section and assess version applicability. A linked page alone is not corroboration.
5. Select an exact excerpt that can stand on its own. Exclude misleading or media-dependent portions, or retain a useful failed attempt only as diagnostic context.
6. Record reviewer, rationale, supporting source IDs, and allowed use in `config/comment_reviews.json`; rebuild and validate.

Allowed uses are `reported_observation`, `diagnostic_context`, and `corroborated_claim`. The builder requires an exact source excerpt, review attribution, a version-applicability note, and supporting source IDs; media-containing comments also require an explicit declaration that the selected excerpt is self-contained without media; a corroborated claim also requires an official documentation/release source. These checks enforce the review record structure, not the semantic accuracy of the reviewer's reasoning.

The current flags include 87 explicit-uncertainty candidates, 67 short/context-dependent candidates, and 10 possible corrections or failed attempts. Flags can overlap and are prioritization hints, not truth labels. Author association, votes, and closed/completed status never automatically approve a statement.

An evaluator-only example is issue 5604: a suggested per-key initialization guard is explicitly distinguished from fixing the reported expander mismatch. Promoting the guard as the definitive solution would misrepresent the thread. This example remains withheld from retrieval.

## 7. Provenance, normalization, and metadata

Duplicate records are merged using stable GitHub identifiers. When supplied snapshots differ, the most recent `updated_at` wins; the focused source wins on a tie. `audit/merge_decisions.jsonl` retains bundle provenance and original body hashes. Input snapshots remain separate, so a reader can inspect both versions rather than trusting the merge silently.

Normalization removes narrow template/voting boilerplate, normalizes line endings, and redacts email addresses. Code fences are protected from boilerplate and media removal. Source language is preserved; no source text is translated. Five email occurrences in the colleague's parsed input snapshots were redacted. The supplied corpus copy is retained as an ID/type/title/URL/content-hash inventory rather than another duplicated body store. Audit snapshots are therefore provided-source copies with documented normalization/redaction, not a claim of byte-identical original files; archive hashes record the original input ZIP identity.

Metadata carries IDs, parent links, source URLs, repository, timestamps, labels, source role, explicit environment evidence, hashes, and review status. Missing per-record retrieval timestamps remain `null` where only bundle-level collection time was supplied. Labels and topic tags are hints, not truth guarantees. Version extraction preserves the explicit reported line and evidence location; it does not infer an unreported version.

Only a record's `text` is submitted for embedding. Metadata and audit information remain outside the embedding request. URLs naturally present in a source's technical prose or code remain source content; metadata URLs are not appended as embedding text.

Documentation IDs are stable canonical path/page IDs; commit and content hashes distinguish snapshot identity. Source Markdown and rendered live pages are not falsely treated as the same representation or commit. The 60 inherited rubric evidence spans point to archived canonical documentation bodies; they do not point to transformed retrieval text offsets. Exact spans and hashes have been independently checked.

## 8. Preserved evaluation design

The existing 30 cases remain fixed at 15 development and 15 test cases. Each split has 6 answerable cases, 6 requiring clarification, and 3 requiring escalation. Each split has 3 cases in each of five topics: initialization/updates, widget/session lifecycle, callbacks/reruns, forms, and multipage behavior.

The previous rubrics retain reference reasoning, acceptable next steps, unacceptable conclusions, and specific source evidence. All 60 retrievable documentation spans still match their archived source text and hashes. This confirms evidence integrity; it does not certify every rubric interpretation as correct. Team signoff remains pending.

Ten multi-turn scenario specifications are preserved, with five development and five test cases. Four follow-up messages are based on source-thread material and six are designed scenarios. They cover approved actions, maintainer rejection/edits, restart while awaiting approval, repeated requests, corrected facts, changed cases, timeout after commit, evidence search failure, and interleaved cases. These scenarios have not been executed.

The evaluator releases a follow-up only when its prerequisite response occurs. Initial agent inputs contain no gold answers, future messages, issue status, labels, or archived comments. Development rubrics support iteration; test material should remain locked during tuning.

Ninety-seven issue numbers are withheld for fixed evaluation, colleague case quarantine, pilots, or conservative related-source blocking. This is not a claim of 97 confirmed duplicates. Additional external issue/PR numbers are blocked from retrieval references. Rule-based relation checks can miss unknown semantic relatives; a final human family review is still needed.

## 9. Validation performed and practical limits

The independent validator passed checks for identifiers, parent links, retained-thread count reconciliation, text/body hashes, allowed input paths, held-out roots and references, known family split separation, one canonical document per page, 60 exact rubric spans, scenario structure, initial-input separation, and snapshot integrity.

An offline rebuild also produced byte-identical outputs. Review-card membership and the embedding-contract path/count boundaries were checked.

No agent, action tracker, model, embedding service, or scenario runner was executed. There are no measured retrieval scores, answer scores, latency figures, or hallucination rates. External model/embedding API calls: **0**. Recorded model/embedding API cost for this preparation: **USD 0**; this does not represent all execution/platform costs.

Remaining work: human review of scope/families, comment claims/media dependence, release-section relevance, documentation version applicability, and rubric/scenario signoff; then implement code-aware chunking, generate embeddings from the allowlist, build the agent/approval-state workflow, and execute the evaluation. Do not report the structural PASS as a final project score.

## 10. How to use the delivery

Read `README.md`, inspect `manifest.json`, and run the three offline commands documented there. Use `Embedding_Structure_v02.json` as the source-to-chunk contract. Review cards include parent text, neighboring comments, and full-thread source IDs; they belong outside the index. After making explicit review decisions, rebuild before chunking and repeat validation.

The complete ZIP includes the retrieval-ready source envelopes, preserved evaluation, audit snapshots, active scripts, documentation sources/license, review queue, and checksums. Historical scripts under `audit/` are retained for inspection, not as the active build entry point.
