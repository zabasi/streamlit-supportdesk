# Generation handoff: Controlled Routing v2

This is the selected frozen generation candidate from the full 15-case development
run, not the older policy-v3 ZIP. Model: gpt-5.4-mini; reasoning_effort=medium;
max_completion_tokens=8000. One structured model assessment per invocation; Python
owns action routing. No confidence threshold, extra reviewer or automatic retries.

## Install and connect

Copy supportdesk_generation/ into the repository src/ directory. Use the existing
project Python environment; the generation package itself uses the standard library.
examples/connect_retrieval.py shows the repository's state['case']/state['packet']
boundary and conversion to its ParsedAnswer type. Wire this node after retrieval and
packet assembly. Do not pass its assessment through the older parse_answer schema.

    from supportdesk_generation import generate_from_packet, to_pipeline_answer
    result = generate_from_packet(case, packet, complete)
    answer = to_pipeline_answer(result)

complete(request) is the pipeline's existing Chat Completions transport: accept the
entire request dict, including strict response_format, and return the raw provider
response dict with choices and usage. For an existing compatible SDK client the
callback body is client.chat.completions.create(**request).model_dump(). Configure
its timeout, max_retries=0 and cost accounting in the host pipeline. Do not translate
this request to a different response schema or remove reasoning/token settings.
The offline package has no HTTP client or automatic batch runner. No credentials,
.env or API keys are supplied. Keep your existing environment configuration private.

## Retrieval input contract

case is a CaseInput dataclass or dict: case_id, title, body, environment, history,
created_at. packet is a Packet dataclass or dict with chunks. Each chunk must have
chunk_id, source_id, source_type and exact text. Preserve title, url, version, as_of,
start_char/end_char, source_text_sha256 and meta authority/usage fields when available.
The adapter takes every supplied chunk in its existing order; it never substitutes
gold evidence, looks up development answers, calls retrieval or forces six chunks.
It rejects empty packets/duplicate chunks and checks context size before the model
call. This boundary expects assembled chunks, not raw retrieval Hit objects.

Reference labels S1...Sn are assigned by evidence order. reference_map in the result
maps them to original chunk IDs; use it for citation UI, not the packet's old label
order. to_pipeline_answer converts cited IDs for existing pipeline storage. Keep the
complete result under extra for diagnostics and maintainer summary. R/H references
refer to user report/history, not evidence chunks.

## Output and conversation

result includes outcome, action, user_reply, routing, assessment, reference_map,
maintainer_summary, reported_observations, content_warnings and provider usage.
outcome is answer, request_info or escalate. Escalation is only a proposal: no ticket
or external action is performed. status remains proposal_requires_content_review.
The app owns memory, durable session isolation and the follow-up counter: append the
assistant question and user reply chronologically to history, increment questions_used
when a question is delivered, and call again only after the user's reply. Preserve the
original report, refresh retrieval for the clarified question when useful, and pass
the new packet without claiming unprovided tests happened. Maximum follow-up questions
is five, refinement count at most one, history 16000 characters, input 60000 characters
and 48000 UTF-8 bytes. No automatic refinement loop or summary memory is implemented.
Do not reset counters on each UI rerun. Revision count is a limit, not a retry scheduler.

Provider failures, refusals, truncation, invalid JSON or unsupported assessments raise
an error; they are not relabeled as product escalation. The transport must record usage
even when validation fails, enforce your overall spending limit, and avoid hidden retries.

## Tested scope and limitations

results/development-results.html is self-contained and opens offline. The same HTML
is delivered separately. It contains the saved 15-case outputs, prompts, evidence,
comparison and review notes. Local download links are embedded in the HTML itself.
It is evaluator-only and must never be passed as model input.

11/15 decisions matched stored development labels (73.3%): answer 5/6, request_info
3/6, escalate 3/3. This is action-label agreement, not overall answer accuracy.
The run used manually curated development passages, not live retrieval. The four
label mismatches and content problems remain documented. No new paid run was made
for packaging, and integrated retrieval or real multi-turn performance is not yet
validated. The adapter has offline tests only. This is the selected prototype for
pipeline integration, not a claim of production readiness.

Core prompt/schema/router/renderer are unchanged from the frozen run. generation.py
has only a package-relative context import. New adapter checks cannot fix omitted
information gaps, missed edits, or unsupported model reasoning. Original exact files
are included under frozen_source/ for audit; its transport.py is an experiment-only
runner with repository dependencies, not the recommended pipeline transport.

## Offline verification

From this extracted package directory:
    python -m unittest discover -s tests -v
    python verify_handoff.py

data/development_model_inputs.jsonl is an optional frozen replay fixture, not a live
retrieval default. There is no test-split dataset or hidden test rubric in this package.
Original development/test data, generation experiments and GitHub were not changed.
