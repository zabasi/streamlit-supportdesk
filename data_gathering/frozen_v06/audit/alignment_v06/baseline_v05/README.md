# Streamlit Session State reviewed sources v05

This version audits first-hop references in the selected v04 text and adds reviewed linked-source excerpts. The 472 original texts remain exact; 68 sources are added, for 540 source records. These are not chunks or embeddings.

Load only the five explicit retrieval_corpus JSONL files with scripts/load_retrieval_sources.py. The compact ZIP contains identical records in corpus.jsonl. Embed only text; preserve metadata separately. Do not recursively ingest audit, evaluator_only, agent_inputs, config, scripts or review_pending. Do not index both copies or both versions.

The added linked_reference type is dated community discussion, a proposed change description or fixed-commit source code. It is not a certified solution. Existing issue/comment evidence roles remain unchanged. Added official API excerpts declare their retrieved version when present; they do not prove historical-version behavior.

metadata.linked_source_ids and metadata.linked_references identify useful neighbor records and reference decisions. Retrieval must explicitly use these relations when appropriate. URLs alone do not import page content, and relations do not automatically retrieve neighbors. audit/link_review_v05 contains the per-URL decisions, source spans, graph and immutable baseline; none of these reports may enter the index.

Some links are unavailable, private, live demos, moving-branch code or citation-only provenance and were not expanded. No live apps, videos, source-code examples, embeddings or RAG evaluation have been executed. The first-hop audit does not recursively crawl newly added sources or fetch URLs occurring only in code examples.

Run python3 scripts/validate_dataset.py for v05 offline integrity checks. Run python3 scripts/build_dataset.py to reproduce the selected envelopes from local baseline and recorded evidence without fetching. The suffixed v03/v04 scripts are historical tools and must not be used to rebuild v05. The 15 development and 15 test inputs and evaluator rubrics remain unchanged.
