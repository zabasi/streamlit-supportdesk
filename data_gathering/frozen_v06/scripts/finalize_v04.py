from pathlib import Path
import json, hashlib, shutil

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def put(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def finalize():
 summary=read('audit/cleanup_summary_v04.json')
 volume=read('audit/prior_v03_retrieval/preparation_summary.json')
 volume.update(summary['after']);volume['comparison_baseline']='v03'
 volume['changed_text_records_from_v03']=summary['changed_text_records']
 volume['metadata_fields_corrected']=summary['metadata_correction_count']
 put('audit/preparation_summary.json',volume)
 manifest=read('audit/prior_v03_retrieval/manifest.json')
 manifest.update(schema_version='4.0',preparation_date='2026-10-08',text_preparation=summary,source_volume=volume)
 manifest['baseline_delivery_sha256']={'Streamlit_Session_State_Reviewed_v03.zip':'b433e22905be17e6bd73a1ba70c1620caca28eef5158a607ffc4ccf2ef7fdfc2','Streamlit_Session_State_Chunking_Input_v03.zip':'a34f9687da548b66b7cb8cda00bc738d009c146c80481e6468b008c4321d538f'}
 put('manifest.json',manifest)
 files=['issues.jsonl','comments.jsonl','documents.jsonl','release_sections.jsonl']
 records=[json.loads(l) for f in files for l in (ROOT/'retrieval_corpus'/f).read_text().splitlines()]
 contract=read('audit/prior_v03_retrieval/Embedding_Structure_v03.json')
 contract['contract_version']='4.0';contract['source_volume']=volume
 contract['actual_source_example']=next(r for r in records if r['source_id']=='issue:7851')
 for typ in contract['source_envelope']['metadata_fields_by_type']:
  contract['source_envelope']['metadata_fields_by_type'][typ]=sorted({k for r in records if r['source_type']==typ for k in r['metadata']})
 contract['approval_policy']['approved_text_rule']='config approved_text preserves the v03 exact selected source substring. The delivered v04 text applies audited nontechnical cleanup after selection; it need not remain a contiguous source substring. Preserve claims, uncertainty and code, and validate via audit/text_cleanup_v04.jsonl.'
 contract['text_preparation']={'version':'v04','baseline':'v03','audit':'audit/text_cleanup_v04.jsonl','metadata_corrections':'audit/metadata_corrections_v04.jsonl','code_policy':'Fenced and inline backtick code tokens are byte-for-byte equal to the v03 selected text. Code is not executed. Technical names and URLs remain.','archives':'Original audit snapshots and historical review decisions remain unmodified and must never be indexed.'}
 put('Embedding_Structure_v04.json',contract)
 (ROOT/'Embedding_Structure_v03.json').unlink(missing_ok=True)
 before=summary['before'];after=summary['after'];delta=before['text_utf8_bytes']-after['text_utf8_bytes'];pct=delta/before['text_utf8_bytes']*100
 table='\n'.join(f'| {t} | {summary["source_counts"][t]} | {summary["changed_text_records_by_type"][t]} |' for t in ['issue','comment','documentation','release_section'])
 report=f'''# Streamlit Session State source review v04

The v04 delivery corrects nontechnical text left in v03. It retains all **472 selected sources**: 126 issues, 214 comment excerpts, 21 documentation pages and 111 release entries across 52 versions. These are source records, not chunks. No chunking, embeddings, paid model calls, source-code execution or agent evaluation has been performed.

## Changes from v03

Reviewed salutations, social thanks, personal addresses, contributor acknowledgements, email envelopes and decorative icons were removed outside code. Unfilled GitHub template fields, empty headings and unchecked default regression checkboxes were removed; an unchecked default is not asserted to mean no regression. Technical versions, outcome reports, failed attempts, uncertainty, links and platform/library names remain. Fenced code and inline backtick code are exactly preserved. A remaining “thanks to Streamlit's execution model” expresses technical causality rather than personal thanks. The “hello dear world” documentation entry is a filename example.

| Source type | Sources retained | Text records changed |
|---|---:|---:|
{table}

**{summary['changed_text_records']} records have text changes**, including formatting and whitespace changes; this is not a count of records containing social chatter. UTF-8 text volume changes from **{before['text_utf8_bytes']:,} to {after['text_utf8_bytes']:,} bytes**, a decrease of {delta:,} bytes ({pct:.2f}%). Characters change from {before['text_characters']:,} to {after['text_characters']:,}; whitespace-separated words from {before['text_whitespace_words']:,} to {after['text_whitespace_words']:,}. These measurements are not model token totals or API price estimates.

**33 incorrect environment metadata fields in 12 issues** are set to null. The previous extractor had crossed an empty field into the following field or heading. For example, issue 6970 did not specify a Streamlit version; its Python version is not a Streamlit version. Original source bodies, historical selected excerpts and review decisions remain in the audit archive. Delivered text hashes are recalculated, while original body hashes continue to describe original bodies. Per-record edit spans and metadata corrections make the change traceable.

## Selection inherited from v03

The source-selection decisions are unchanged. The source archive reconciles 690 unique issues and 3,181 comments. The 487 screened comment candidates remain partitioned into 214 selected, 260 excluded and 13 pending. Of the selected comments, 175 came from the earlier 343-candidate shortlist and 39 were recovered from automatic exclusions. All 20 earlier documentation page identities remain plus the relevant caching overview. The 111 release entries describe state/value/identity, callbacks, reruns, lifecycle and related tests; full annual release pages are audit material.

Selected comment roles remain **140 reported observations, 72 diagnostic-context excerpts and 2 narrowly corroborated claims**. Approval is an AI assistant selection of a specified evidence role, not a human review, a paid external judge run or certification of every claim. Source closure and contributor status are not truth guarantees. Historical feature requests do not prove an API was implemented. Apply documentation and release statements within their recorded version/date scope.

## Preservation checks and evaluation boundary

Offline checks verify unique IDs and parent links, source/text hashes, the explicit index allowlist, unchanged selected IDs and roles, byte-for-byte preserved protected code, reproducible cleanup, all comment dispositions, honest AI provenance and held-out/pilot exclusions. All 60 inherited official rubric evidence spans remain exact in the archive and their normalized content remains retrievable. The compact corpus must equal the four selected files in the full package.

**15 development cases and 15 test cases** are retained in `agent_inputs/development_inputs.jsonl` and `agent_inputs/test_inputs.jsonl`, with evaluator rubrics in `evaluator_only`. Each split has 6 answerable, 6 clarification and 3 escalation cases, with 3 cases per each of 5 topics. The 10 multi-turn specifications remain 5 development and 5 test; they are not executed. Agent inputs and evaluator files are byte-for-byte unchanged from v03 and are not embedding input. Rubrics, oracle replies and future turns must stay inaccessible to the evaluated agent except for authorized due user turns.

## Use and remaining work

For chunking, use only `corpus.jsonl` from the compact v04 ZIP and submit only each record's `text` for embedding. Retain `metadata`, `source_id` and hashes for citations, roles, versions and identity. Alternatively load only the four `retrieval_corpus` files with the full package's explicit loader. Do not index both representations, any Word/report/manifest, `audit`, `config`, `review_pending`, `agent_inputs` or `evaluator_only`. Audit copies intentionally preserve original social wording and are excluded from the index.

The preparation is ready for a separate chunking step. Retrieval recall, answer quality and scenario behavior remain unmeasured. Tune using development cases and freeze settings before final test use. The supplied project budget is USD 5 for the whole team/project through Metis, including development, corpus/query embeddings, experiments, evaluation and final runs. This preparation made zero Metis or embedding calls; prior spending elsewhere is unknown. Choose a tokenizer/model and use actual rates and remaining balance before paid execution. Volume reduction alone does not guarantee the USD 5 cap.
'''
 (ROOT/'REVIEW_REPORT.md').write_text(report)
 change=f'''# گزارش تغییرات نسخه ۴ داده های Streamlit Session State

نسخه ۴ متن منتخب را پیش از چانکینگ اصلاح می‌کند. در نسخه ۳، بخشی از تشکرها و خطاب‌های شخصی باقی مانده بود؛ در این نسخه این عبارت‌ها همراه با نویز قالب و فاصله‌گذاری اصلاح شدند. تعداد و هویت همه ۴۷۲ رکورد حفظ شده است. کدهای محصور در حصار Markdown و کدهای درون backtick با نسخه ۳ برابرند. کدها اجرا نشده‌اند.

| شاخص | نسخه ۳ | نسخه ۴ |
|---|---:|---:|
| رکورد منتخب | 472 | 472 |
| ایشو | 126 | 126 |
| کامنت | 214 | 214 |
| مستند | 21 | 21 |
| ورودی یادداشت انتشار | 111 | 111 |
| حجم UTF-8 فیلد متن | {before['text_utf8_bytes']} | {after['text_utf8_bytes']} |
| نویسه | {before['text_characters']} | {after['text_characters']} |
| واژه با جداسازی فاصله | {before['text_whitespace_words']} | {after['text_whitespace_words']} |

متن {summary['changed_text_records']} رکورد تغییر کرده است؛ این عدد شامل تغییر فاصله‌گذاری و قالب نیز هست. کاهش حجم متن {delta:,} بایت معادل {pct:.2f} درصد است و برآورد توکن یا هزینه نیست. ۳۳ مقدار نادرست فراداده محیط در ۱۲ ایشو نیز با null اصلاح شده‌اند؛ مقدار خالی قبلاً از خط بعدی استخراج شده بود. گزارش خطا، موفقیت یا شکست راهکار، شرط نسخه و عدم قطعیت نویسندگان حفظ شده‌اند.

مجموعه‌های ۱۵تایی توسعه و تست، معیارهای ارزیاب و ۱۰ سناریوی چندمرحله‌ای بدون تغییر بایتی حفظ شده‌اند. هر بخش ۶ مورد پاسخ‌پذیر، ۶ مورد نیازمند توضیح و ۳ مورد نیازمند ارجاع دارد. نقش‌های ۲۱۴ کامنت همچنان شامل ۱۴۰ مشاهده گزارش‌شده، ۷۲ زمینه تشخیصی و ۲ ادعای محدود همراه شاهد رسمی است. این‌ها نتیجه انتخاب دستیار هوش مصنوعی هستند و تأیید اجرای مستقل یا بازبینی انسانی محسوب نمی‌شوند.

کنترل‌های محلی شامل شناسه و رابطه والد، هش متن و بدنه، بازسازی تغییرات، حفظ کد، مرز داده آزمون و حفظ ۶۰ قطعه شاهد رسمی است. آرشیو اصلی و تصمیم‌های قبلی برای ممیزی نگه داشته شده‌اند و نباید وارد شاخص شوند. برای چانکینگ تنها فیلد text در corpus.jsonl نسخه ۴ استفاده شود؛ metadata برای استناد نگه داشته شود. از ورود هم‌زمان نسخه‌های ۳ و ۴ یا دو نمایش یک داده به شاخص جلوگیری شود.

این نسخه اصلاح ورودی داده است؛ چانکینگ، امبدینگ، اجرای کد و ارزیابی عملکرد RAG انجام نشده است. هیچ تماس پولی API در این مرحله صورت نگرفته است. سقف ۵ دلار کل پروژه باید پس از انتخاب مدل، محاسبه توکن و بررسی موجودی و تعرفه واقعی کنترل شود.
'''
 (ROOT/'CHANGE_REPORT_v04_FA.md').write_text(change)
 (ROOT/'README.md').write_text('''# Streamlit Session State reviewed sources v04

Use only the four explicit retrieval_corpus JSONL files for chunking. The compact ZIP supplies the identical records in corpus.jsonl. Embed only text; retain metadata separately. This is 472 sources, not chunks or embeddings.

## Data and evaluation

- retrieval_corpus: 126 issues, 214 comments, 21 documents, 111 release sections.
- agent_inputs/development_inputs.jsonl: 15 fixed development cases.
- agent_inputs/test_inputs.jsonl: 15 fixed test cases.
- evaluator_only: scoring rubrics, held-out context and 10 multi-turn specifications; never index or preload them for the agent.
- audit: original archives, immutable v03 baseline, before/after cleanup and metadata-correction evidence; never index.
- config and review_pending: source-selection decisions and 13 pending comments; never index.

REVIEW_REPORT.md describes selection and limits. CHANGE_REPORT_v04_FA.md describes changes in Persian. Embedding_Structure_v04.json defines envelope, metadata and future chunk contracts. validation.json records offline checks. An AI selection role does not certify runtime correctness.

## Offline commands

From this extracted directory:

```sh
python3 scripts/load_retrieval_sources.py
python3 scripts/validate_dataset.py
```

To reconstruct v04 selected envelopes from the immutable delivered v03 baseline, including reviewed cleanup, metadata corrections and reports:

```sh
python3 scripts/build_dataset.py
```

The v04 rebuild requires the complete extracted package and does not fetch data, change evaluation, generate chunks, or call APIs. scripts/build_dataset_v03.py is preserved only as the historical v03 builder and normalization reference; do not run it to generate v04. Newly acquired sources require a separate selection review, not this fixed-baseline rebuild. The loader is an allowlist, not operating-system isolation.

Original source snapshots intentionally retain original social phrasing. Do not scan recursively or ingest reports, baseline copies, evaluation, approval files or Word displays. Select one v04 representation, not both full and compact packages. Revalidate and freeze any later chunking configuration on development data before final test use. No source snippets or scenarios have been executed.
''')

def checksums():
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts}
 put('SHA256SUMS.json',hashes)

if __name__=='__main__':finalize();checksums()
