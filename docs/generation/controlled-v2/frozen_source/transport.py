"""Frozen six-call live diagnostic; no retries, evaluator access or external writes."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from generation import BUNDLE, issues, rendered, request, rows, save, sha, validate

ROOT = BUNDLE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supportdesk.llm.accounting import BudgetLedger

CASE_IDS = ('DEV-7044', 'DEV-3842', 'DEV-10805')
MODELS = ('gpt-4o-mini', 'gpt-5.4-mini')
CAP = 0.10


class Stop(Exception):
    """Safe error text only; raw SDK exceptions never enter exported results."""


class Spending:
    def __init__(self, ledger, cap=CAP):
        self.ledger, self.cap = ledger, cap
        self.exposure, self.calls = 0.0, 0

    def reserve(self, options, rates):
        if self.calls >= 6:
            raise Stop('Six-call limit reached')
        input_bound = len(json.dumps(options, ensure_ascii=False).encode('utf8')) + 4096
        amount = (input_bound * rates['input'] + options['max_completion_tokens'] * rates['output']) / 1e6
        if self.exposure + amount > self.cap + 1e-12:
            raise Stop('Additional estimated spend cap would be exceeded')
        if self.ledger.would_exceed(amount, 0.25):
            raise Stop('Shared project budget would be exceeded')
        self.exposure += amount
        self.calls += 1
        return amount

    def settle(self, reserve, usage, rates):
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                             for k in ('input_tokens', 'output_tokens')):
            raise Stop('Missing usage; reservation retained and run stopped')
        amount = (usage['input_tokens'] * rates['input'] + usage['output_tokens'] * rates['output']) / 1e6
        if not math.isfinite(amount) or amount > reserve + 1e-12:
            raise Stop('Usage exceeded reservation; stop and reconcile')
        self.exposure += amount - reserve
        return amount


def refreshed_prices(out, *, allow_saved=False):
    import httpx
    url = 'https://www.metisai.ir/models.json'
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
        catalog = response.json()
        source_note = 'Fresh public catalog'
    except Exception:
        if not allow_saved:
            raise Stop('Public price refresh failed; no paid calls started') from None
        # Explicit operator option; this is disclosed as an estimate using an older snapshot.
        snapshot = ROOT.parent / 'metis-models-catalog-latest.json'
        if not snapshot.is_file():
            raise Stop('Saved provider catalog is missing') from None
        catalog = json.loads(snapshot.read_text(encoding='utf8'))
        source_note = 'Live refresh blocked; saved project catalog, rates previously verified 2026-10-14'
    if not isinstance(catalog, list):
        raise Stop('Unrecognized provider price catalog')
    prices = {}
    for model in MODELS:
        matches = [r for r in catalog if r.get('metis_model_id') == model]
        if len(matches) != 1:
            raise Stop('Expected model missing or ambiguous in provider catalog')
        row = matches[0]
        price = {'input': row.get('input_price'), 'output': row.get('output_price')}
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in price.values()):
            raise Stop('Provider prices are not usable')
        params = row.get('supported_parameters', [])
        if not {'response_format', 'structured_outputs', 'max_completion_tokens'} <= set(params):
            raise Stop('Required structured-output parameters absent from catalog')
        if model == 'gpt-5.4-mini' and 'reasoning_effort' not in params:
            raise Stop('Reasoning effort is absent from provider catalog')
        prices[model] = {**price, 'supported_parameters': params}
    save(out / 'prices.json', {'source': url, 'checked_at': datetime.now(timezone.utc).isoformat(),
                             'source_note': source_note, 'live_refresh_succeeded': source_note == 'Fresh public catalog',
                             'models': prices, 'unit': 'USD per million tokens',
                             'note': 'Estimates without cache discounts; not an invoice.'})
    return prices


def run_call(client, options, row, out, spending, rates):
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'api-request.json', options)
    fingerprint = sha(json.dumps(options, ensure_ascii=False, sort_keys=True).encode('utf8'))
    reservation = spending.reserve(options, rates)
    event = {'case_id': row['case_id'], 'model': options['model'], 'request_sha256': fingerprint,
             'status': 'pending', 'reserved_usd': reservation, 'usage': None,
             'contract_valid': False, 'estimated_cost_usd': reservation}
    # Survives interruption before settlement; the next invocation must reconcile it.
    save(out / 'event.json', event)
    started = time.perf_counter()
    stop = None
    try:
        try:
            response = client.chat.completions.create(**options)
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            raise Stop('Provider call failed' + (f' (HTTP {status})' if isinstance(status, int) else '')
                       + '; reservation retained, no retry') from None
        usage = getattr(response, 'usage', None)
        tokens = {'input_tokens': getattr(usage, 'prompt_tokens', None),
                  'output_tokens': getattr(usage, 'completion_tokens', None)}
        details = getattr(usage, 'completion_tokens_details', None)
        event['usage'] = {**tokens, 'reasoning_tokens': getattr(details, 'reasoning_tokens', None)}
        # Save the actual text even when usage/finish/schema is unusable.
        choice = response.choices[0] if response.choices else None
        raw = {'text': choice.message.content if choice else None,
               'finish_reason': choice.finish_reason if choice else None,
               'refusal': bool(getattr(choice.message, 'refusal', None)) if choice else False,
               'usage': event['usage'], 'request_sha256': fingerprint,
               'returned_model': response.model}
        save(out / 'response.json', raw)
        event['estimated_cost_usd'] = spending.settle(reservation, tokens, rates)
        event['usage_known'] = True
        if raw['refusal'] or raw['finish_reason'] != 'stop':
            event.update(status='invalid_output', error='Refused or incomplete response; raw output retained')
        else:
            try:
                candidate = json.loads(raw['text'])
                validate(candidate, row)
            except (TypeError, ValueError, KeyError):
                event.update(status='invalid_output', error='JSON/schema/reference validation failed; raw output retained')
            else:
                save(out / 'output.json', candidate)
                save(out / 'rendered.json', rendered(candidate, row))
                event.update(status='proposal', contract_valid=True, outcome=rendered(candidate, row)['outcome'],
                             deterministic_warnings=issues(candidate), semantic_accuracy=None)
    except Stop as exc:
        event.update(status='stopped', error=str(exc))
        stop = exc
    finally:
        event['latency_seconds'] = round(time.perf_counter() - started, 4)
        spending.ledger.record(stage='generation_redesign_v4', request_type='chat', mode='live',
                              model=options['model'], request_hash=fingerprint[:32], items=1,
                              input_tokens=(event['usage'] or {}).get('input_tokens') or 0,
                              output_tokens=(event['usage'] or {}).get('output_tokens') or 0,
                              cached_input_tokens=0, cost_usd=event['estimated_cost_usd'],
                              status='ok' if event['status'] == 'proposal' else event['status'],
                              latency_seconds=event['latency_seconds'],
                              cost_note='Token-rate estimate without cache discount; unknown usage retains reservation.')
        save(out / 'event.json', event)
    if stop:
        raise stop
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--allow-saved-prices', action='store_true',
                        help='Allow the disclosed saved provider catalog if live public refresh fails')
    args = parser.parse_args()
    if not args.live:
        parser.error('Live calls require --live; offline requests are already exported in api/')
    work = ROOT / 'artifacts/reports/generation-redesign-v4-20261015'
    marker = work / 'latest-run.json'
    if marker.exists():
        raise Stop('Comparison already started; inspect saved results before authorizing another run')
    # Detect an unsettled prior v4 call instead of starting a second spend stream.
    for event_file in (ROOT / 'artifacts/runs').glob('generation-redesign-v4-*/**/event.json'):
        if json.loads(event_file.read_text(encoding='utf8'))['status'] == 'pending':
            raise Stop('Unsettled previous call; reconcile before any new run')
    all_rows = {r['case_id']: r for r in rows()}
    chosen = [all_rows[cid] for cid in CASE_IDS]
    for row in chosen:
        a, b = [request(row, model) for model in MODELS]
        assert a['messages'] == b['messages'] and a['response_format'] == b['response_format']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'artifacts/runs' / ('generation-redesign-v4-' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    rates = refreshed_prices(out, allow_saved=args.allow_saved_prices)
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / '.env', override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        raise Stop('Existing local API credentials are missing')
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ.get('OPENAI_BASE_URL') or None,
                    timeout=60, max_retries=0)
    ledger = BudgetLedger(ceiling_usd=5.0, path=ROOT / 'artifacts/cost_ledger.jsonl')
    mark, prior = ledger.mark(), ledger.total_cost_usd
    spending = Spending(ledger)
    plan = {'case_ids': CASE_IDS, 'models': MODELS, 'max_calls': 6, 'estimated_new_spend_cap_usd': CAP,
            'shared_project_cost_before_usd': prior, 'ledger_mark': mark,
            'selection': 'One case from each action category, frozen before calls; two models receive identical inputs.',
            'workflow': 'Single generation per model/case; no model reviewer or automatic refinement',
            'held_out_test_access': False, 'evaluator_rubrics_loaded': False,
            'retrieval_calls': 0, 'embedding_calls': 0, 'tracker_writes': 0,
            'source_dataset_sha256': sha((BUNDLE / 'inputs/model_inputs.jsonl').read_bytes()),
            'prompt_sha256': sha((BUNDLE / 'prompt.txt').read_bytes()),
            'blind_evaluation': False, 'accuracy': None}
    save(marker, {'path': str(out)})
    save(out / 'experiment.json', plan)
    events, stopped = [], None
    started = time.perf_counter()
    try:
        for row in chosen:
            for model in MODELS:
                print(f'{model} {row["case_id"]}: starting', flush=True)
                event = run_call(client, request(row, model), row, out / model / row['case_id'], spending, rates[model])
                events.append(event)
                save(out / 'progress.json', {'events': events, 'exposure_usd': spending.exposure})
                print(f'{model} {row["case_id"]}: {event["status"]} {event.get("outcome", "")}', flush=True)
    except Stop as exc:
        stopped = str(exc)
    finally:
        save(out / 'execution-summary.json', {**ledger.summarize(mark),
             'wall_time_seconds': round(time.perf_counter() - started, 3),
             'budget_exposure_usd': spending.exposure, 'stopped': stopped, 'accuracy': None})
        client.close()
    print(json.dumps({'run_path': str(out), 'summary': ledger.summarize(mark), 'stopped': stopped}), flush=True)
    if stopped:
        raise Stop(stopped)


if __name__ == '__main__':
    try:
        main()
    except Stop as exc:
        print('Stopped: ' + str(exc), flush=True)
        raise SystemExit(1) from None
