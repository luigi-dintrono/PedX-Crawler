"""Tests for scripts/evaluate_filter.py (stratified precision/recall estimation)."""
import csv

import pytest

pytest.importorskip('scipy')
pytest.importorskip('numpy')
from conftest import REPO_ROOT, _load_path  # noqa: E402


@pytest.fixture
def ev():
    return _load_path(REPO_ROOT / 'scripts' / 'evaluate_filter.py', 'evaluate_under_test')


def _decisions(n_accept=100, n_meta=300, n_visual=100, n_error=7):
    rows = {}
    def add(prefix, n, decision, stage):
        for i in range(n):
            vid = f'{prefix}{i}'
            rows[vid] = {'video_id': vid, 'decision': decision, 'stage': stage,
                         'timestamp_utc': '2026-10-10T00:00:00Z'}
    add('a', n_accept, 'accept', 'visual')
    add('m', n_meta, 'reject', 'metadata')
    add('v', n_visual, 'reject', 'visual')
    add('e', n_error, 'error', 'visual')
    return rows


def _labels(prefix, n, usable):
    return [{'video_id': f'{prefix}{i}', 'label': '1' if i < usable else '0'} for i in range(n)]


def test_stratified_estimates(ev):
    labels = _labels('a', 20, 18) + _labels('m', 20, 2) + _labels('v', 20, 4)
    labels.append({'video_id': 'a99', 'label': ''})           # unlabelled -> ignored
    labels.append({'video_id': 'ghost', 'label': '1'})        # not in decisions
    r = ev.evaluate(_decisions(), labels, n_boot=500)
    assert r['population'] == {'accept': 100, 'reject:metadata': 300, 'reject:visual': 100}
    assert r['errors_excluded'] == 7 and r['labels_unmatched'] == 1
    assert r['precision']['estimate'] == pytest.approx(0.9)
    lo, hi = r['precision']['ci95']
    assert lo < 0.9 < hi
    # recall = 100*0.9 / (100*0.9 + 300*0.1 + 100*0.2) = 90 / 140
    assert r['recall']['estimate'] == pytest.approx(90 / 140)
    assert r['recall']['ci95'][0] < 90 / 140 < r['recall']['ci95'][1]
    assert r['prevalence'] == pytest.approx(140 / 500)
    assert r['false_rejection_rate']['metadata']['estimate'] == pytest.approx(0.1)


def test_recall_needs_every_stratum_labelled(ev):
    r = ev.evaluate(_decisions(), _labels('a', 10, 9), n_boot=50)
    assert r['precision']['estimate'] == pytest.approx(0.9)
    assert r['recall']['estimate'] is None


@pytest.mark.parametrize("raw,value", [('1', 1), ('Yes', 1), ('usable', 1), ('0', 0), ('n', 0),
                                       ('', None), ('maybe', None)])
def test_parse_label(ev, raw, value):
    assert ev.parse_label(raw) == value


def test_sample_command_is_stratified_and_shuffled(ev, tmp_path):
    path = tmp_path / 'decisions.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['video_id', 'decision', 'stage', 'timestamp_utc', 'title'])
        w.writeheader()
        w.writerows(_decisions(n_accept=30, n_meta=5, n_visual=12).values())
    out = tmp_path / 'labels.csv'
    ev.main(['sample', '--decisions', str(path), '--per-stratum', '10', '--out', str(out)])
    rows = list(csv.DictReader(open(out, encoding='utf-8')))
    counts = {h: sum(r['stratum'] == h for r in rows) for h in ev.STRATA}
    assert counts == {'accept': 10, 'reject:metadata': 5, 'reject:visual': 10}
    assert all(r['label'] == '' for r in rows)
    assert [r['stratum'] for r in rows] != sorted(r['stratum'] for r in rows)  # not grouped
