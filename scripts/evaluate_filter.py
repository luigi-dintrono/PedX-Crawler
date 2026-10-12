#!/usr/bin/env python3
"""Validate the crawler's filters against hand labels.

The crawler logs every examined candidate in runs/<run id>/decisions.csv. This script

  1. `sample`:   draws a stratified random sample of those decisions for labelling.
                 Strata: accepted / rejected by the metadata stage / rejected by the
                 visual stage ("could not check" errors are excluded and counted).
  2. `evaluate`: reads the filled-in labels (1 = usable for PedX-Insight, 0 = not)
                 and estimates, with 95% intervals,
                   - precision   P(usable | accepted)               (Wilson interval)
                   - recall      share of usable candidates accepted (stratified bootstrap)
                   - per-stage false-rejection rate P(usable | rejected by stage)
                   - prevalence  of usable videos among examined candidates.

Because strata are sampled equally but differ in size, recall and prevalence weight
each stratum's labelled proportion by its population size (a stratified estimator);
precision is unbiased within the accepted stratum.

Usage:
  python scripts/evaluate_filter.py sample --decisions data/outputs/runs/<id>/decisions.csv \
      --per-stratum 50 --out labels.csv
  # ... open labels.csv, fill the 'label' column with 1/0 ...
  python scripts/evaluate_filter.py evaluate --decisions data/outputs/runs/<id>/decisions.csv \
      --labels labels.csv [--json report.json]
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest

STRATA = ('accept', 'reject:metadata', 'reject:visual')
POSITIVE = {'1', 'yes', 'y', 'true', 'relevant', 'usable'}
NEGATIVE = {'0', 'no', 'n', 'false', 'irrelevant', 'unusable'}
LABEL_FIELDS = ['video_id', 'video_url', 'title', 'city', 'stratum', 'stage', 'reason', 'score',
                'label', 'notes']


def stratum_of(row: Dict[str, str]) -> Optional[str]:
    decision = row.get('decision', '')
    if decision == 'accept':
        return 'accept'
    if decision == 'reject':
        return 'reject:visual' if row.get('stage') == 'visual' else 'reject:metadata'
    return None  # 'error': the filter could not judge the video


def load_decisions(paths: List[str]) -> Dict[str, Dict[str, str]]:
    """Latest decision per video id across one or more decision logs."""
    latest: Dict[str, Dict[str, str]] = {}
    for path in paths:
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                vid = row.get('video_id', '')
                if vid and (vid not in latest or row.get('timestamp_utc', '') >= latest[vid].get('timestamp_utc', '')):
                    latest[vid] = row
    return latest


def parse_label(raw: str) -> Optional[int]:
    value = (raw or '').strip().lower()
    if value in POSITIVE:
        return 1
    if value in NEGATIVE:
        return 0
    return None


def wilson(k: int, n: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if n == 0:
        return None, None, None
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method='wilson')
    return k / n, float(ci.low), float(ci.high)


def stratified_recall(sizes: Dict[str, int], labels: Dict[str, List[int]]) -> Optional[float]:
    est = {h: sizes[h] * (np.mean(labels[h]) if labels[h] else 0.0) for h in STRATA}
    total = sum(est.values())
    return est['accept'] / total if total > 0 else None


def bootstrap_recall(sizes, labels, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {h: np.asarray(labels[h], dtype=float) for h in STRATA}
    draws = []
    for _ in range(n_boot):
        resampled = {h: list(rng.choice(a, size=len(a), replace=True)) if len(a) else []
                     for h, a in arrays.items()}
        r = stratified_recall(sizes, resampled)
        if r is not None:
            draws.append(r)
    if not draws:
        return None, None
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def evaluate(decisions: Dict[str, Dict[str, str]], label_rows: List[Dict[str, str]],
             n_boot: int = 4000, seed: int = 0) -> dict:
    sizes = {h: 0 for h in STRATA}
    errors = 0
    for row in decisions.values():
        h = stratum_of(row)
        if h:
            sizes[h] += 1
        else:
            errors += 1
    labels: Dict[str, List[int]] = {h: [] for h in STRATA}
    unmatched = 0
    for row in label_rows:
        y = parse_label(row.get('label', ''))
        if y is None:
            continue
        decision = decisions.get(row.get('video_id', ''))
        h = stratum_of(decision) if decision else None
        if h is None:
            unmatched += 1
            continue
        labels[h].append(y)

    def proportion(h):
        p, lo, hi = wilson(sum(labels[h]), len(labels[h]))
        return {'labelled': len(labels[h]), 'usable': sum(labels[h]), 'estimate': p,
                'ci95': [lo, hi], 'population': sizes[h]}

    recall = stratified_recall(sizes, labels) if all(labels[h] or not sizes[h] for h in STRATA) else None
    recall_ci = bootstrap_recall(sizes, labels, n_boot, seed) if recall is not None else (None, None)
    total = sum(sizes.values())
    prevalence = (sum(sizes[h] * np.mean(labels[h]) for h in STRATA if labels[h]) / total
                  if total and all(labels[h] or not sizes[h] for h in STRATA) else None)
    return {
        'population': sizes, 'errors_excluded': errors, 'labels_unmatched': unmatched,
        'precision': proportion('accept'),
        'false_rejection_rate': {'metadata': proportion('reject:metadata'),
                                 'visual': proportion('reject:visual')},
        'recall': {'estimate': recall, 'ci95': list(recall_ci), 'method': 'stratified, bootstrap percentile'},
        'prevalence': None if prevalence is None else float(prevalence),
    }


def cmd_sample(args):
    decisions = load_decisions(args.decisions)
    rng = np.random.default_rng(args.seed)
    rows = []
    for h in STRATA:
        pool = sorted((r for r in decisions.values() if stratum_of(r) == h), key=lambda r: r['video_id'])
        take = min(args.per_stratum, len(pool))
        for i in sorted(rng.choice(len(pool), size=take, replace=False)) if take else []:
            r = pool[int(i)]
            rows.append({'video_id': r['video_id'], 'video_url': f"https://www.youtube.com/watch?v={r['video_id']}",
                         'title': r.get('title', ''), 'city': r.get('city', ''), 'stratum': h,
                         'stage': r.get('stage', ''), 'reason': r.get('reason', ''),
                         'score': r.get('score', ''), 'label': '', 'notes': ''})
        print(f"{h}: sampled {take} of {len(pool)}")
    # Shuffle so the labeller does not see the filter's verdict as an ordering cue.
    order = rng.permutation(len(rows))
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        w.writerows(rows[int(i)] for i in order)
    print(f"Wrote {len(rows)} rows to {args.out}. Fill 'label' with 1 (usable) or 0, ideally "
          f"without looking at the stratum/reason columns (hide them for blind labelling).")


def fmt(p):
    if p['estimate'] is None:
        return 'n/a (no labels)'
    lo, hi = p['ci95']
    return f"{p['estimate']:.1%} [{lo:.1%}, {hi:.1%}]  ({p['usable']}/{p['labelled']} labelled; N={p['population']})"


def cmd_evaluate(args):
    decisions = load_decisions(args.decisions)
    with open(args.labels, newline='', encoding='utf-8-sig') as f:
        label_rows = list(csv.DictReader(f))
    report = evaluate(decisions, label_rows, n_boot=args.bootstrap, seed=args.seed)
    print(f"Population: {report['population']}  (errors excluded: {report['errors_excluded']}, "
          f"labels not in decisions: {report['labels_unmatched']})")
    print(f"Precision (usable | accepted):            {fmt(report['precision'])}")
    print(f"False rejections, metadata stage:          {fmt(report['false_rejection_rate']['metadata'])}")
    print(f"False rejections, visual stage:            {fmt(report['false_rejection_rate']['visual'])}")
    rec = report['recall']
    if rec['estimate'] is None:
        print("Recall: n/a (every non-empty stratum needs labels)")
    else:
        print(f"Recall (stratified):                       {rec['estimate']:.1%} "
              f"[{rec['ci95'][0]:.1%}, {rec['ci95'][1]:.1%}]")
        print(f"Prevalence of usable videos (examined):    {report['prevalence']:.1%}")
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.json}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sample', help='Draw a stratified sample of decisions to label')
    s.add_argument('--decisions', action='append', required=True, help='decisions.csv (repeatable)')
    s.add_argument('--per-stratum', type=int, default=50)
    s.add_argument('--seed', type=int, default=1)
    s.add_argument('--out', default='labels.csv')
    s.set_defaults(func=cmd_sample)
    e = sub.add_parser('evaluate', help='Estimate precision/recall from labels')
    e.add_argument('--decisions', action='append', required=True, help='decisions.csv (repeatable)')
    e.add_argument('--labels', required=True)
    e.add_argument('--bootstrap', type=int, default=4000)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--json', default=None)
    e.set_defaults(func=cmd_evaluate)
    args = ap.parse_args(argv)
    for path in args.decisions:
        if not os.path.exists(path):
            print(f"Error: decisions file not found: {path}")
            sys.exit(1)
    args.func(args)


if __name__ == '__main__':
    main()
