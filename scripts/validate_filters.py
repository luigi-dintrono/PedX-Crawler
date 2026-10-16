#!/usr/bin/env python3
"""Reproduce the filter validation reported in crawler/QUALITY_FILTER_README.md.

No hand labels are needed, but the labels are weak (see the README's caveats):

  positives  random videos from PedX-Insight's curated mapping.csv
  negatives  search results for queries that should NOT count as road-user footage
  frames     YouTube's public thumbnail frames (mq1-mq3), so no cookies / PO token needed

Steps (each resumable; all state lives in --out-dir):

  collect   pick positives, run the negative searches (100 API units per query), fetch
            thumbnail frames and video metadata            -> set.json
  score     YOLO street-scene fraction + frame brightness (and optionally InternVL3 yes
            probabilities) for every video                 -> scores.json
  report    sensitivity / specificity (Wilson 95% intervals) for metadata, YOLO, InternVL3
            and their combinations, per negative group     -> printed + report.json
  brightness  calibrate the day/night luma thresholds from title cues (300 API units)

Example:
  python scripts/validate_filters.py collect --mapping ../PedX-Insight/mapping.csv --out-dir val
  python scripts/validate_filters.py score --out-dir val [--internvl3-model OpenGVLab/InternVL3-1B-hf]
  python scripts/validate_filters.py report --out-dir val
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import median

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'crawler'))
import filter_common as fc  # noqa: E402

NEGATIVE_QUERIES = {
    'easy:vlog': 'morning routine vlog',
    'easy:cooking': 'cooking recipe tutorial',
    'easy:podcast': 'podcast interview full episode',
    'easy:game': 'minecraft survival lets play',
    'easy:music': 'piano tutorial for beginners',
    'hard:drone': 'Tokyo drone footage 4K',
    'hard:timelapse': 'city traffic timelapse 4K',
    'hard:gta': 'GTA 5 driving around the city gameplay',
    'hard:train': 'train cab ride driver view',
    'hard:car-interior': 'car interior review walkaround',
}
BRIGHTNESS_QUERIES = ['night drive city 4K', 'evening sunset drive city 4K',
                      'daytime driving tour city 4K', 'morning drive city 4K']
NIGHT_CUE = re.compile(r'\bnight\b|\bnighttime\b|noche|nuit|\bnacht\b', re.I)
DAY_CUE = re.compile(r'\bmorning\b|\bafternoon\b|\bmidday\b|\bdaytime\b|\bday\b|\bnoon\b', re.I)
THUMBS = ('mq1', 'mq2', 'mq3')
DASHCAM_EXTRA_NEGATIVES = ('drone', 'cab ride', 'cockpit', 'train ride')


def _path(out_dir, name):
    return os.path.join(out_dir, name)


def _youtube():
    from dotenv import load_dotenv
    from googleapiclient.discovery import build
    load_dotenv()
    key = os.environ.get('YOUTUBE_API_KEY')
    if not key:
        sys.exit('Error: set YOUTUBE_API_KEY (or put it in .env)')
    return build('youtube', 'v3', developerKey=key)


def _iso_seconds(duration):
    m = re.fullmatch(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?', duration or '')
    d, h, mi, s = (int(g or 0) for g in m.groups()) if m else (0, 0, 0, 0)
    return d * 86400 + h * 3600 + mi * 60 + s


def _download_thumbnails(videos, out_dir, workers=4):
    import requests
    session = requests.Session()
    frames_dir = _path(out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    def fetch(vid):
        got = []
        for name in THUMBS:
            p = os.path.join(frames_dir, f'{vid}.{name}.jpg')
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                for _ in range(4):
                    try:
                        r = session.get(f'https://i.ytimg.com/vi/{vid}/{name}.jpg', timeout=30)
                    except requests.RequestException:
                        continue
                    if r.status_code == 200 and r.content:
                        with open(p, 'wb') as f:
                            f.write(r.content)
                    break
            if os.path.exists(p) and os.path.getsize(p) > 0:
                got.append(p)
        return vid, got

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(fetch, videos))


# ---------------------------------------------------------------------------- collect
def cmd_collect(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(args.mapping, encoding='utf-8-sig')))
    ids = [v.strip() for r in rows for v in r['videos'].strip('[]').split(',') if v.strip()]
    random.Random(args.seed).shuffle(ids)
    items = {v: {'label': 1, 'group': 'curated'} for v in ids[:args.n_pos]}

    cache = _path(args.out_dir, 'negatives_search.json')
    if os.path.exists(cache):
        negatives = json.load(open(cache, encoding='utf-8'))
        print(f'{len(negatives)} negatives loaded from cache (no API calls)')
    else:
        yt = _youtube()
        negatives = {}
        for group, q in NEGATIVE_QUERIES.items():
            resp = yt.search().list(q=q, part='id,snippet', type='video', maxResults=args.per_query,
                                    videoDuration='medium', publishedAfter='2020-01-01T00:00:00Z').execute()
            for it in resp.get('items', []):
                vid = it['id']['videoId']
                if vid not in items and vid not in negatives:
                    negatives[vid] = {'label': 0, 'group': group, 'query': q}
        json.dump(negatives, open(cache, 'w', encoding='utf-8'), ensure_ascii=False)
    items.update(negatives)

    frames = _download_thumbnails(items, args.out_dir)
    items = {v: {**m, 'frames': frames[v]} for v, m in items.items() if len(frames.get(v, [])) >= 2}

    yt, ids = _youtube(), list(items)                     # metadata for the metadata filter (1 unit / 50 videos)
    for i in range(0, len(ids), 50):
        for it in yt.videos().list(part='snippet,contentDetails', id=','.join(ids[i:i + 50])).execute()['items']:
            sn, cd = it['snippet'], it['contentDetails']
            items[it['id']].update(name=sn['title'], tags=sn.get('tags') or [], description=sn.get('description', ''),
                                   duration_seconds=_iso_seconds(cd['duration']),
                                   live_broadcast=sn.get('liveBroadcastContent', 'none'),
                                   projection=cd.get('projection', ''))
    items = {v: m for v, m in items.items() if 'name' in m}
    json.dump(items, open(_path(args.out_dir, 'set.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    pos = sum(m['label'] for m in items.values())
    print(f'set.json: {pos} positives, {len(items) - pos} negatives')


# ---------------------------------------------------------------------------- score
def cmd_score(args):
    items = json.load(open(_path(args.out_dir, 'set.json'), encoding='utf-8'))
    out_path = _path(args.out_dir, 'scores.json')
    scores = json.load(open(out_path, encoding='utf-8')) if os.path.exists(out_path) else {}

    from video_quality_filter_yolo import VideoQualityFilter
    yolo = VideoQualityFilter(yolo_model_path=args.yolo_model, frame_source='thumbnails',
                              temp_dir=_path(args.out_dir, 'tmp'))
    ivl = None
    if args.internvl3_model:
        from video_quality_filter_internvl3 import VideoQualityFilterInternVL3
        ivl = VideoQualityFilterInternVL3(model_name=args.internvl3_model, device=args.device,
                                          frame_source='thumbnails', temp_dir=_path(args.out_dir, 'tmp'))
    # negatives interleaved across groups, so a partial run is still stratified
    by_group = defaultdict(list)
    for v, m in items.items():
        by_group[m['group']].append(v)
    order = [by_group[g][k] for k in range(max(map(len, by_group.values()))) for g in sorted(by_group)
             if k < len(by_group[g])]
    for n, vid in enumerate(order, 1):
        rec = scores.setdefault(vid, {})
        frames = items[vid]['frames']
        if 'street_fraction' not in rec:
            s, d = yolo.score_frames(frames)
            rec.update(street_fraction=sum(s) / len(s), pedestrian=any(x['pedestrian'] for x in d),
                       person_frames=sum(1 for x in d if x['counts'].get('person')),
                       luma=[round(fc.frame_luma(p), 1) for p in frames])
        if ivl and 'internvl' not in rec:
            s, _ = ivl.score_frames(frames)
            rec['internvl'] = sum(s) / len(s)
        if n % 25 == 0:
            json.dump(scores, open(out_path, 'w'))
            print(f'  {n}/{len(order)}', flush=True)
    json.dump(scores, open(out_path, 'w'))
    print(f'scores.json: {len(scores)} videos')


# ---------------------------------------------------------------------------- report
def wilson(k, n):
    from scipy.stats import binomtest
    c = binomtest(k, n).proportion_ci(0.95, method='wilson')
    return k / n, float(c.low), float(c.high)


def fmt(k, n):
    if not n:
        return 'n/a'
    p, lo, hi = wilson(k, n)
    return f'{p:5.1%} [{lo:.1%}-{hi:.1%}] ({k}/{n})'


def build_report(items, scores, ivl_threshold=0.5, yolo_threshold=0.5):
    """Sensitivity/specificity tables from set.json + scores.json (pure; unit-tested)."""
    meta = fc.MetadataFilter(negative_keywords=fc.COMMON_NEGATIVE_KEYWORDS + DASHCAM_EXTRA_NEGATIVES)
    rows = []
    for vid, m in items.items():
        s = scores.get(vid)
        if not s or 'street_fraction' not in s:
            continue
        rows.append({
            'vid': vid, 'label': m['label'], 'group': m['group'],
            'meta': meta.check({**m, 'id': vid}).accepted,
            'yolo': s['street_fraction'] >= yolo_threshold, 'person': s['pedestrian'],
            'ivl': (s['internvl'] >= ivl_threshold) if 'internvl' in s else None})
    pos = [r for r in rows if r['label'] == 1]
    neg = [r for r in rows if r['label'] == 0]
    configs = {
        'metadata only': lambda r: r['meta'],
        'YOLO street-scene only': lambda r: r['yolo'],
        'metadata + YOLO (default)': lambda r: r['meta'] and r['yolo'],
        'metadata + YOLO + person': lambda r: r['meta'] and r['yolo'] and r['person'],
    }
    if all(r['ivl'] is not None for r in rows) and rows:
        configs['InternVL3 only'] = lambda r: r['ivl']
        configs['metadata + InternVL3'] = lambda r: r['meta'] and r['ivl']
    table = {}
    for name, rule in configs.items():
        table[name] = {'kept_curated': [sum(rule(r) for r in pos), len(pos)],
                       'rejected_negatives': [sum(not rule(r) for r in neg), len(neg)]}
    groups = defaultdict(list)
    for r in neg:
        groups[r['group']].append(r)
    by_group = {g: {name: [sum(not rule(r) for r in rs), len(rs)] for name, rule in configs.items()}
                for g, rs in sorted(groups.items())}
    return {'n_positives': len(pos), 'n_negatives': len(neg), 'configs': table, 'by_group': by_group}


def cmd_report(args):
    items = json.load(open(_path(args.out_dir, 'set.json'), encoding='utf-8'))
    scores = json.load(open(_path(args.out_dir, 'scores.json'), encoding='utf-8'))
    rep = build_report(items, scores, args.internvl3_threshold, args.yolo_threshold)
    print(f"{rep['n_positives']} curated positives, {rep['n_negatives']} query-defined negatives\n")
    print(f'{"configuration":30}{"sensitivity (curated kept)":36}{"specificity (negatives rejected)":36}')
    for name, t in rep['configs'].items():
        print(f'{name:30}{fmt(*t["kept_curated"]):36}{fmt(*t["rejected_negatives"]):36}')
    print('\nNegatives rejected, by group:')
    for g, d in rep['by_group'].items():
        print(f'  {g:18}' + '  '.join(f'{n.split(" (")[0]}: {k}/{tot}' for n, (k, tot) in d.items()))
    json.dump(rep, open(_path(args.out_dir, 'report.json'), 'w'), indent=2)


# ---------------------------------------------------------------------------- brightness
def cmd_brightness(args):
    os.makedirs(args.out_dir, exist_ok=True)
    cache = _path(args.out_dir, 'brightness_titles.json')
    if os.path.exists(cache):
        titles = json.load(open(cache, encoding='utf-8'))
    else:
        yt, titles = _youtube(), {}
        for q in BRIGHTNESS_QUERIES:
            resp = yt.search().list(q=q, part='id,snippet', type='video', maxResults=50, videoDuration='medium',
                                    publishedAfter='2020-01-01T00:00:00Z').execute()
            titles.update({it['id']['videoId']: it['snippet']['title'] for it in resp['items']})
        json.dump(titles, open(cache, 'w', encoding='utf-8'), ensure_ascii=False)
    labels = {}
    for vid, title in titles.items():
        n, d = bool(NIGHT_CUE.search(title)), bool(DAY_CUE.search(title))
        if n != d:
            labels[vid] = 'night' if n else 'day'
    frames = _download_thumbnails(labels, args.out_dir)
    data = [(labels[v], median(fc.frame_luma(p) for p in fs)) for v, fs in frames.items() if len(fs) >= 2]
    print(f'{len(data)} videos with an unambiguous title cue '
          f'({sum(1 for l, _ in data if l == "night")} night, {sum(1 for l, _ in data if l == "day")} day)')
    for nmax, dmin in ((fc.NIGHT_MAX_LUMA, fc.DAY_MIN_LUMA), (55, 80)):
        print(f'thresholds night<={nmax:g}, day>={dmin:g}:')
        for lab in ('night', 'day'):
            vals = [m for l, m in data if l == lab]
            pred = ['night' if m <= nmax else 'day' if m >= dmin else 'unknown' for m in vals]
            print(f'  title says {lab:5}: ' + ', '.join(f'{p}={pred.count(p)}' for p in ('night', 'unknown', 'day'))
                  + f'   (n={len(vals)})')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect', help='Build the validation set')
    c.add_argument('--mapping', required=True, help="PedX-Insight's curated mapping.csv")
    c.add_argument('--out-dir', required=True)
    c.add_argument('--n-pos', type=int, default=200)
    c.add_argument('--per-query', type=int, default=40)
    c.add_argument('--seed', type=int, default=2026)
    c.set_defaults(func=cmd_collect)
    s = sub.add_parser('score', help='Score every video with YOLO (and optionally InternVL3)')
    s.add_argument('--out-dir', required=True)
    s.add_argument('--yolo-model', default='yolo26n.pt')
    s.add_argument('--internvl3-model', default=None, help='e.g. OpenGVLab/InternVL3-1B-hf (omit to skip)')
    s.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    s.set_defaults(func=cmd_score)
    r = sub.add_parser('report', help='Sensitivity / specificity tables')
    r.add_argument('--out-dir', required=True)
    r.add_argument('--yolo-threshold', type=float, default=0.5)
    r.add_argument('--internvl3-threshold', type=float, default=0.5)
    r.set_defaults(func=cmd_report)
    b = sub.add_parser('brightness', help='Check day/night thresholds against title cues')
    b.add_argument('--out-dir', required=True)
    b.set_defaults(func=cmd_brightness)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
