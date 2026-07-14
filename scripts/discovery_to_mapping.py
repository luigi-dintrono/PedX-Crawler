#!/usr/bin/env python3
"""Convert crawler discovery.csv into PedX-Insight's input CSVs.

Bridges the Crawler -> Insight seam:
  discovery.csv (one row per found video)
    -> mapping_one_each.csv  (drives PedX-Insight run.py: download/analyze/delete)
    -> mapping.csv           (per-city row: consumed by Insight aggregators and by
                              --mode localize city inference; socio-economic columns
                              are external enrichment and stay blank)

Conventions preserved (from PedX-Insight):
  * mapping_one_each 'name' must contain NO underscore: run.py builds the video file
    {name}_{video_id}.mp4 and the analysis folder <name>_<video_id>, which everything
    downstream splits at the FIRST underscore (YouTube ids may contain underscores).
    We use dataset.py's convention: name = City (spaces/underscores removed) + index.
  * mapping.csv 'videos' is a bracketed, comma+space separated, unquoted id list:
    "[id1, id2]"; time_of_day/start_time/end_time/upload_date are nested lists
    aligned with videos, e.g. "[[0], [1]]".

Usage:
  python scripts/discovery_to_mapping.py \
      --discovery data/outputs/discovery.csv \
      --out-mapping-one-each mapping_one_each.csv \
      --out-mapping mapping.csv [--append]
"""

import argparse
import csv
import os
import re
import sys

# Minimal ISO2 -> country name map for the codes the crawler's region heuristic and
# cities.txt commonly produce; unknown codes pass through as-is.
ISO2_TO_NAME = {
    'US': 'United States', 'GB': 'United Kingdom', 'UK': 'United Kingdom',
    'DE': 'Germany', 'FR': 'France', 'IT': 'Italy', 'ES': 'Spain', 'NL': 'Netherlands',
    'JP': 'Japan', 'KR': 'South Korea', 'CN': 'China', 'TW': 'Taiwan', 'HK': 'Hong Kong',
    'IN': 'India', 'BD': 'Bangladesh', 'PK': 'Pakistan', 'TH': 'Thailand', 'VN': 'Vietnam',
    'ID': 'Indonesia', 'PH': 'Philippines', 'MY': 'Malaysia', 'SG': 'Singapore',
    'CA': 'Canada', 'MX': 'Mexico', 'BR': 'Brazil', 'AR': 'Argentina', 'CL': 'Chile',
    'AU': 'Australia', 'NZ': 'New Zealand', 'RU': 'Russia', 'TR': 'Turkey',
    'EG': 'Egypt', 'ZA': 'South Africa', 'NG': 'Nigeria', 'KE': 'Kenya',
    'PL': 'Poland', 'CZ': 'Czechia', 'AT': 'Austria', 'CH': 'Switzerland',
    'BE': 'Belgium', 'SE': 'Sweden', 'NO': 'Norway', 'DK': 'Denmark', 'FI': 'Finland',
    'PT': 'Portugal', 'GR': 'Greece', 'IE': 'Ireland', 'HU': 'Hungary', 'RO': 'Romania',
    'UA': 'Ukraine', 'IL': 'Israel', 'AE': 'United Arab Emirates', 'SA': 'Saudi Arabia',
}

MAPPING_HEADER = [
    'id', 'city', 'state', 'country', 'iso3', 'continent', 'lat', 'lon', 'gmp',
    'population_city', 'population_country', 'traffic_mortality', 'literacy_rate',
    'avg_height', 'med_age', 'gini', 'traffic_index', 'videos', 'time_of_day',
    'start_time', 'end_time', 'vehicle_type', 'upload_date', 'fps_list', 'channel'
]

MAPPING_ONE_EACH_HEADER = ['id', 'name', 'city', 'video', 'time_of_day',
                           'start_time', 'end_time', 'downloaded', 'finished']


def clean_city_slug(city):
    """City name safe for the {name}_{video_id} convention: no spaces/underscores."""
    return re.sub(r'[\s_]+', '', str(city).strip())


def parse_duration_seconds(hms):
    """'H:MM:SS' / 'MM:SS' / 'SS' -> int seconds; None when unparseable."""
    if hms is None:
        return None
    parts = str(hms).strip().split(':')
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if not parts:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def time_of_day_to_flag(value):
    """morning/afternoon -> 0 (day); evening/night -> 1; unknown/other -> 0."""
    return 1 if str(value).strip().lower() in ('evening', 'night') else 0


def published_to_upload_date(published_at):
    """ISO 8601 '2023-05-17T12:00:00Z' -> 20230517 (int); 0 when absent."""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', str(published_at or ''))
    return int(''.join(m.groups())) if m else 0


def bracketed(values):
    """Insight's list format: '[a,b,c]' (unquoted, no spaces — matches the native
    mapping.csv style; Insight's parsers strip whitespace so both forms read fine)."""
    return '[' + ','.join(str(v) for v in values) + ']'


def nested(values):
    """Insight's nested list format: '[[a],[b]]' (native no-space style)."""
    return '[' + ','.join(f'[{v}]' for v in values) + ']'


def parse_bracketed(raw):
    """Inverse of bracketed(): '[a, b]' -> ['a', 'b'] (tolerates quotes/blanks)."""
    if raw is None:
        return []
    inner = str(raw).strip().strip('[]')
    return [v.strip().strip("'\"") for v in inner.split(',') if v.strip()]


def parse_nested(raw):
    """Inverse of the nested format: '[[0], [12, 30]]' -> ['[0]', '[12, 30]'].

    Items keep their own brackets so they can be re-joined verbatim. A plain
    strip('[]')+split(',') would shred the inner lists — this splits on the
    '], [' boundaries between items instead.
    """
    s = str(raw or '').strip()
    if not (s.startswith('[') and s.endswith(']')):
        return []
    inner = s[1:-1].strip()
    if not inner:
        return []
    items = re.split(r'\]\s*,\s*\[', inner)
    return ['[' + item.strip().lstrip('[').rstrip(']') + ']' for item in items if item.strip()]


def read_csv_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--discovery', default='data/outputs/discovery.csv')
    ap.add_argument('--out-mapping-one-each', default='mapping_one_each.csv')
    ap.add_argument('--out-mapping', default='mapping.csv')
    ap.add_argument('--append', action='store_true',
                    help='Merge into existing output files instead of overwriting')
    args = ap.parse_args()

    if not os.path.exists(args.discovery):
        print(f"Error: discovery CSV not found: {args.discovery}")
        sys.exit(1)
    rows = read_csv_rows(args.discovery)
    print(f"Read {len(rows)} discovery rows from {args.discovery}")

    # --- Existing state (for --append and global dedup) ---
    one_each_rows = []
    known_ids = set()
    next_id = 1
    city_counters = {}
    if args.append and os.path.exists(args.out_mapping_one_each):
        one_each_rows = read_csv_rows(args.out_mapping_one_each)
        for r in one_each_rows:
            known_ids.add(r.get('video', ''))
            try:
                next_id = max(next_id, int(r.get('id', 0)) + 1)
            except ValueError:
                pass
            m = re.match(r'(.*?)(\d+)$', r.get('name', ''))
            if m:
                slug, n = m.group(1), int(m.group(2))
                city_counters[slug] = max(city_counters.get(slug, 0), n)

    # Ordered list of row dicts: mapping.csv can legitimately contain MULTIPLE rows for
    # the same city name (disambiguated by their videos lists — Insight's
    # find_city_in_mapping checks the link), so rows are never keyed/merged by city name.
    # New videos append to the FIRST row of their city; row order is preserved verbatim.
    mapping_rows = []
    city_first_row = {}
    if args.append and os.path.exists(args.out_mapping):
        mapping_rows = read_csv_rows(args.out_mapping)
        for i, r in enumerate(mapping_rows):
            city_first_row.setdefault(r.get('city', ''), i)
            known_ids.update(parse_bracketed(r.get('videos')))

    converted = deduped = bad = 0

    for row in rows:
        vid = (row.get('video') or row.get('id') or '').strip()
        city = (row.get('city') or '').strip()
        if not vid or not city:
            bad += 1
            continue
        if vid in known_ids:
            deduped += 1
            continue
        known_ids.add(vid)

        slug = clean_city_slug(city)
        city_counters[slug] = city_counters.get(slug, 0) + 1
        name = f"{slug}{city_counters[slug]}"

        tod = time_of_day_to_flag(row.get('time_of_day'))
        end_sec = parse_duration_seconds(row.get('end_time'))
        upload = published_to_upload_date(row.get('published_at'))
        channel = (row.get('channel_name') or '').strip()
        cc = (row.get('country_code') or row.get('region_code') or '').strip().upper()
        country = ISO2_TO_NAME.get(cc, cc if cc and cc != 'UNKNOWN' else '')

        one_each_rows.append({
            'id': next_id, 'name': name, 'city': city, 'video': vid,
            'time_of_day': tod, 'start_time': 0,
            'end_time': end_sec if end_sec is not None else '',
            'downloaded': '', 'finished': '',
        })
        next_id += 1

        row_idx = city_first_row.get(city)
        if row_idx is None:
            new_row = {col: '' for col in MAPPING_HEADER}
            new_row['city'] = city
            mapping_rows.append(new_row)
            row_idx = len(mapping_rows) - 1
            city_first_row[city] = row_idx
        entry = mapping_rows[row_idx]
        if country and not entry.get('country'):
            entry['country'] = country
        # Native formats: videos/upload_date/channel are FLAT lists ([a,b,...]);
        # time_of_day/start_time/end_time are NESTED ([[0],[1],...]).
        entry.setdefault('_videos', parse_bracketed(entry.get('videos')))
        entry.setdefault('_tod', parse_nested(entry.get('time_of_day')))
        entry.setdefault('_start', parse_nested(entry.get('start_time')))
        entry.setdefault('_end', parse_nested(entry.get('end_time')))
        entry.setdefault('_upload', parse_bracketed(entry.get('upload_date')))
        entry.setdefault('_channel', parse_bracketed(entry.get('channel')))
        entry['_videos'].append(vid)
        entry['_tod'].append(f'[{tod}]')
        entry['_start'].append('[0]')
        entry['_end'].append(f'[{end_sec if end_sec is not None else 0}]')
        entry['_upload'].append(str(upload))
        entry['_channel'].append(channel or 'unknown')

        converted += 1

    # --- Write mapping_one_each.csv ---
    with open(args.out_mapping_one_each, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MAPPING_ONE_EACH_HEADER, extrasaction='ignore')
        w.writeheader()
        w.writerows(one_each_rows)

    # --- Write mapping.csv (original row order preserved; untouched rows verbatim) ---
    existing_num_ids = [int(r['id']) for r in mapping_rows
                        if str(r.get('id', '')).strip().isdigit()]
    next_num_id = max(existing_num_ids, default=0) + 1
    touched = 0
    for entry in mapping_rows:
        if not str(entry.get('id', '')).strip():
            entry['id'] = next_num_id
            next_num_id += 1
        if '_videos' in entry:
            touched += 1
            entry['videos'] = bracketed(entry.pop('_videos'))
            entry['time_of_day'] = '[' + ','.join(entry.pop('_tod')) + ']'
            entry['start_time'] = '[' + ','.join(entry.pop('_start')) + ']'
            entry['end_time'] = '[' + ','.join(entry.pop('_end')) + ']'
            entry['upload_date'] = bracketed(entry.pop('_upload'))
            entry['channel'] = bracketed(entry.pop('_channel'))
    with open(args.out_mapping, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=MAPPING_HEADER, extrasaction='ignore')
        w.writeheader()
        w.writerows(mapping_rows)

    print(f"Converted {converted} videos ({deduped} duplicates skipped, {bad} bad rows)")
    print(f"Wrote {len(one_each_rows)} rows -> {args.out_mapping_one_each}")
    print(f"Wrote {len(mapping_rows)} city rows ({touched} touched) -> {args.out_mapping}")
    print("Note: mapping.csv socio-economic columns (lat/lon, population, ...) are left "
          "blank — they come from external enrichment, not the crawler.")


if __name__ == '__main__':
    main()
