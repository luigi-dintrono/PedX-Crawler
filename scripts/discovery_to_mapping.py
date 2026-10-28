#!/usr/bin/env python3
"""Convert crawler discovery.csv into PedX-Insight's input CSVs.

Bridges the Crawler -> Insight seam:
  discovery.csv (one row per found video)
    -> mapping_one_each.csv  (drives PedX-Insight run.py: download/analyze/delete)
    -> mapping.csv           (per-city row: consumed by Insight aggregators and by
                              --mode localize city inference)

Conventions preserved (from PedX-Insight):
  * mapping_one_each 'name' must contain NO underscore: run.py builds the video file
    {name}_{video_id}.mp4 and the analysis folder <name>_<video_id>, which everything
    downstream splits at the FIRST underscore (YouTube ids may contain underscores).
    We use dataset.py's convention: name = City (spaces/underscores removed) + index.
  * mapping.csv 'videos' is a bracketed, comma-separated, unquoted id list "[id1,id2]".
    Two different list shapes are aligned with it, matching the native file:
      - NESTED  ("[[0],[1]]"):  time_of_day, start_time, end_time
      - FLAT    ("[a,b]"):      upload_date, vehicle_type, fps_list, channel
  * channel holds YouTube channel IDs (UC...), like 99% of the native entries.
  * time_of_day is 0 (day) / 1 (night); a video whose time of day is unknown gets -1
    rather than being silently counted as daytime.
  * Rows this script CREATES get country / iso3 / continent / lat / lon / population
    from GeoNames (crawler/gazetteer.py); country and continent names reuse the
    spelling already in mapping.csv for the same iso3. Rows that already exist are
    never re-geocoded, and the remaining socio-economic columns (gmp, gini, ...) stay
    blank - they come from external enrichment.
  * upload_date is written as YYYYMMDD (e.g. 20230829). NOTE: most rows in the
    historical mapping.csv use an unpadded DDMMYYYY form (29082023, 3042024) which is
    ambiguous (3042024 reads as either 3/04 or 30/4), so this converter emits the
    unambiguous ISO-ordered form. No code in PedX-Insight or PedX-Visualizer parses
    this column today, so the two forms coexist without breaking anything.

Usage:
  python scripts/discovery_to_mapping.py \
      --discovery data/outputs/discovery.csv \
      --out-mapping-one-each mapping_one_each.csv \
      --out-mapping mapping.csv [--append]
"""

import argparse
import csv
import json
import math
import os
import re
import sys

# The geocoder lives with the crawler; this script is run as scripts/<name>.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'crawler'))
from gazetteer import Gazetteer  # noqa: E402
from contracts import CONTRACT_FIELDS, IDENTITY_FIELDS, contract_values  # noqa: E402

UNKNOWN_TIME_OF_DAY = -1
VIDEO_EVIDENCE_FIELDS = [
    'published_at', 'search_profile', 'search_query', 'search_language', 'search_duration',
    'search_rank', 'filter_backend', 'filter_score', 'frame_source', 'video_url',
    'channel_id', 'latitude', 'longitude',
]

MAPPING_HEADER = [
    'id', 'city', 'state', 'country', 'iso3', 'continent', 'lat', 'lon', 'gmp',
    'population_city', 'population_country', 'traffic_mortality', 'literacy_rate',
    'avg_height', 'med_age', 'gini', 'traffic_index', 'videos', 'time_of_day',
    'start_time', 'end_time', 'vehicle_type', 'upload_date', 'fps_list', 'channel'
] + ['country_code'] + CONTRACT_FIELDS + ['video_provenance']

MAPPING_ONE_EACH_HEADER = ['id', 'name', 'city', 'video', 'time_of_day',
                           'start_time', 'end_time', 'downloaded', 'finished',
                           'country_code'] + CONTRACT_FIELDS + VIDEO_EVIDENCE_FIELDS


def clean_city_slug(city):
    """City name safe for the {name}_{video_id} convention: no spaces/underscores."""
    return re.sub(r'[\s_]+', '', str(city).strip())


def parse_duration_seconds(hms):
    """'H:MM:SS' / 'MM:SS' / 'SS' -> int seconds; None when unparseable."""
    if hms is None:
        return None
    parts = str(hms).strip().split(':')
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    if not parts or any(not math.isfinite(p) or p < 0 for p in parts):
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return int(seconds) if seconds.is_integer() else seconds


def time_of_day_to_flag(value):
    """day/morning/afternoon -> 0; evening/night -> 1; unknown -> -1 (never guessed)."""
    value = str(value or '').strip().lower()
    if value in ('evening', 'night'):
        return 1
    if value in ('day', 'morning', 'afternoon'):
        return 0
    return UNKNOWN_TIME_OF_DAY


def format_number(value):
    """Match the native mapping.csv style for populations ('2600000.0'); '' for unknown."""
    return str(float(value)) if value else ''


def geocode_row(entry, cc, gazetteer, names_by_iso3):
    """Fill the location columns of a row this script created. Returns a warning or None."""
    country = gazetteer.country(cc) if cc else None
    if country:
        native_name, native_continent = names_by_iso3.get(country.iso3, (None, None))
        entry['country'] = native_name or country.name
        entry['iso3'] = country.iso3
        entry['continent'] = native_continent or country.continent
        entry['population_country'] = format_number(country.population)
    if not cc:
        # Without a country the name alone is too ambiguous to place (see gazetteer.py).
        return f"no country code for '{entry['city']}'; location columns left blank"
    city = gazetteer.resolve_city(entry['city'], cc, entry.get('admin1_code'), entry.get('geonameid'))
    if not city:
        return f"could not geocode '{entry['city']}' ({cc}); lat/lon left blank"
    entry['lat'], entry['lon'] = city.latitude, city.longitude
    entry['population_city'] = format_number(city.population)
    if city.ambiguous:
        return (f"'{entry['city']}' is ambiguous within {city.country_code}; used GeoNames "
                f"{city.geonameid} ({city.name}, pop {city.population:,}) - check lat/lon")
    return None


def published_to_upload_date(published_at):
    """ISO 8601 '2023-05-17T12:00:00Z' -> 20230517 (int); 0 when absent."""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', str(published_at or ''))
    return int(''.join(m.groups())) if m else 0


def bracketed(values):
    """Insight's list format: '[a,b,c]' (unquoted, no spaces — matches the native
    mapping.csv style; Insight's parsers strip whitespace so both forms read fine)."""
    return '[' + ','.join(str(v) for v in values) + ']'


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


CHANNEL_ID_RE = re.compile(r'/channel/(UC[\w-]{22})')


def channel_id_from_row(row):
    """YouTube channel ID (UC...) for a discovery row, or 'unknown'.

    The native mapping.csv 'channel' list holds channel IDs (99% of entries), not
    display names. Names can also contain commas, which would split an unquoted
    "[a,b]" list and misalign it with 'videos'. The ID is parsed from channel_url.
    """
    explicit = (row.get('channel_id') or '').strip()
    if re.fullmatch(r'UC[\w-]{22}', explicit):
        return explicit
    m = CHANNEL_ID_RE.search(row.get('channel_url') or '')
    return m.group(1) if m else 'unknown'


def write_csv_atomic(path, fieldnames, rows):
    """Write a CSV via a temp file + os.replace, so a crash mid-write cannot leave
    PedX-Insight's mapping files truncated."""
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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

    # Each output is reconciled independently. If a crash replaces only one file,
    # rerunning against the same discovery repairs the other instead of skipping it
    # because its ID exists in the union of both files.
    one_each_rows = []
    next_id = 1
    city_counters = {}
    if args.append and os.path.exists(args.out_mapping_one_each):
        one_each_rows = read_csv_rows(args.out_mapping_one_each)
        for r in one_each_rows:
            try:
                next_id = max(next_id, int(r.get('id', 0)) + 1)
            except ValueError:
                pass
            m = re.match(r'(.*?)(\d+)$', r.get('name', ''))
            if m:
                slug, n = m.group(1), int(m.group(2))
                city_counters[slug] = max(city_counters.get(slug, 0), n)

    mapping_rows = []
    names_by_iso3 = {}  # keep the native file's country/continent spelling for new rows
    if args.append and os.path.exists(args.out_mapping):
        mapping_rows = read_csv_rows(args.out_mapping)
        for r in mapping_rows:
            if r.get('iso3') and r.get('country'):
                names_by_iso3.setdefault(r['iso3'], (r['country'], r.get('continent') or None))
    gazetteer = Gazetteer()
    if not gazetteer.available:
        print("Warning: geonamescache is not installed; new mapping.csv rows get no lat/lon/iso3.")
    geo_warnings = []

    def identity_for(row):
        cc = (row.get('country_code') or row.get('region_code') or '').strip().upper()
        if cc == 'UNKNOWN':
            cc = ''
        cc = cc or gazetteer.country_code(row.get('iso3') or row.get('country'))
        admin = row.get('admin1_code') or row.get('state') or ''
        geo = row.get('geonameid') or ''
        explicit = row.get('searched_city_id') or row.get('city_id')
        if not explicit and cc:
            match = gazetteer.resolve_city(row.get('city', ''), cc, admin, geo)
            if match and not match.ambiguous:
                geo, admin = match.geonameid, match.admin1_code
        values = contract_values(row, country_code=cc, admin1_code=admin, geonameid=geo)
        return cc, values

    queue_by_video = {r.get('video'): r for r in one_each_rows}
    city_first_row, mapping_video_rows = {}, {}
    for i, entry in enumerate(mapping_rows):
        _, identity = identity_for(entry)
        city_first_row.setdefault(identity['city_id'], i)
        for vid in parse_bracketed(entry.get('videos')):
            mapping_video_rows.setdefault(vid, i)
    # A previous run may have replaced the queue and crashed before the mapping.
    # Repair from the durable queue even if this invocation has a different input.
    input_ids = {(r.get('video') or r.get('id') or '').strip() for r in rows}
    rows += [dict(r) for r in one_each_rows
             if r.get('video') not in mapping_video_rows and r.get('video') not in input_ids]

    # Preserve extension columns owned by users/downstream tools when widening files.
    queue_header = list(dict.fromkeys(MAPPING_ONE_EACH_HEADER +
                        [key for row in one_each_rows for key in row]))
    mapping_header = list(dict.fromkeys(MAPPING_HEADER +
                          [key for row in mapping_rows for key in row]))

    converted = deduped = bad = 0
    examined_ids = set()

    for row in rows:
        vid = (row.get('video') or row.get('id') or '').strip()
        city = (row.get('city') or '').strip()
        if not vid or not city:
            bad += 1
            continue
        if vid in examined_ids:
            deduped += 1
            continue
        examined_ids.add(vid)
        had_queue, had_mapping = vid in queue_by_video, vid in mapping_video_rows
        queue = queue_by_video.get(vid)
        # Existing names, analysis windows and job/verification state are authoritative.
        # They must survive both an append and a retry of a half-written bridge.
        merged = dict(row)
        if queue:
            merged.update({key: value for key, value in queue.items() if value not in ('', None)})
            # An explicit review may enrich an unverified queued record, but a
            # subsequent automatic discovery must never undo an existing review.
            if row.get('location_verification_source') and not queue.get('location_verification_source'):
                for key in ('verified_city_id', 'eligibility_status', 'location_verification_source'):
                    merged[key] = row.get(key, '')
            city = queue.get('city') or city
        cc, evidence = identity_for(merged)
        evidence.update({key: merged.get(key, '') or '' for key in VIDEO_EVIDENCE_FIELDS})
        tod_raw = merged.get('time_of_day')
        tod = int(tod_raw) if str(tod_raw).strip() in ('-1', '0', '1') else time_of_day_to_flag(tod_raw)
        start_sec = parse_duration_seconds(queue.get('start_time')) if queue else None
        if start_sec is None:
            start_sec = parse_duration_seconds(merged.get('clip_start_seconds'))
        if start_sec is None:
            start_sec = parse_duration_seconds(merged.get('start_time')) or 0
        end_sec = parse_duration_seconds(queue.get('end_time')) if queue else None
        if end_sec is None:
            end_sec = parse_duration_seconds(merged.get('clip_end_seconds'))
        if end_sec is None:
            end_sec = parse_duration_seconds(merged.get('end_time'))
        evidence['clip_start_seconds'] = start_sec
        evidence['clip_end_seconds'] = end_sec if end_sec is not None else ''
        upload = published_to_upload_date(row.get('published_at'))
        channel = channel_id_from_row(row)

        if queue is None:
            slug = clean_city_slug(city)
            city_counters[slug] = city_counters.get(slug, 0) + 1
            queue = {
                'id': next_id, 'name': f"{slug}{city_counters[slug]}", 'city': city, 'video': vid,
                'time_of_day': tod, 'start_time': start_sec,
                'end_time': end_sec if end_sec is not None else '',
                'downloaded': '', 'finished': '', 'country_code': cc, **evidence,
            }
            one_each_rows.append(queue)
            queue_by_video[vid] = queue
            next_id += 1
        else:
            for key, value in {'country_code': cc, **evidence}.items():
                if queue.get(key) in ('', None):
                    queue[key] = value
            if evidence.get('location_verification_source'):
                for key in ('verified_city_id', 'eligibility_status', 'location_verification_source'):
                    queue[key] = evidence[key]

        row_idx = mapping_video_rows.get(vid, city_first_row.get(evidence['city_id']))
        if row_idx is None:
            new_row = {col: '' for col in MAPPING_HEADER}
            new_row.update(city=city, country_code=cc,
                           **{key: evidence[key] for key in IDENTITY_FIELDS})
            # Verification belongs to each video, not to an entire city row.
            new_row.update(verified_city_id='', eligibility_status='unverified',
                           location_verification_source='')
            warning = geocode_row(new_row, cc, gazetteer, names_by_iso3)
            if warning:
                geo_warnings.append(warning)
            if not new_row['country'] and cc:
                new_row['country'] = cc  # unknown to GeoNames: pass the code through
            mapping_rows.append(new_row)
            row_idx = len(mapping_rows) - 1
            city_first_row[evidence['city_id']] = row_idx
        entry = mapping_rows[row_idx]
        existing_cc, city_evidence = identity_for(entry)
        for key in ('city_id', 'searched_city_id', 'admin1_code', 'geonameid', 'city_identity_source'):
            if not entry.get(key):
                entry[key] = city_evidence[key]
        if not entry.get('country_code'):
            entry['country_code'] = existing_cc
        if not entry.get('eligibility_status'):
            entry['eligibility_status'] = 'unverified'
        # Keep one evidence object per video; scalar dates, clips and eligibility would
        # incorrectly apply one video's claims to every video in the city's list.
        try:
            per_video = json.loads(entry.get('video_provenance') or '{}')
        except (TypeError, ValueError):
            raise ValueError(f"Invalid video_provenance for city {entry.get('city')!r}")
        if not isinstance(per_video, dict):
            raise ValueError('video_provenance must be an object keyed by video ID')
        previous = per_video.get(vid) or {}
        updated = {**previous, **{k: v for k, v in evidence.items() if v not in ('', None)}}
        if previous.get('location_verification_source') and not evidence.get('location_verification_source'):
            for key in ('verified_city_id', 'eligibility_status', 'location_verification_source'):
                updated[key] = previous.get(key, '')
        per_video[vid] = updated
        entry['video_provenance'] = json.dumps(per_video, ensure_ascii=False, sort_keys=True)
        if had_mapping:
            deduped += int(had_queue)
            converted += int(not had_queue)
            continue
        # Native formats: videos/upload_date/channel are FLAT lists ([a,b,...]);
        # time_of_day/start_time/end_time are NESTED ([[0],[1],...]).
        entry.setdefault('_videos', parse_bracketed(entry.get('videos')))
        entry.setdefault('_tod', parse_nested(entry.get('time_of_day')))
        entry.setdefault('_start', parse_nested(entry.get('start_time')))
        entry.setdefault('_end', parse_nested(entry.get('end_time')))
        entry.setdefault('_upload', parse_bracketed(entry.get('upload_date')))
        entry.setdefault('_channel', parse_bracketed(entry.get('channel')))
        entry.setdefault('_vtype', parse_bracketed(entry.get('vehicle_type')))
        entry.setdefault('_fps', parse_bracketed(entry.get('fps_list')))
        entry['_videos'].append(vid)
        entry['_tod'].append(f'[{tod}]')
        entry['_start'].append(f'[{start_sec}]')
        entry['_end'].append(f'[{end_sec if end_sec is not None else 0}]')
        entry['_upload'].append(str(upload))
        entry['_channel'].append(channel)
        # vehicle_type/fps_list are analysis-derived and unknown to the crawler, but they
        # are aligned element-for-element with 'videos'. Extend them only when the row
        # already has them populated (crawler-only rows keep them blank), padding with
        # 0 = unknown rather than a fabricated plausible value (e.g. fps 30), so the
        # lists never fall out of step with 'videos'.
        if entry['_vtype']:
            entry['_vtype'].append('0')
        if entry['_fps']:
            entry['_fps'].append('0')
        mapping_video_rows[vid] = row_idx

        converted += 1

    # --- Write mapping_one_each.csv ---
    write_csv_atomic(args.out_mapping_one_each, queue_header, one_each_rows)

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
            # Only rewrite these two when the row actually had them; otherwise leave
            # the original blank cell alone.
            vtype, fps = entry.pop('_vtype'), entry.pop('_fps')
            if vtype:
                entry['vehicle_type'] = bracketed(vtype)
            if fps:
                entry['fps_list'] = bracketed(fps)
    write_csv_atomic(args.out_mapping, mapping_header, mapping_rows)

    print(f"Converted {converted} videos ({deduped} duplicates skipped, {bad} bad rows)")
    print(f"Wrote {len(one_each_rows)} rows -> {args.out_mapping_one_each}")
    print(f"Wrote {len(mapping_rows)} city rows ({touched} touched) -> {args.out_mapping}")
    for warning in geo_warnings:
        print(f"Geocoding: {warning}")
    print("Note: new mapping.csv rows get location columns from GeoNames; socio-economic "
          "columns (gmp, gini, traffic_index, ...) are left blank for external enrichment.")


if __name__ == '__main__':
    main()
