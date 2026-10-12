# PedX Crawler

Discovers YouTube road-user videos by city and writes a CSV for the PedX pipeline
(**Crawler** → [PedX-Insight](../PedX-Insight) → [PedX-Visualizer](../PedX-Visualizer)).

What it searches for is set by a **search profile**. The default, `dashcam`, targets
in-vehicle driving footage, because that is what PedX-Insight's curated dataset contains:
in a random sample of 98 of its 8,608 videos, 56 titles contain "driving" and 29 "drive",
none mention walking or crossings, and the analysed segments run from 155 s (1st
percentile) to 14,000 s (99th), median 1,499 s. (The crawler's original behaviour - short
clips of pedestrian crossings - is still available as `--profile crossing`.)

## How a run works

```
cities file ─► search streams ─► candidates ─► metadata filter ─► visual filter ─► discovery.csv
 (City,CC,     (query × language   (YouTube      (stage 1: always   (stage 2: optional,   + runs/<id>/
  Local name)   × duration)         search)       on, no ML)         YOLO or InternVL3)     decisions.csv
                                                                                            manifest.json
```

1. **Plan** - each city is resolved offline with GeoNames (country, coordinates), then
   expanded into *search streams*: the profile's query template in the country's language
   and in English, crossed with YouTube's `videoDuration` buckets (`medium` + `long`) and
   `videoDefinition=high`. Queries are localized: `Tokyo 車載動画`, `Moscow на машине`, ...
2. **Search** - `sampling=balanced` (default) fetches one page from *every* stream per
   round and examines results interleaved by rank, so no single query or language dominates
   the sample. `--sampling greedy` stops as soon as the target is met (cheapest, but the
   first stream dominates).
3. **Metadata filter** (always on) - rejects live streams, videos outside 4 min - 3 h, 360°
   video, `#shorts`, game footage / simulators, compilations, time-lapses and videos whose
   title says they are sped up (`3x Speed`). See [crawler/QUALITY_FILTER_README.md](crawler/QUALITY_FILTER_README.md).
4. **Visual filter** (optional) - samples 5 frames spread across the video, scores them with
   YOLO or InternVL3 and accepts or rejects. **Read [Visual filter access](#visual-filter-access)
   first: it needs login cookies or a PO-token plugin to work on most videos.**
5. **Record** - every examined candidate is logged with its search provenance and verdict.

## Quick start

```bash
make install-dev          # core + test deps (or: make install-yolo / install-internvl3)
make setup                # copies .env.example -> .env; add YOUTUBE_API_KEY
make test-unit            # offline test suite, no key or network needed
make dry-run              # print the search plan and quota estimate; spends nothing
make run                  # crawl data/cities.txt with the metadata filter
```

On Windows pass `PYTHON=py` (e.g. `make test-unit PYTHON=py`); `python3` is usually a
Microsoft Store stub there. Install the visual-filter extras only if you need them -
`pip install -e ".[yolo]"` or `".[internvl3]"` - and put `ffmpeg` on your `PATH`.

Get a YouTube Data API v3 key from the [Google Cloud Console](https://console.cloud.google.com/apis/library/youtube.googleapis.com).
It is read from `--api-key`, `--api-key-file` or `$YOUTUBE_API_KEY` (`.env` is loaded).

### Cities file

```
# City | City,CC | City,CC,Local name
London,GB
Tokyo,JP,東京
Portland,US
```

Always give the country code where you can: bare names are ambiguous (Portland, OR vs
ME; London, GB vs CA). The crawler resolves a missing code from GeoNames by population and
**warns when a rival candidate has at least 10% of the chosen city's population**. The
local name is used in non-English queries.

## Search profiles

| Profile | Looks for | Duration window | Title-keyword gate (`--require-title-keywords`) |
|---|---|---|---|
| `dashcam` (default) | in-vehicle driving footage; ~25 languages | 4 min - 3 h | driving / drive / dashcam / 車載 / 行车 / ... |
| `walking` | walking tours at street level | 4 min - 3 h | walk / caminando / 散歩 / ... |
| `crossing` | short static clips of street crossings (original behaviour) | 1 - 20 min | crosswalk / zebra / 横断歩道 / ... |

Templates are in [crawler/search_profiles.py](crawler/search_profiles.py). They are a
starting point: **have native speakers review the languages you rely on.** The 4 min - 3 h
dashcam window keeps about 96% of the curated PedX-Insight videos (2.2% are shorter than
4 min, 1.9% longer than 3 h).

## Options

| Option | Description | Default |
|---|---|---|
| `--cities-file PATH` | Cities list (format above) | `data/cities.txt` |
| `--output PATH` | Output CSV | `data/outputs/discovery.csv` |
| `--append` | Append to `--output` (an older schema is widened in place) instead of writing a new numbered file | off |
| `--exclude-existing CSV` | Skip IDs already in this CSV (repeatable) | - |
| `--runs-dir DIR` | Where `runs/<run id>/` goes | `<output dir>/runs` |
| `--per-city N` | Target number of *accepted* videos per city | 50 |
| `--since` / `--until YYYY-MM-DD` | Publication window; set both for a reproducible crawl | `2020-01-01` / none |
| `--quota-limit N` | Daily quota budget; the crawler stops before exceeding it | 10000 |
| `--dry-run` | Print the plan + quota estimate, no API calls | off |
| `--profile {dashcam,walking,crossing}` | What to search for | `dashcam` |
| `--languages` | `auto` (country language + English), `none` (English, no `relevanceLanguage`), or e.g. `ja,en` | `auto` |
| `--video-duration` | YouTube `videoDuration` buckets, comma-separated from `any,short,medium,long` | profile's |
| `--definition {any,high,standard}` | YouTube `videoDefinition` | profile's |
| `--sampling {balanced,greedy}` | See [How a run works](#how-a-run-works) | `balanced` |
| `--max-pages-per-city N` | Search pages (100 units each) allowed per city | 10 |
| `--use-multiple-search` | Use every query template of the profile, not just the first per language | off |
| `--no-metadata-filter` | Raw crawl, stage 1 off | off |
| `--min-duration` / `--max-duration S` | Metadata duration window in seconds (`0` = no max) | profile's |
| `--max-upload-months N` | Reject videos older than N months before `--until` / run start | off |
| `--require-title-keywords` | Also require a profile keyword in the title or tags | off |
| `--filter {none,yolo,internvl3}` | Visual filter backend | none |
| `--threshold F` | Acceptance threshold on the mean frame score | 0.5 |
| `--frames N` | Frames sampled across each video | 5 |
| `--yolo-model`, `--yolo-conf` | YOLO checkpoint and detection confidence | `yolo26n.pt`, 0.25 |
| `--internvl3-model`, `--device` | InternVL3 `-hf` checkpoint and device | `OpenGVLab/InternVL3-8B-hf`, `auto` |
| `--ytdlp-cookies FILE` | `cookies.txt` for frame sampling | - |
| `--ytdlp-js-runtime` | JS runtime for yt-dlp (`auto` = deno or node, `none` to disable) | `auto` |
| `--max-consecutive-errors N` | Abort after N visual checks in a row could not be completed (`0` = never) | 5 |

The legacy `--enable-quality-filter` / `--filter-type` pair still works; `--filter` overrides it.

### Quota

Search pages cost 100 units each; details are fetched in batches (1 unit per 50 videos).
With `balanced` sampling a city costs at least one page per stream: a non-English country
with the default profile has 4 streams (local + English × medium + long), an English one 2.
Worst case is `--max-pages-per-city` × 100 units. `--dry-run` prints the exact streams and
a min-max estimate.

## Visual filter access

Frame sampling reads a few seconds from several points in each video. **In October 2026,
on a machine with no cookies and no PO-token plugin, yt-dlp could reach only the first
~6 MB of the one video tested, and any seek deeper into the stream failed with HTTP 403**
(YouTube's current player clients appear to require a PO token or a logged-in session for
the rest). yt-dlp's own full download of that video failed the same way, so
PedX-Insight's `run.py` downloads may be subject to the same requirement. This was
observed on a single video and network, not surveyed.

What the crawler does about it:

* ffmpeg cannot read YouTube URLs directly (it sends open-ended range requests, which are
  refused). Frames are read through a small localhost range-chunking proxy
  ([crawler/video_access.py](crawler/video_access.py)), which works whenever the stream is accessible.
* If YouTube refuses the media request, the visual check fails **closed** (the video is
  not accepted) with a message naming the fix, and **the run aborts after
  `--max-consecutive-errors` (default 5) failures in a row**, saving what it has.

To get the visual filter working, give yt-dlp either:

1. **Login cookies:** `--ytdlp-cookies cookies.txt` (export from a browser you are signed
   in with; treat the file as a credential and keep it out of git), or
2. **A PO-token provider plugin** installed next to yt-dlp, e.g.
   [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider);
   yt-dlp loads installed plugins automatically.

Without either, run `--filter none` (metadata filter only) and judge frames some other way.

## Output

### `discovery.csv`

| Column(s) | Meaning |
|---|---|
| `id`, `video`, `video_url`, `name`, `city` | video id, URL, (HTML-unescaped) title, city |
| `start_time`, `end_time`, `duration_seconds` | `0:00`, duration `H:MM:SS`, duration in seconds |
| `time_of_day`, `time_of_day_source` | `night`/`day`/`morning`/... or `unknown`; source is `frames` (median frame brightness), `title` or `none` |
| `country_code`, `country_code_source`, `region_code` | ISO2; source `cities_file` or `gazetteer`. `region_code` is the code passed as `regionCode` |
| `channel_name`, `channel_url`, `channel_id` | channel (the ID is what PedX-Insight stores) |
| `published_at`, `view_count`, `like_count`, `comment_count`, `thumbnail_url`, `latitude`, `longitude` | free enrichment from the same `videos.list` call; `latitude`/`longitude` are the uploader's geotag, only when set |
| `definition`, `projection`, `default_audio_language`, `live_broadcast` | content metadata |
| `search_profile`, `search_query`, `search_language`, `search_duration`, `search_rank` | which stream found it, and at what rank |
| `filter_backend`, `filter_score`, `run_id` | filter that accepted it, its score, the run |

Columns are only ever appended, so older files are widened in place by `--append`.

### Run records: `runs/<run id>/`

* `manifest.json` - arguments, all parameters, blocked keywords, tool versions
  (Python, yt-dlp, ultralytics, ...), git commit and dirty flag, per-city outcome
  (resolved country / GeoNames id / coordinates, streams, counts), quota used, status
  (`completed`, `interrupted`, `quota_exceeded`, `visual_filter_failed`).
* `decisions.csv` - **one row per examined candidate**: stream, rank, metadata, stage,
  `accept` / `reject` / `error`, reason, score, per-frame timestamps, scores and brightness,
  filter model and threshold. This is the audit trail for a PRISMA-style flow diagram.

### Validating the filters

The thresholds have not been validated against labelled data. `scripts/evaluate_filter.py`
does that:

```bash
make label-sample DECISIONS=data/outputs/runs/<id>/decisions.csv    # writes labels.csv
# fill the 'label' column (1 = usable for PedX-Insight, 0 = not); hide the 'stratum'/'reason' columns to label blind
make evaluate DECISIONS=data/outputs/runs/<id>/decisions.csv        # precision, recall, false-rejection rates
```

It samples equally from accepted / metadata-rejected / visual-rejected decisions and
reports precision (Wilson interval), per-stage false-rejection rate and a stratified
recall with a bootstrap 95% interval. `error` decisions are excluded and counted.

## Feeding PedX-Insight

```bash
make bridge                       # = scripts/discovery_to_mapping.py --append against ../PedX-Insight
make bridge INSIGHT=/path/to/PedX-Insight DISCOVERY=data/outputs/discovery.csv
```

* `mapping_one_each.csv` drives Insight's `run.py`. Names are underscore-free
  (`London1`, `Toronto2`, ...) so the `<name>_<videoid>` folder convention splits at the
  first underscore even for YouTube IDs that contain one.
* `mapping.csv` gets one row per city. **New rows are geocoded offline from GeoNames**:
  `country`, `iso3`, `continent`, `lat`, `lon`, `population_city`, `population_country`
  (country and continent keep the spelling already used in `mapping.csv` for the same
  `iso3`). Existing rows are never re-geocoded; the socio-economic columns (`gmp`, `gini`,
  `traffic_index`, ...) stay blank for external enrichment. A city without a country
  code is *not* geocoded (a name alone is too ambiguous), and ambiguous matches are reported.
* `channel` holds channel IDs, `upload_date` is `YYYYMMDD`, and **`time_of_day` is `0`
  (day), `1` (night) or `-1` (unknown)** - an unknown time of day is no longer silently
  counted as day. Check that Insight's consumers tolerate `-1` before relying on it.
* List shapes match the native file: `videos`, `upload_date`, `vehicle_type`, `fps_list`
  and `channel` are flat lists (`[a,b]`); `time_of_day`, `start_time`, `end_time` are
  nested (`[[0],[1]]`), each aligned element-for-element with `videos`. `--append` merges
  idempotently and preserves untouched rows byte-for-byte; files are written atomically.
* End to end: crawler → `discovery.csv` → bridge → PedX-Insight `run.py --localize` →
  `get_all_video_info.py` / `get_all_video_locations.py` → PedX-Visualizer `make db-pipeline`.

### Geocoding accuracy

Checked against the 2,346 hand-geocoded cities of PedX-Insight's `mapping.csv` (with the
country code supplied): 83% resolve; the median error is 0.6 km, 90th percentile 4.3 km.
Among unambiguous matches 2.2% are more than 50 km off - almost all US towns that share a
name with another US town (the bridge reports "ambiguous" for these). Names are matched on
the GeoNames primary name before alternate names, because alternates include nicknames
("Crescent City" is also an alternate name of New Orleans).

## Tests and CI

```bash
make test-unit        # ~200 offline tests; ML libraries are stubbed, no network needed
```

GitHub Actions runs the same suite on Python 3.10 and 3.12 ([.github/workflows/tests.yml](.github/workflows/tests.yml)).
The live paths (YouTube search, yt-dlp, ffmpeg, model inference) are exercised by `make smoke`,
not by the suite.

## Project structure

```
crawler/
  pedx-crawler.py                  # entry point: planning, search, CSV, run records
  search_profiles.py               # profiles, localized query templates, stream building
  filter_common.py                 # metadata filter, frame sampler, VisualFilter base class
  video_quality_filter_yolo.py     # YOLO backend
  video_quality_filter_internvl3.py# InternVL3 backend (native Transformers)
  video_access.py                  # yt-dlp stream resolution + localhost range proxy for ffmpeg
  gazetteer.py                     # offline GeoNames city / country lookup
  provenance.py                    # decisions.csv + manifest.json
  QUALITY_FILTER_README.md         # filter details
  tests/                           # offline pytest suite
scripts/
  discovery_to_mapping.py          # Crawler -> PedX-Insight bridge
  evaluate_filter.py               # label sampling + precision/recall estimation
data/cities*.txt, data/outputs/    # inputs; committed reference outputs
pyproject.toml                     # dependencies and optional extras
```

Requires Python 3.10+ (the ML extras follow their own libraries' requirements).

## License

The repository previously stated "MIT License - see LICENSE" but contains no LICENSE file.
Add one (and confirm the copyright holder) before publishing.
