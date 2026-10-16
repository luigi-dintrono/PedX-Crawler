#!/usr/bin/env python3
"""
PedX Crawler - YouTube road-user video discovery for the PedX pipeline.

Discovers YouTube videos per city and writes a discovery CSV for PedX-Insight.
What it searches for is set by a search profile (default: `dashcam`, matching the
curated PedX-Insight dataset); see search_profiles.py. Every examined candidate is
recorded with its search provenance and filter decision under <output dir>/runs/<run id>/.
"""

import os
import re
import sys
import csv
import html
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

# Windows consoles default to a legacy code page (cp1252); printing a video title
# containing an emoji then raises UnicodeEncodeError, which the search loop's
# catch-all turns into a silently discarded search term. Force UTF-8 with
# replacement so no title can crash a crawl.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

from filter_common import (COMMON_NEGATIVE_KEYWORDS, FilterResult, MetadataFilter,
                           time_of_day_from_title)
from gazetteer import Gazetteer
from provenance import DecisionLog, environment_info, new_run_id, utc_now, write_manifest
from search_profiles import (PROFILES, VALID_DEFINITIONS, VALID_DURATIONS, SearchStream,
                             build_streams, resolve_languages)

# The visual filters are imported lazily inside create_quality_filter() so the heavy ML
# dependencies (torch, ultralytics, transformers, ...) are only needed when enabled.

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV schema. Columns are only ever APPENDED (downstream readers index by name and
# --append widens older files in place), never reordered or removed.
CSV_FIELDNAMES = [
    'id', 'name', 'city', 'video', 'video_url', 'time_of_day',
    'start_time', 'end_time', 'region_code', 'channel_name', 'channel_url',
    'published_at', 'country_code',
    'duration_seconds', 'view_count', 'like_count', 'comment_count',
    'thumbnail_url', 'latitude', 'longitude',
    # v0.2: content metadata + provenance (all from calls the crawler already makes)
    'channel_id', 'definition', 'projection', 'default_audio_language', 'live_broadcast',
    'time_of_day_source', 'country_code_source',
    'search_profile', 'search_query', 'search_language', 'search_duration', 'search_rank',
    'filter_backend', 'filter_score', 'run_id', 'frame_source',
]

SEARCH_PAGE_SIZE = 50  # search.list maximum; a page costs 100 units whatever its size
DETAIL_PARTS = 'snippet,contentDetails,statistics,recordingDetails'  # 1 unit per 50 ids


class QuotaTracker:
    """Tracks YouTube API quota usage."""

    def __init__(self, daily_limit: int = 10000):
        self.daily_limit = daily_limit
        self.used_quota = 0
        self.search_cost = 100  # Cost per search request
        self.video_details_cost = 1  # Cost per video details request

    def add_search_request(self, count: int = 1):
        """Add search request cost to quota usage."""
        self.used_quota += self.search_cost * count

    def add_video_details_request(self, count: int = 1):
        """Add video details request cost to quota usage (1 unit per call, any #IDs)."""
        self.used_quota += self.video_details_cost * count

    def get_remaining_quota(self) -> int:
        """Get remaining quota."""
        return max(0, self.daily_limit - self.used_quota)

    def get_quota_percentage(self) -> float:
        """Get quota usage percentage."""
        if self.daily_limit <= 0:
            return 0.0
        return (self.used_quota / self.daily_limit) * 100

    def can_make_search_request(self) -> bool:
        """Check if we can make a search request."""
        return self.get_remaining_quota() >= self.search_cost

    def get_status(self) -> str:
        """Get quota status string."""
        remaining = self.get_remaining_quota()
        percentage = self.get_quota_percentage()
        return f"Quota: {self.used_quota:,}/{self.daily_limit:,} ({percentage:.1f}%) - {remaining:,} remaining"


class VisualFilterBroken(RuntimeError):
    """Raised when the visual filter keeps failing (access denied, model error, ...), so
    the run stops instead of burning quota on candidates it cannot judge."""


@dataclass
class Candidate:
    """One search result, with where it came from."""
    stream: SearchStream
    page: int
    rank: int
    video_id: str
    snippet: Dict[str, Any]


def interleave(pages: List[List[Candidate]]) -> List[Candidate]:
    """Round-robin by rank across pages: s1#1, s2#1, s1#2, s2#2, ... so that no single
    stream (language / duration bucket) gets priority within a sampling round."""
    out = []
    for i in range(max((len(p) for p in pages), default=0)):
        for page in pages:
            if i < len(page):
                out.append(page[i])
    return out


class YouTubeDiscovery:
    """Discovers YouTube videos for a city over one or more search streams."""

    def __init__(self, api_key: Optional[str] = None, quota_tracker: Optional[QuotaTracker] = None,
                 metadata_filter: Optional[MetadataFilter] = None, visual_filter=None,
                 decision_log: Optional[DecisionLog] = None, sampling: str = 'balanced',
                 max_pages: int = 10, definition: str = 'any', run_id: str = '',
                 profile: str = '', youtube=None, max_consecutive_errors: int = 5):
        self.youtube = youtube if youtube is not None else build('youtube', 'v3', developerKey=api_key)
        self.quota_tracker = quota_tracker or QuotaTracker()
        self.metadata_filter = metadata_filter
        self.visual_filter = visual_filter
        self.decision_log = decision_log
        self.sampling = sampling
        self.max_pages = max_pages
        self.definition = definition
        self.run_id = run_id
        self.profile = profile
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_visual_errors = 0
        # Stats from the most recent search_city() call (for the run summary/manifest).
        self.last_stats: Dict[str, int] = {}

    def _execute_with_retry(self, request, what: str, max_retries: int = 3):
        """Execute an API request, retrying transient (5xx / network) errors with
        exponential backoff. quotaExceeded is never retried (it will not recover)."""
        delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                return request.execute()
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    raise
                status = getattr(getattr(e, 'resp', None), 'status', None)
                if status in (500, 502, 503, 504) and attempt < max_retries:
                    print(f"  Transient API error ({status}) on {what}; retry {attempt + 1}/{max_retries} in {delay:.0f}s")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except (TimeoutError, ConnectionError, OSError) as e:
                if attempt < max_retries:
                    print(f"  Network error on {what}: {e}; retry {attempt + 1}/{max_retries} in {delay:.0f}s")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    # --- search ---------------------------------------------------------------

    def search_city(self, city: str, streams: List[SearchStream], max_results: int,
                    published_after: str, published_before: Optional[str] = None,
                    region_code: Optional[str] = None, seen_ids: Optional[set] = None,
                    country_code: str = '') -> List[Dict[str, Any]]:
        """Collect up to `max_results` accepted videos for one city.

        sampling='balanced': each round fetches one page from EVERY active stream and
        examines the candidates interleaved by rank, so the sample's composition does
        not depend on stream order (costs >= one page per stream).
        sampling='greedy': streams are paged round-robin and the city stops as soon as
        the target is met (cheapest; the first stream dominates).

        `seen_ids` is shared across cities/runs; IDs are marked seen only once actually
        examined, so candidates beyond the cap stay discoverable later.
        """
        seen = seen_ids if seen_ids is not None else set()
        kept: List[Dict[str, Any]] = []
        stats = {'pages': 0, 'examined': 0, 'kept': 0, 'rejected_metadata': 0,
                 'rejected_visual': 0, 'errors': 0}
        quota_stop = False
        print(f"  Searching for {max_results} videos in {city} over {len(streams)} search stream(s) "
              f"[{self.sampling}]...")

        while len(kept) < max_results and stats['pages'] < self.max_pages and not quota_stop:
            active = [s for s in streams if not s.exhausted]
            if not active:
                print(f"  All search streams exhausted for {city}")
                break
            round_pages = []
            for stream in active:
                if stats['pages'] >= self.max_pages or len(kept) >= max_results:
                    break
                if not self.quota_tracker.can_make_search_request():
                    print(f"  QUOTA WARNING: Not enough quota to continue searching. {self.quota_tracker.get_status()}")
                    quota_stop = True
                    break
                page = self._fetch_page(stream, published_after, published_before, region_code)
                stats['pages'] += 1
                if self.sampling == 'greedy':
                    self._examine(page, city, kept, max_results, seen, stats, region_code, country_code)
                else:
                    round_pages.append(page)
            if round_pages:
                self._examine(interleave(round_pages), city, kept, max_results, seen, stats,
                              region_code, country_code)

        stats['kept'] = len(kept)
        self.last_stats = stats
        print(f"  Search complete: kept {len(kept)} of {stats['examined']} examined "
              f"({stats['pages']} search page(s))")
        return kept

    def _fetch_page(self, stream: SearchStream, published_after: str,
                    published_before: Optional[str], region_code: Optional[str]) -> List[Candidate]:
        params = {
            'q': stream.query,
            'part': 'id,snippet',
            'type': 'video',
            'maxResults': SEARCH_PAGE_SIZE,
            'order': 'relevance',
            'publishedAfter': published_after,
        }
        if published_before:
            params['publishedBefore'] = published_before
        if stream.duration and stream.duration != 'any':
            params['videoDuration'] = stream.duration
        if self.definition and self.definition != 'any':
            params['videoDefinition'] = self.definition
        if stream.language:
            params['relevanceLanguage'] = stream.language
        if region_code:
            params['regionCode'] = region_code
        if stream.page_token:
            params['pageToken'] = stream.page_token
        try:
            response = self._execute_with_retry(self.youtube.search().list(**params),
                                                f"search {stream.label()}")
        except HttpError as e:
            if "quotaExceeded" in str(e):
                print(f"  QUOTA EXCEEDED on {stream.label()}. Wait for the daily reset or request more quota.")
                raise
            print(f"  Error searching {stream.label()}: {e}")
            stream.exhausted = True
            return []
        except Exception as e:
            print(f"  Unexpected error searching {stream.label()}: {e}")
            stream.exhausted = True
            return []
        finally:
            self.quota_tracker.add_search_request()

        stream.pages += 1
        candidates = []
        items = response.get('items', [])
        for i, item in enumerate(items):
            try:
                video_id = item['id']['videoId']
                snippet = item['snippet']
            except (KeyError, TypeError):
                # Guard against malformed items so one bad entry does not abort the batch.
                continue
            candidates.append(Candidate(stream, stream.pages, stream.items_seen + i + 1, video_id, snippet))
        stream.items_seen += len(items)

        token = response.get('nextPageToken')
        if not token or token in stream.seen_page_tokens:
            stream.exhausted = True
            print(f"  Exhausted {stream.label()} after {stream.pages} page(s)")
        else:
            stream.seen_page_tokens.add(token)
            stream.page_token = token
        return candidates

    def _examine(self, candidates: List[Candidate], city: str, kept: List[Dict[str, Any]],
                 max_results: int, seen: set, stats: Dict[str, int],
                 region_code: Optional[str], country_code: str) -> None:
        fresh, batch_ids = [], set()
        for c in candidates:
            if c.video_id in seen or c.video_id in batch_ids:
                continue
            batch_ids.add(c.video_id)
            fresh.append(c)
        # One videos.list call per 50 IDs (1 unit) instead of one per video.
        details_by_id = self._get_video_details_batch([c.video_id for c in fresh])

        for c in fresh:
            if len(kept) >= max_results:
                break
            details = details_by_id.get(c.video_id)
            if not details:
                # Detail fetch missing/failed: leave it unseen so a later run can retry.
                continue
            seen.add(c.video_id)  # mark seen only once actually examined
            stats['examined'] += 1
            video = self._build_video_data(c, details, city, region_code)
            result = self._judge(video)

            self._track_visual_errors(result)
            video['frame_source'] = result.details.get('frame_source', '')
            frames_tod = result.details.get('time_of_day', 'unknown')
            if frames_tod != 'unknown':
                video['time_of_day'], video['time_of_day_source'] = frames_tod, 'frames'
            self._log_decision(c, video, result, city, country_code)

            short_title = video['name'][:60]
            if self._breaker_tripped(result):
                print(f"    [ERR] {short_title} - {result.stage}: {result.reason}")
                stats['errors'] += 1
                raise VisualFilterBroken(
                    f"{self.consecutive_visual_errors} visual checks in a row could not be completed; "
                    f"last: {result.reason}")
            if result.accepted:
                video['filter_backend'] = self._backend_label()
                video['filter_score'] = '' if result.score is None else result.score
                kept.append(video)
                print(f"    [OK] {short_title}")
            else:
                if result.error:
                    stats['errors'] += 1
                elif result.stage == 'metadata':
                    stats['rejected_metadata'] += 1
                else:
                    stats['rejected_visual'] += 1
                print(f"    [{'ERR' if result.error else 'NO'}] {short_title} - {result.stage}: {result.reason}")

    def _track_visual_errors(self, result: FilterResult) -> None:
        if result.stage == 'visual':
            self.consecutive_visual_errors = self.consecutive_visual_errors + 1 if result.error else 0

    def _breaker_tripped(self, result: FilterResult) -> bool:
        return (result.error and result.stage == 'visual' and self.max_consecutive_errors > 0
                and self.consecutive_visual_errors >= self.max_consecutive_errors)

    def _judge(self, video: Dict[str, Any]) -> FilterResult:
        result = FilterResult(True, 'none', 'no filters enabled')
        if self.metadata_filter:
            result = self.metadata_filter.check(video)
            if not result.accepted:
                return result
        if self.visual_filter:
            result = self.visual_filter.check(video)
        return result

    def _backend_label(self) -> str:
        if self.visual_filter:
            return getattr(self.visual_filter, 'backend', 'visual')
        return 'metadata' if self.metadata_filter else 'none'

    def _log_decision(self, c: Candidate, video: Dict[str, Any], result: FilterResult,
                      city: str, country_code: str) -> None:
        if not self.decision_log:
            return
        details = result.details
        visual = result.stage == 'visual'
        self.decision_log.write({
            'run_id': self.run_id, 'city': city, 'country_code': country_code,
            'profile': self.profile, 'search_query': c.stream.query,
            'search_language': c.stream.language or '', 'search_duration': c.stream.duration,
            'search_page': c.page, 'search_rank': c.rank, 'video_id': c.video_id,
            'title': video['name'], 'channel_id': video['channel_id'],
            'published_at': video['published_at'], 'duration_seconds': video['duration_seconds'],
            'definition': video['definition'], 'projection': video['projection'],
            'live_broadcast': video['live_broadcast'],
            'default_audio_language': video['default_audio_language'],
            'stage': result.stage,
            'decision': 'error' if result.error else ('accept' if result.accepted else 'reject'),
            'reason': result.reason, 'score': '' if result.score is None else result.score,
            'backend': self.visual_filter.backend if (visual and self.visual_filter) else '',
            'model': getattr(self.visual_filter, 'model_name', '') if visual else '',
            'threshold': getattr(self.visual_filter, 'threshold', '') if visual else '',
            'frame_timestamps': details.get('frame_timestamps', ''),
            'frame_scores': details.get('frame_scores', ''),
            'frame_brightness': details.get('frame_brightness', ''),
            'time_of_day': video['time_of_day'],
            'frame_source': details.get('frame_source', ''),
        })

    def _build_video_data(self, c: Candidate, details: Dict[str, Any], city: str,
                          region_code: Optional[str]) -> Dict[str, Any]:
        """Assemble a CSV row from a search result + its videos.list item.

        All fields come from parts already fetched (1 unit per 50 videos), so the
        enrichment costs no extra quota.
        """
        snippet = details.get('snippet') or c.snippet
        content = details.get('contentDetails', {}) or {}
        stats = details.get('statistics', {}) or {}
        recording = details.get('recordingDetails', {}) or {}
        location = recording.get('location', {}) or {}
        duration_iso = content.get('duration', 'PT0S')

        # Search results are HTML-escaped (e.g. "Ben &amp; Jerry&#39;s"); unescape both sources.
        title = html.unescape(snippet.get('title', '') or c.snippet.get('title', ''))
        channel_name = html.unescape(snippet.get('channelTitle', ''))
        channel_id = snippet.get('channelId', '')
        thumbnails = snippet.get('thumbnails', {}) or {}
        thumbnail = (thumbnails.get('high') or thumbnails.get('default') or {}).get('url', '')
        time_of_day = time_of_day_from_title(title)

        return {
            'id': c.video_id,
            'name': title,
            'city': city,
            'video': c.video_id,
            'video_url': f"https://www.youtube.com/watch?v={c.video_id}",
            'time_of_day': time_of_day,
            'start_time': '0:00',
            'end_time': self._format_duration(duration_iso),
            'region_code': region_code or 'UNKNOWN',
            'channel_name': channel_name,
            'channel_url': f"https://www.youtube.com/channel/{channel_id}",
            'published_at': snippet.get('publishedAt', ''),
            'duration_seconds': self._iso_duration_seconds(duration_iso),
            'view_count': stats.get('viewCount', ''),
            'like_count': stats.get('likeCount', ''),
            'comment_count': stats.get('commentCount', ''),
            'thumbnail_url': thumbnail,
            'latitude': location.get('latitude', ''),
            'longitude': location.get('longitude', ''),
            'channel_id': channel_id,
            'definition': content.get('definition', ''),
            'projection': content.get('projection', ''),
            'default_audio_language': snippet.get('defaultAudioLanguage') or snippet.get('defaultLanguage') or '',
            'live_broadcast': snippet.get('liveBroadcastContent') or c.snippet.get('liveBroadcastContent', ''),
            'time_of_day_source': 'title' if time_of_day != 'unknown' else 'none',
            'search_profile': self.profile,
            'search_query': c.stream.query,
            'search_language': c.stream.language or '',
            'search_duration': c.stream.duration,
            'search_rank': c.rank,
            'run_id': self.run_id,
            # Used by the metadata filter only; not CSV columns (dropped on write).
            'description': snippet.get('description', ''),
            'tags': snippet.get('tags') or [],
        }

    def _get_video_details_batch(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch details for many videos, 50 IDs per videos.list call.

        videos.list costs 1 quota unit regardless of how many IDs (or parts) are
        requested, so batching turns up to 50 single-video lookups into one.
        """
        details: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(video_ids), 50):
            chunk = video_ids[start:start + 50]
            if not chunk:
                continue
            try:
                response = self._execute_with_retry(
                    self.youtube.videos().list(part=DETAIL_PARTS, id=','.join(chunk)),
                    'videos.list details')
                self.quota_tracker.add_video_details_request()  # 1 unit per call
                for item in response.get('items', []):
                    details[item['id']] = item
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    raise  # stop the run instead of quietly continuing to spend quota
                print(f"Error getting video details batch: {e}")
            except Exception as e:
                print(f"Error getting video details batch: {e}")
        return details

    @staticmethod
    def _format_duration(duration: str) -> str:
        """Format an ISO 8601 duration (e.g. PT1H2M3S) as H:MM:SS."""
        seconds = YouTubeDiscovery._iso_duration_seconds(duration)
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

    @staticmethod
    def _iso_duration_seconds(duration: str) -> int:
        """Convert an ISO 8601 duration (e.g. PT1H2M3S, P1DT2H) to total seconds."""
        match = re.fullmatch(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?', duration or '')
        if not match:
            return 0
        days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
        return days * 86400 + hours * 3600 + minutes * 60 + seconds


# --- cities ---------------------------------------------------------------------

@dataclass
class CityEntry:
    name: str
    country_code: Optional[str] = None   # from cities.txt ("City,CC")
    local_name: Optional[str] = None     # from cities.txt ("City,CC,Local name")


@dataclass
class CityPlan:
    entry: CityEntry
    country_code: str                     # resolved ISO2 or 'UNKNOWN'
    country_code_source: str              # 'cities_file' | 'gazetteer' | 'none'
    languages: List[Optional[str]]
    streams: List[SearchStream]
    geonameid: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ambiguous: bool = False
    stats: Dict[str, int] = field(default_factory=dict)


def load_cities(file_path: str) -> List[CityEntry]:
    """Load cities from a text file: one `City`, `City,CC` or `City,CC,Local name` per
    line ('#' comments). The local name is used in non-English search queries."""
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                entries.append(CityEntry(name=parts[0],
                                         country_code=(parts[1].upper() if len(parts) > 1 and parts[1] else None),
                                         local_name=(parts[2] if len(parts) > 2 and parts[2] else None)))
    except FileNotFoundError:
        print(f"Error: Cities file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading cities file: {e}")
        sys.exit(1)
    return entries


def plan_city(entry: CityEntry, gazetteer: Gazetteer, profile, language_spec: str,
              durations: List[str], all_terms: bool) -> CityPlan:
    """Resolve a city's country/language and build its search streams."""
    city_info = gazetteer.resolve_city(entry.name, entry.country_code)
    if entry.country_code:
        cc, source = entry.country_code, 'cities_file'
    elif city_info:
        cc, source = city_info.country_code, 'gazetteer'
    else:
        cc, source = 'UNKNOWN', 'none'
    country = gazetteer.country(cc) if cc != 'UNKNOWN' else None
    country_languages = [country.primary_language] if country and country.primary_language else []
    languages = resolve_languages(language_spec, profile, country_languages)
    streams = build_streams(profile, entry.name, languages, durations, all_terms=all_terms,
                            local_name=entry.local_name)
    return CityPlan(entry=entry, country_code=cc, country_code_source=source, languages=languages,
                    streams=streams, geonameid=city_info.geonameid if city_info else None,
                    latitude=city_info.latitude if city_info else None,
                    longitude=city_info.longitude if city_info else None,
                    ambiguous=bool(city_info and city_info.ambiguous and not entry.country_code))


# --- CSV I/O --------------------------------------------------------------------

def convert_date_to_iso(since_date: str) -> str:
    """Convert YYYY-MM-DD date to ISO 8601 format for YouTube API."""
    try:
        date_obj = datetime.strptime(since_date, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%dT00:00:00Z')
    except ValueError:
        print(f"Error: Invalid date format '{since_date}'. Please use YYYY-MM-DD format.")
        sys.exit(1)


def load_seen_ids(paths: List[str]) -> set:
    """Load video IDs from existing discovery CSV(s) so a new crawl can skip them
    (cross-run dedup — avoids re-spending search/detail quota on known videos)."""
    seen = set()
    for path in paths:
        try:
            # utf-8-sig transparently strips a BOM (else the first column would be
            # keyed '﻿id' and every id would be missed).
            with open(path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_id = (row.get('id') or '').strip()
                    if video_id:
                        seen.add(video_id)
        except FileNotFoundError:
            print(f"Warning: --exclude-existing file not found: {path}")
        except Exception as e:
            print(f"Warning: could not read '{path}': {e}")
    return seen


def get_unique_filename(output_path: str) -> str:
    """Get a unique filename by adding a number if the file already exists."""
    if not os.path.exists(output_path):
        return output_path
    directory = os.path.dirname(output_path)
    name, ext = os.path.splitext(os.path.basename(output_path))
    counter = 1
    while True:
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def save_to_csv(videos: List[Dict[str, Any]], output_path: str, append: bool = False) -> str:
    """Save video data to CSV.

    Default: write to a unique, non-conflicting filename.
    append=True: append rows to output_path (writing the header only when the file
    is being created), for incremental / top-up crawls.
    """
    # Ensure output directory exists (skip when --output is a bare filename,
    # where os.path.dirname returns '' and os.makedirs('') would raise).
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if append:
        # Treat a missing OR empty file as new, so a header is still written.
        has_content = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        if has_content:
            # Appending rows with a different column set than the existing header
            # would produce a ragged, misaligned CSV, so the header must match.
            with open(output_path, 'r', newline='', encoding='utf-8-sig') as existing:
                existing_header = next(csv.reader(existing), [])
            if existing_header != CSV_FIELDNAMES:
                # Columns are only ever APPENDED to CSV_FIELDNAMES, so a file written
                # by an older version has a header that is a strict prefix of the
                # current one and can be widened losslessly. Upgrade it in place and
                # append, instead of silently diverting the run to a new file (which
                # left the target file never growing and the diverted IDs outside the
                # next run's dedup set, re-spending quota on the same videos).
                if existing_header and existing_header == CSV_FIELDNAMES[:len(existing_header)]:
                    _upgrade_csv_schema(output_path)
                    print(f"Note: upgraded '{output_path}' from {len(existing_header)} to "
                          f"{len(CSV_FIELDNAMES)} columns before appending.")
                else:
                    fallback = get_unique_filename(output_path)
                    print(f"Warning: '{output_path}' has an incompatible CSV header; "
                          f"writing to '{fallback}' instead of appending.")
                    return _write_rows(fallback, videos, write_header=True, mode='w')
        return _write_rows(output_path, videos, write_header=not has_content, mode='a')

    return _write_rows(get_unique_filename(output_path), videos, write_header=True, mode='w')


def _upgrade_csv_schema(path: str) -> None:
    """Rewrite a prefix-schema discovery CSV in place with the full current header.

    Rows are widened by filling the newly added columns with ''. The rewrite goes to a
    temporary file and is swapped in with os.replace, so an interrupted upgrade cannot
    leave the original truncated.
    """
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    tmp = path + '.upgrading'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (row.get(key) or '') for key in CSV_FIELDNAMES})
    os.replace(tmp, path)


def _write_rows(path: str, videos: List[Dict[str, Any]], write_header: bool, mode: str) -> str:
    """Write video rows to a CSV, dropping any fields not in the schema."""
    with open(path, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerows(videos)
    return path


# --- filters --------------------------------------------------------------------

def create_quality_filter(filter_type: str, **kwargs):
    """Create a visual (stage-2) filter. Imports are lazy so the crawler runs with only
    the core dependencies when visual filtering is not requested."""
    if filter_type == "yolo":
        from video_quality_filter_yolo import VideoQualityFilter
        return VideoQualityFilter(**kwargs)
    elif filter_type == "internvl3":
        from video_quality_filter_internvl3 import VideoQualityFilterInternVL3
        return VideoQualityFilterInternVL3(**kwargs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Choose 'yolo' or 'internvl3'.")


def parse_choice_list(raw: str, valid: Tuple[str, ...], flag: str) -> List[str]:
    values = [v.strip() for v in raw.split(',') if v.strip()]
    bad = [v for v in values if v not in valid]
    if bad or not values:
        print(f"Error: {flag} accepts a comma-separated list of {', '.join(valid)} (got '{raw}')")
        sys.exit(2)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PedX Crawler - discover YouTube road-user videos by city')
    parser.add_argument('--api-key', help='YouTube API key')
    parser.add_argument('--api-key-file', help='Path to file containing YouTube API key')
    parser.add_argument('--cities-file', default='data/cities.txt',
                        help='Cities file: one "City", "City,CC" or "City,CC,Local name" per line')
    parser.add_argument('--output', default='data/outputs/discovery.csv', help='Output CSV file path')
    parser.add_argument('--append', action='store_true', help='Append to --output instead of writing a new numbered file')
    parser.add_argument('--exclude-existing', action='append', metavar='CSV', default=None,
                        help='Skip video IDs already present in this discovery CSV (repeatable) to avoid re-spending quota')
    parser.add_argument('--runs-dir', default=None,
                        help='Where run provenance (manifest.json, decisions.csv) goes (default: <output dir>/runs)')
    parser.add_argument('--max-results', type=int, default=None, help='Maximum results per city (deprecated, use --per-city)')
    parser.add_argument('--per-city', type=int, default=None, help='Target number of accepted videos per city (default 50)')
    parser.add_argument('--since', default='2020-01-01', help='Only videos published after this date (YYYY-MM-DD)')
    parser.add_argument('--until', default=None, help='Only videos published before this date (YYYY-MM-DD); fixes the window for reproducible crawls')
    parser.add_argument('--quota-limit', type=int, default=10000, help='Daily quota limit (default: 10000)')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan + quota estimate and exit without any API calls')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    search = parser.add_argument_group('search')
    search.add_argument('--profile', choices=sorted(PROFILES), default='dashcam',
                        help='What to search for (default: dashcam, matching the curated PedX-Insight dataset)')
    search.add_argument('--languages', default='auto',
                        help="Query languages: 'auto' (country's language + English), 'none' (English, no relevanceLanguage), or codes like 'ja,en'")
    search.add_argument('--video-duration', default=None,
                        help="YouTube videoDuration filter(s), comma-separated from any,short,medium,long (default: profile's)")
    search.add_argument('--definition', choices=VALID_DEFINITIONS, default=None,
                        help="YouTube videoDefinition filter (default: profile's; 'high' = HD only)")
    search.add_argument('--sampling', choices=['balanced', 'greedy'], default='balanced',
                        help='balanced: every search stream contributes each round (>= 1 page per stream); greedy: stop as soon as the target is met (cheapest)')
    search.add_argument('--max-pages-per-city', type=int, default=10, help='Search pages (100 units each) allowed per city (default: 10)')
    search.add_argument('--use-multiple-search', action='store_true', help='Use every query template of the profile, not just the first per language')

    meta = parser.add_argument_group('metadata filter (stage 1, always on)')
    meta.add_argument('--no-metadata-filter', action='store_true', help='Disable stage 1 (raw crawl)')
    meta.add_argument('--min-duration', type=int, default=None, help="Minimum duration in seconds (default: profile's)")
    meta.add_argument('--max-duration', type=int, default=None, help="Maximum duration in seconds, 0 = no limit (default: profile's)")
    meta.add_argument('--max-upload-months', type=int, default=None,
                      help='Reject videos older than N months before --until/run start (default: off; prefer --since)')
    meta.add_argument('--require-title-keywords', action='store_true',
                      help="Also require one of the profile's (multilingual) keywords in the title or tags")

    visual = parser.add_argument_group('visual filter (stage 2)')
    visual.add_argument('--filter', choices=['none', 'yolo', 'internvl3'], default=None,
                        help="Visual filter backend; overrides --enable-quality-filter/--filter-type")
    visual.add_argument('--filter-type', choices=['yolo', 'internvl3'], default='yolo', help='Legacy: backend for --enable-quality-filter')
    visual.add_argument('--enable-quality-filter', action='store_true', help='Legacy: enable the visual filter')
    visual.add_argument('--threshold', type=float, default=None, help='Acceptance threshold on the mean frame score (default 0.5)')
    visual.add_argument('--frames', type=int, default=5, help='Frames sampled evenly across each video (default: 5; stream source only)')
    visual.add_argument('--frame-source', choices=['auto', 'stream', 'thumbnails'], default='auto',
                        help="Where frames come from: 'stream' = full-resolution frames spread across the video (needs cookies or a PO-token plugin), "
                             "'thumbnails' = YouTube's 3 public 320x180 auto-generated frames (no token needed), 'auto' = stream, falling back to thumbnails once YouTube refuses it (default)")
    visual.add_argument('--require-person', action='store_true',
                        help='YOLO: also require a person in a street-scene frame (off by default: it rejected 64%% of curated videos)')
    visual.add_argument('--yolo-model', default='yolo26n.pt', help='Ultralytics YOLO checkpoint (default: yolo26n.pt)')
    visual.add_argument('--yolo-conf', type=float, default=0.25, help='YOLO detection confidence (default: 0.25)')
    visual.add_argument('--internvl3-model', default='OpenGVLab/InternVL3-8B-hf', help='InternVL3 -hf checkpoint (default: OpenGVLab/InternVL3-8B-hf)')
    visual.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device for InternVL3 (default: auto)')
    visual.add_argument('--max-consecutive-errors', type=int, default=5,
                        help='Abort the run after N visual checks in a row could not be completed (0 = never; default 5)')
    visual.add_argument('--ytdlp-cookies', default=None, help='cookies.txt passed to yt-dlp for frame sampling')
    visual.add_argument('--ytdlp-js-runtime', default='auto', help="JS runtime for yt-dlp ('auto' = deno or node if installed, 'none' to disable)")
    visual.add_argument('--temp-dir', default='tmp', help='Directory for temporary clips/frames (default: tmp)')
    return parser


def print_plan(plans: List[CityPlan], args, profile, sampling: str, max_pages: int,
               definition: str, durations: List[str], metadata_desc: str, filter_desc: str,
               per_city: int) -> None:
    print(f"Profile: {profile.name} - {profile.description}")
    print(f"Published: after {args.since}" + (f", before {args.until}" if args.until else " (no end date)"))
    print(f"YouTube filters: videoDuration={','.join(durations)}, videoDefinition={definition}")
    print(f"Sampling: {sampling}, up to {max_pages} search page(s) per city, target {per_city} per city")
    print(f"Metadata filter: {metadata_desc}")
    print(f"Visual filter: {filter_desc}")
    total_min = total_max = 0
    for plan in plans:
        n = len(plan.streams)
        min_pages = 1 if sampling == 'greedy' else min(n, max_pages)
        total_min += min_pages * 100
        total_max += max_pages * 100
        note = ' (AMBIGUOUS name - add the country code)' if plan.ambiguous else ''
        langs = ','.join(l or '-' for l in plan.languages)
        print(f"  {plan.entry.name} [{plan.country_code} via {plan.country_code_source}{note}; "
              f"languages {langs}; {n} stream(s)]")
        for s in plan.streams:
            print(f"      {s.label()}")
    print(f"Estimated search quota: {total_min:,} - {total_max:,} units "
          f"(+ ~1 unit per 50 candidates for batched detail lookups)")


def main():
    """Main entry point."""
    args = build_parser().parse_args()
    load_dotenv()

    profile = PROFILES[args.profile]
    durations = (parse_choice_list(args.video_duration, VALID_DURATIONS, '--video-duration')
                 if args.video_duration else list(profile.video_durations))
    definition = args.definition or profile.definition
    min_duration = profile.min_duration if args.min_duration is None else args.min_duration
    max_duration = profile.max_duration if args.max_duration is None else (args.max_duration or None)
    since_date_iso = convert_date_to_iso(args.since)
    until_date_iso = convert_date_to_iso(args.until) if args.until else None

    # An explicit --per-city always wins, then the deprecated --max-results, else 50.
    if args.per_city is not None:
        per_city = args.per_city
    elif args.max_results is not None:
        per_city = args.max_results
    else:
        per_city = 50

    # The unified --filter flag (when given) overrides the legacy pair.
    if args.filter is not None:
        enable_filter = args.filter != 'none'
        filter_type = args.filter if enable_filter else args.filter_type
    else:
        enable_filter = args.enable_quality_filter
        filter_type = args.filter_type
    threshold = 0.5 if args.threshold is None else args.threshold

    gazetteer = Gazetteer()
    if not gazetteer.available:
        print("Warning: geonamescache is not installed; cities without a country code stay "
              "UNKNOWN and queries are English-only. Install it with: pip install geonamescache")
    plans = [plan_city(entry, gazetteer, profile, args.languages, durations, args.use_multiple_search)
             for entry in load_cities(args.cities_file)]

    negative = COMMON_NEGATIVE_KEYWORDS + profile.negative_keywords
    metadata_desc = ('disabled' if args.no_metadata_filter else
                     f"{min_duration}s - {max_duration or 'no limit'}{'s' if max_duration else ''}, "
                     f"{len(negative)} blocked keywords, title-keyword gate "
                     f"{'on' if args.require_title_keywords else 'off'}"
                     + (f", max age {args.max_upload_months} months" if args.max_upload_months else ''))
    filter_desc = (f"{filter_type} (threshold {threshold}, frame source {args.frame_source}, up to {args.frames} stream frames)"
                   if enable_filter else 'disabled')

    if args.dry_run:
        print("=== DRY RUN (no API calls, no quota spent) ===")
        print_plan(plans, args, profile, args.sampling, args.max_pages_per_city, definition,
                   durations, metadata_desc, filter_desc, per_city)
        print(f"Output: {args.output}{' (append)' if args.append else ''}")
        return

    # API key
    if args.api_key:
        api_key = args.api_key
    elif args.api_key_file:
        try:
            with open(args.api_key_file, 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            print(f"Error: API key file not found at {args.api_key_file}")
            sys.exit(1)
    else:
        api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("Error: YouTube API key not provided. Use --api-key, --api-key-file, or set YOUTUBE_API_KEY environment variable.")
        sys.exit(1)

    run_id = new_run_id()
    started = datetime.now(timezone.utc)
    reference_date = (datetime.strptime(args.until, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                      if args.until else started)

    metadata_filter = None if args.no_metadata_filter else MetadataFilter(
        min_duration=min_duration, max_duration=max_duration, negative_keywords=negative,
        required_keywords=profile.title_keywords if args.require_title_keywords else None,
        max_upload_months=args.max_upload_months, reference_date=reference_date)

    visual_filter = None
    if enable_filter:
        filter_kwargs = {'threshold': threshold, 'temp_dir': args.temp_dir, 'n_frames': args.frames,
                         'cookies_file': args.ytdlp_cookies, 'frame_source': args.frame_source,
                         'js_runtime': None if args.ytdlp_js_runtime == 'none' else args.ytdlp_js_runtime}
        if filter_type == 'yolo':
            filter_kwargs.update(yolo_model_path=args.yolo_model, conf=args.yolo_conf,
                                 require_person=args.require_person)
        else:
            filter_kwargs.update(model_name=args.internvl3_model, device=args.device)
        try:
            visual_filter = create_quality_filter(filter_type, **filter_kwargs)
            print(f"Visual filter ({filter_type}) initialized")
        except Exception as e:
            # Fail closed: a run that asked for filtering must not quietly produce
            # an unfiltered CSV.
            print(f"Error: could not initialize quality filter ({filter_type}): {e}")
            print("Aborting. Fix the filter setup, or run with --filter none.")
            sys.exit(1)

    runs_dir = args.runs_dir or os.path.join(os.path.dirname(args.output) or '.', 'runs')
    run_dir = os.path.join(runs_dir, run_id)
    decisions_path = os.path.join(run_dir, 'decisions.csv')
    manifest_path = os.path.join(run_dir, 'manifest.json')
    decision_log = DecisionLog(decisions_path)

    packages = ['google-api-python-client', 'geonamescache', 'yt-dlp', 'ultralytics',
                'transformers', 'torch', 'Pillow']
    manifest: Dict[str, Any] = {
        'run_id': run_id, 'status': 'running', 'started_utc': utc_now(), 'finished_utc': None,
        'argv': sys.argv[1:],
        'parameters': {
            'profile': profile.name, 'languages': args.languages, 'video_durations': durations,
            'definition': definition, 'sampling': args.sampling,
            'max_pages_per_city': args.max_pages_per_city, 'all_terms': args.use_multiple_search,
            'published_after': since_date_iso, 'published_before': until_date_iso,
            'per_city': per_city, 'metadata_filter': metadata_desc,
            'negative_keywords': list(negative) if metadata_filter else [],
            'visual_filter': filter_type if visual_filter else None,
            'visual_model': getattr(visual_filter, 'model_name', None),
            'threshold': threshold if visual_filter else None, 'frames': args.frames if visual_filter else None,
            'frame_source': args.frame_source if visual_filter else None,
            'quota_limit': args.quota_limit,
        },
        'environment': environment_info(REPO_DIR, packages),
        'output_csv': None, 'decisions_csv': decisions_path, 'cities': [], 'quota_used': 0,
    }
    write_manifest(manifest_path, manifest)
    print(f"Run {run_id}: provenance in {run_dir}")

    exclude_paths = list(args.exclude_existing or [])
    if args.append and os.path.exists(args.output):
        exclude_paths.append(args.output)  # appending is idempotent
    global_seen = load_seen_ids(exclude_paths)
    if global_seen:
        print(f"Excluding {len(global_seen):,} already-known video ID(s) from prior CSV(s)")

    if args.verbose:
        print_plan(plans, args, profile, args.sampling, args.max_pages_per_city, definition,
                   durations, metadata_desc, filter_desc, per_city)

    quota_tracker = QuotaTracker(daily_limit=args.quota_limit)
    discovery = YouTubeDiscovery(api_key, quota_tracker, metadata_filter, visual_filter,
                                 decision_log, sampling=args.sampling,
                                 max_pages=args.max_pages_per_city, definition=definition,
                                 run_id=run_id, profile=profile.name,
                                 max_consecutive_errors=args.max_consecutive_errors)

    all_videos: List[Dict[str, Any]] = []
    processed: List[CityPlan] = []
    status = 'completed'
    try:
        for i, plan in enumerate(plans, 1):
            city = plan.entry.name
            if plan.ambiguous:
                print(f"Warning: '{city}' is ambiguous without a country code; resolved to "
                      f"{plan.country_code}. Add 'City,CC' to the cities file to be sure.")
            print(f"[{i}/{len(plans)}] {city} ({plan.country_code}) - {quota_tracker.get_status()}")
            region = plan.country_code if plan.country_code != 'UNKNOWN' else None
            try:
                videos = discovery.search_city(city, plan.streams, per_city, since_date_iso,
                                               until_date_iso, region_code=region,
                                               seen_ids=global_seen, country_code=plan.country_code)
                for v in videos:
                    v['country_code'] = plan.country_code
                    v['country_code_source'] = plan.country_code_source
                all_videos.extend(videos)
                plan.stats = dict(discovery.last_stats)
                processed.append(plan)
            except VisualFilterBroken as e:
                print(f"ABORTING: {e}")
                print("The visual filter is not working in this environment; fix it (see README, "
                      "'Visual filter access') or run with --filter none. Results so far are saved.")
                status = 'visual_filter_failed'
                break
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print(f"QUOTA EXCEEDED: Stopping search. {quota_tracker.get_status()}")
                    print("Please wait 24 hours for quota reset or request quota increase.")
                    status = 'quota_exceeded'
                    break
                print(f"Error searching videos for {city}: {e}")
            except Exception as e:
                print(f"Error searching videos for {city}: {e}")
    except KeyboardInterrupt:
        status = 'interrupted'
        print("\nInterrupted (Ctrl-C) - saving the results gathered so far...")
    finally:
        # Always persist whatever was collected, so an interrupt / crash / quota
        # stop never discards quota-expensive results.
        actual_output_path = save_to_csv(all_videos, args.output, append=args.append)
        decision_log.close()
        if status == 'completed' and sys.exc_info()[0] is not None:
            status = 'error'
        manifest.update({
            'status': status, 'finished_utc': utc_now(), 'output_csv': actual_output_path,
            'quota_used': quota_tracker.used_quota,
            'cities': [{'city': p.entry.name, 'country_code': p.country_code,
                        'country_code_source': p.country_code_source, 'geonameid': p.geonameid,
                        'latitude': p.latitude, 'longitude': p.longitude, 'ambiguous': p.ambiguous,
                        'languages': p.languages, 'streams': [s.label() for s in p.streams],
                        'processed': p in processed, **p.stats} for p in plans],
        })
        write_manifest(manifest_path, manifest)

        print()
        print("=== Run summary ===")
        for plan in processed:
            st = plan.stats
            print(f"  {plan.entry.name}: kept {st.get('kept', 0)} / examined {st.get('examined', 0)} "
                  f"(metadata rejects {st.get('rejected_metadata', 0)}, visual rejects "
                  f"{st.get('rejected_visual', 0)}, errors {st.get('errors', 0)}; {st.get('pages', 0)} page(s))")
        print(f"  Total: {len(all_videos)} videos across {len(processed)}/{len(plans)} cities ({status})")
        print(f"Results saved to: {actual_output_path}")
        print(f"Decisions log: {decisions_path}")
        print(f"Final quota usage: {quota_tracker.get_status()}")


if __name__ == '__main__':
    main()
