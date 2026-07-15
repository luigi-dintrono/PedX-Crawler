#!/usr/bin/env python3
"""
PedX Crawler - YouTube Street-Crossing Video Discovery Tool

Discovers YouTube videos of street crossings by city and outputs CSV data
compatible with the PedX pipeline.
"""

import os
import re
import sys
import csv
import html
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# The quality filters are imported lazily inside create_quality_filter() so the
# heavy ML dependencies (torch, ultralytics, transformers, ...) are only needed
# when quality filtering is actually enabled.

# CSV schema. Columns after country_code are enrichment fields fetched from the
# same (1-unit) videos.list call; downstream readers index by name, so appending
# columns is schema-safe.
CSV_FIELDNAMES = [
    'id', 'name', 'city', 'video', 'video_url', 'time_of_day',
    'start_time', 'end_time', 'region_code', 'channel_name', 'channel_url',
    'published_at', 'country_code',
    'duration_seconds', 'view_count', 'like_count', 'comment_count',
    'thumbnail_url', 'latitude', 'longitude',
]


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


class YouTubeDiscovery:
    """Discovers YouTube street-crossing videos by city."""

    def __init__(self, api_key: str, quota_tracker: QuotaTracker = None, quality_filter = None):
        """Initialize with YouTube API key, optional quota tracker, and quality filter."""
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.quota_tracker = quota_tracker or QuotaTracker()
        self.quality_filter = quality_filter
        # Stats from the most recent search_videos() call (for the run summary).
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

    def search_videos(self, city: str, max_results: int = 50,
                      since_date: str = '2020-01-01T00:00:00Z', use_single_search: bool = True,
                      region_code: Optional[str] = None, seen_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """
        Search for street-crossing videos in a specific city.
        Continues searching until it finds the requested number of quality videos.

        Args:
            city: City name to search for
            max_results: Maximum number of quality videos to return
            since_date: ISO 8601 date string for filtering videos published after this date
            use_single_search: If True, use only one search term to save quota (100 units vs 500)
            region_code: Optional ISO 3166-1 alpha-2 code to localize search results
            seen_ids: Optional shared set of already-seen video IDs. Passing one set
                across cities/runs prevents re-discovering and re-paying quota for
                the same videos. A new set is used per call when None.

        Returns:
            List of video metadata dictionaries that passed quality filtering
        """
        # Optimize search terms based on quota usage
        if use_single_search:
            # Use only the most effective search term to save quota (100 units instead of 500)
            search_terms = [f"{city} street crossing pedestrian"]
        else:
            # Original multiple search terms (uses 500 units per city)
            search_terms = [
                f"{city} street crossing pedestrian",
                f"{city} crosswalk pedestrian",
                f"{city} traffic light pedestrian",
                f"{city} walking street crossing",
                f"{city} pedestrian crossing street"
            ]

        quality_videos = []
        if seen_ids is None:
            seen_ids = set()
        considered = 0
        filtered = 0
        search_term_index = 0
        exhausted_terms = set()  # indices of search terms with no more pages
        next_page_token = None
        max_search_attempts = 10  # Prevent infinite loops
        search_attempts = 0

        print(f"  Searching for {max_results} quality videos in {city}...")

        while len(quality_videos) < max_results and search_attempts < max_search_attempts:
            # Check quota before making request
            if not self.quota_tracker.can_make_search_request():
                print(f"  QUOTA WARNING: Not enough quota to continue searching. {self.quota_tracker.get_status()}")
                break

            # Get current search term
            current_idx = search_term_index % len(search_terms)
            search_term = search_terms[current_idx]
            search_attempts += 1

            try:
                # Search for videos with pagination
                search_params = {
                    'q': search_term,
                    'part': 'id,snippet',
                    'type': 'video',
                    'maxResults': 50,  # Always get max results per request
                    'order': 'relevance',
                    'publishedAfter': since_date
                }
                if region_code:
                    search_params['regionCode'] = region_code
                if next_page_token:
                    search_params['pageToken'] = next_page_token

                search_response = self._execute_with_retry(
                    self.youtube.search().list(**search_params), f"search '{search_term}'")

                # Track quota usage
                self.quota_tracker.add_search_request()

                # Collect this page's new (unseen) candidates up front so their
                # details can be fetched in a single batched videos.list call.
                # Candidates are NOT marked seen here — only once actually examined
                # below — so that (a) videos beyond the per-city cap on this page and
                # (b) videos whose detail fetch fails stay discoverable for a later
                # city/run instead of being silently burned in the shared set.
                page = []
                page_ids = set()
                for item in search_response.get('items', []):
                    try:
                        video_id = item['id']['videoId']
                        snippet = item['snippet']
                    except (KeyError, TypeError):
                        # Guard against malformed items so one bad entry does not
                        # abort the rest of the batch.
                        continue
                    if video_id in seen_ids or video_id in page_ids:
                        continue
                    page_ids.add(video_id)  # intra-page dedup only
                    page.append((video_id, snippet))

                # Batch-fetch details for the whole page in ONE call (1 unit for up
                # to 50 IDs, versus 1 unit *per video* previously).
                details_by_id = self._get_video_details_batch([vid for vid, _ in page])

                for video_id, snippet in page:
                    if len(quality_videos) >= max_results:
                        break

                    details = details_by_id.get(video_id)
                    if not details:
                        # Detail fetch missing/failed for this video: leave it unseen
                        # so it can be retried later rather than permanently lost.
                        continue
                    seen_ids.add(video_id)  # mark seen only once actually examined
                    considered += 1

                    video_data = self._build_video_data(video_id, snippet, details, city)

                    # Apply quality filter if available
                    if self.quality_filter:
                        is_quality, filter_reason = self.quality_filter.filter_video(video_data)
                        if not is_quality:
                            filtered += 1
                            print(f"    Filtered out: '{video_data['name'][:50]}...' - {filter_reason}")
                            continue
                        else:
                            print(f"    [OK] Quality video: '{video_data['name'][:50]}...'")
                    else:
                        print(f"    [OK] Video added: '{video_data['name'][:50]}...'")

                    quality_videos.append(video_data)

                # Advance pagination / search term.
                next_page_token = search_response.get('nextPageToken')
                if not next_page_token:
                    # This search term is fully paginated; move on to the next one.
                    print(f"  Completed search term: '{search_term}' - Found {len(quality_videos)} quality videos so far")
                    exhausted_terms.add(current_idx)
                    search_term_index += 1
                    # Stop once every term has been drained, instead of wrapping
                    # around and re-crawling page 1 of an exhausted term (wasted quota).
                    if len(exhausted_terms) >= len(search_terms):
                        print(f"  All search terms exhausted for {city}")
                        break

            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print(f"  QUOTA EXCEEDED: Cannot search for '{search_term}'. Please wait 24 hours or request quota increase.")
                    raise e  # Re-raise to stop processing
                else:
                    print(f"  Error searching for '{search_term}': {e}")
                    exhausted_terms.add(current_idx)
                    search_term_index += 1
                    next_page_token = None
                    if len(exhausted_terms) >= len(search_terms):
                        break
                    continue
            except Exception as e:
                print(f"  Unexpected error searching for '{search_term}': {e}")
                exhausted_terms.add(current_idx)
                search_term_index += 1
                next_page_token = None
                if len(exhausted_terms) >= len(search_terms):
                    break
                continue

        self.last_stats = {'considered': considered, 'kept': len(quality_videos), 'filtered': filtered}
        print(f"  Search complete: Found {len(quality_videos)} quality videos out of {considered} candidates examined")
        return quality_videos[:max_results]

    def _build_video_data(self, video_id: str, snippet: Dict[str, Any],
                          details: Dict[str, Any], city: str) -> Dict[str, Any]:
        """Assemble a CSV row from a search snippet + a videos.list details item.

        Enrichment fields (view/like/comment counts, geo) come from parts already
        fetched by _get_video_details_batch, so they cost no extra quota.
        """
        content = details.get('contentDetails', {}) or {}
        stats = details.get('statistics', {}) or {}
        recording = details.get('recordingDetails', {}) or {}
        location = recording.get('location', {}) or {}
        duration_iso = content.get('duration', 'PT0S')

        # YouTube returns HTML-escaped text (e.g. "Ben &amp; Jerry&#39;s").
        title = html.unescape(snippet.get('title', ''))
        channel_name = html.unescape(snippet.get('channelTitle', ''))
        thumbnails = snippet.get('thumbnails', {}) or {}
        thumbnail = (thumbnails.get('high') or thumbnails.get('default') or {}).get('url', '')

        return {
            'id': video_id,
            'name': title,
            'city': city,
            'video': video_id,
            'video_url': f"https://www.youtube.com/watch?v={video_id}",
            'time_of_day': self._extract_time_of_day(title),
            'start_time': '0:00',  # Default start time
            'end_time': self._format_duration(duration_iso),
            'region_code': self._extract_region_code(city),
            'channel_name': channel_name,
            'channel_url': f"https://www.youtube.com/channel/{snippet.get('channelId', '')}",
            # Helper field for the quality filter's upload-date check; also written
            # to the CSV (feeds mapping.csv's upload_date downstream).
            'published_at': snippet.get('publishedAt', ''),
            # Enrichment columns (free — same 1-unit videos.list call).
            'duration_seconds': self._iso_duration_seconds(duration_iso),
            'view_count': stats.get('viewCount', ''),
            'like_count': stats.get('likeCount', ''),
            'comment_count': stats.get('commentCount', ''),
            'thumbnail_url': thumbnail,
            'latitude': location.get('latitude', ''),
            'longitude': location.get('longitude', ''),
        }

    def _get_video_details_batch(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch details for many videos, 50 IDs per videos.list call.

        videos.list costs 1 quota unit regardless of how many IDs are requested, so
        batching turns up to 50 single-video lookups (50 units) into one (1 unit).
        """
        details: Dict[str, Dict[str, Any]] = {}
        for start in range(0, len(video_ids), 50):
            chunk = video_ids[start:start + 50]
            if not chunk:
                continue
            try:
                response = self._execute_with_retry(
                    self.youtube.videos().list(
                        part='contentDetails,statistics,recordingDetails',
                        id=','.join(chunk)),
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

    def _extract_time_of_day(self, title: str) -> str:
        """Extract time of day from video title."""
        title_lower = title.lower()

        if any(word in title_lower for word in ['morning', 'dawn', 'sunrise']):
            return 'morning'
        elif any(word in title_lower for word in ['afternoon', 'midday', 'noon']):
            return 'afternoon'
        elif any(word in title_lower for word in ['evening', 'dusk', 'sunset']):
            return 'evening'
        elif any(word in title_lower for word in ['night', 'nighttime', 'dark']):
            return 'night'
        else:
            return 'unknown'

    def _extract_duration(self, video_details: Dict[str, Any]) -> str:
        """Extract duration from a videos.list item (ISO 8601 -> H:MM:SS)."""
        return self._format_duration(video_details.get('contentDetails', {}).get('duration', 'PT0S'))

    def _format_duration(self, duration: str) -> str:
        """Format an ISO 8601 duration (e.g. PT1H2M3S) as H:MM:SS."""
        # Parse the hours/minutes/seconds components independently so none is
        # dropped (a split-based approach zeroed seconds for durations with hours).
        match = re.fullmatch(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration or '')
        if not match:
            return '0:00:00'
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def _iso_duration_seconds(self, duration: str) -> int:
        """Convert an ISO 8601 duration (e.g. PT1H2M3S) to total seconds."""
        match = re.fullmatch(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration or '')
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    def _extract_region_code(self, city: str) -> str:
        """Extract region code from city name."""
        # Simple mapping - in production you'd want a more comprehensive mapping
        city_country_map = {
            'New York': 'US',
            'London': 'GB',
            'Tokyo': 'JP',
            'Paris': 'FR',
            'Berlin': 'DE',
            'Mumbai': 'IN',
            'São Paulo': 'BR',
            'Mexico City': 'MX',
            'Cairo': 'EG',
            'Lagos': 'NG',
            'Sydney': 'AU',
            'Toronto': 'CA',
            'Moscow': 'RU',
            'Istanbul': 'TR',
            'Bangkok': 'TH',
            'Jakarta': 'ID',
            'Manila': 'PH'
        }
        return city_country_map.get(city, 'UNKNOWN')


def convert_date_to_iso(since_date: str) -> str:
    """Convert YYYY-MM-DD date to ISO 8601 format for YouTube API."""
    try:
        # Parse the date and convert to ISO format
        date_obj = datetime.strptime(since_date, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%dT00:00:00Z')
    except ValueError:
        print(f"Error: Invalid date format '{since_date}'. Please use YYYY-MM-DD format.")
        sys.exit(1)


def load_cities(file_path: str) -> tuple:
    """Load cities from text file.

    Returns (cities, country_map): the city names in file order, plus a
    {city: country_code} map for lines using the "City,CC" format (previously
    the CC was parsed and discarded). Cities without a CC are absent from the map.
    """
    cities = []
    country_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                city = line.strip()
                if city and not city.startswith('#'):
                    # Handle "City,CC" format
                    if ',' in city:
                        city, _, cc = city.partition(',')
                        city = city.strip()
                        cc = cc.strip().upper()
                        if cc:
                            country_map[city] = cc
                    cities.append(city)
    except FileNotFoundError:
        print(f"Error: Cities file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading cities file: {e}")
        sys.exit(1)

    return cities, country_map


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

    # Split path into directory, filename, and extension
    directory = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    name, ext = os.path.splitext(filename)

    # Find the next available number
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename)
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
            # would produce a ragged, misaligned CSV. Verify the schema matches;
            # if not (e.g. a file from an older version), write a new file instead.
            with open(output_path, 'r', newline='', encoding='utf-8-sig') as existing:
                existing_header = next(csv.reader(existing), [])
            if existing_header != CSV_FIELDNAMES:
                fallback = get_unique_filename(output_path)
                print(f"Warning: '{output_path}' has an incompatible CSV header; "
                      f"writing to '{fallback}' instead of appending.")
                return _write_rows(fallback, videos, write_header=True, mode='w')
        return _write_rows(output_path, videos, write_header=not has_content, mode='a')

    # Get unique filename to avoid overwriting existing files
    return _write_rows(get_unique_filename(output_path), videos, write_header=True, mode='w')


def _write_rows(path: str, videos: List[Dict[str, Any]], write_header: bool, mode: str) -> str:
    """Write video rows to a CSV, dropping any fields not in the schema."""
    with open(path, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerows(videos)
    return path


def create_quality_filter(filter_type: str, **kwargs):
    """Create a quality filter instance based on the specified type.

    Imports are performed lazily so the crawler can run with only the core
    dependencies installed when quality filtering is not requested.
    """
    if filter_type == "yolo":
        from video_quality_filter_yolo import VideoQualityFilter
        return VideoQualityFilter(**kwargs)
    elif filter_type == "internvl3":
        from video_quality_filter_internvl3 import VideoQualityFilterInternVL3
        return VideoQualityFilterInternVL3(**kwargs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Choose 'yolo' or 'internvl3'.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PedX Crawler - Discover YouTube street-crossing videos by city')
    parser.add_argument('--api-key', help='YouTube API key')
    parser.add_argument('--api-key-file', help='Path to file containing YouTube API key')
    parser.add_argument('--cities-file', default='data/cities.txt', help='Path to cities file')
    parser.add_argument('--output', default='data/outputs/discovery.csv', help='Output CSV file path')
    parser.add_argument('--append', action='store_true', help='Append to --output instead of writing a new numbered file')
    parser.add_argument('--exclude-existing', action='append', metavar='CSV', default=None,
                        help='Skip video IDs already present in this discovery CSV (repeatable) to avoid re-spending quota')
    parser.add_argument('--max-results', type=int, default=None, help='Maximum results per city (deprecated, use --per-city)')
    parser.add_argument('--per-city', type=int, default=None, help='Maximum quality videos per city (continues searching until target reached)')
    parser.add_argument('--since', default='2020-01-01', help='Filter videos published after this date (YYYY-MM-DD format)')
    parser.add_argument('--use-multiple-search', action='store_true', help='Use multiple search terms per city (uses 5x more quota)')
    parser.add_argument('--quota-limit', type=int, default=10000, help='Daily quota limit (default: 10000)')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan + quota estimate and exit without any API calls')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    # Quality filter arguments
    parser.add_argument('--filter', choices=['none', 'yolo', 'internvl3'], default=None,
                        help="Unified filter switch; overrides --enable-quality-filter/--filter-type (e.g. --filter yolo)")
    parser.add_argument('--filter-type', choices=['yolo', 'internvl3'], default='yolo',
                       help='Choose quality filter type: yolo (YOLO26) or internvl3 (InternVL3)')
    parser.add_argument('--enable-quality-filter', action='store_true', help='Enable quality filtering')
    parser.add_argument('--max-upload-months', type=int, default=36, help='Maximum age of videos in months for quality filter (default: 36)')

    # YOLO-specific arguments
    parser.add_argument('--yolo-model', default='yolo26n.pt', help='Path to an Ultralytics YOLO model file (default: yolo26n.pt)')

    # InternVL3-specific arguments
    parser.add_argument('--internvl3-model', default='OpenGVLab/InternVL3-8B', help='InternVL3 model name (default: OpenGVLab/InternVL3-8B)')
    parser.add_argument('--threshold', type=float, default=0.9, help='Score threshold for InternVL3 filter (0.0-1.0, default: 0.9)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device for InternVL3 model (default: auto)')

    # Common arguments
    parser.add_argument('--temp-dir', default='tmp', help='Directory for temporary files (default: tmp)')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Convert date format
    since_date_iso = convert_date_to_iso(args.since)

    # Determine per-city limit: an explicit --per-city always wins, then the
    # deprecated --max-results, otherwise the default of 50. (Using None
    # defaults lets us distinguish an explicit value from the default.)
    if args.per_city is not None:
        per_city = args.per_city
    elif args.max_results is not None:
        per_city = args.max_results
    else:
        per_city = 50

    # Resolve which quality filter to use. The unified --filter flag (when given)
    # overrides the legacy --enable-quality-filter / --filter-type pair.
    if args.filter is not None:
        enable_filter = args.filter != 'none'
        filter_type = args.filter if enable_filter else args.filter_type
    else:
        enable_filter = args.enable_quality_filter
        filter_type = args.filter_type

    # Load cities
    cities, country_map = load_cities(args.cities_file)

    # --dry-run: describe the plan and exit before touching the API (no quota, no key needed).
    if args.dry_run:
        est = len(cities) * (500 if args.use_multiple_search else 100)
        print("=== DRY RUN (no API calls, no quota spent) ===")
        print(f"Cities ({len(cities)}): {', '.join(cities) if cities else '(none)'}")
        print(f"Per-city target: {per_city} quality videos")
        print(f"Published after: {args.since} ({since_date_iso})")
        print(f"Search mode: {'multiple terms (5x quota)' if args.use_multiple_search else 'single term (optimized)'}")
        print(f"Quality filter: {filter_type if enable_filter else 'disabled'}")
        print(f"Output: {args.output}{' (append)' if args.append else ''}")
        print(f"Estimated minimum quota: {est:,} units (search only; excludes pagination retries "
              f"and ~1 unit/page for batched detail lookups)")
        return

    # Get API key
    api_key = None
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

    # Initialize quota tracker
    quota_tracker = QuotaTracker(daily_limit=args.quota_limit)

    # Initialize quality filter if enabled
    quality_filter = None
    if enable_filter:
        try:
            # Prepare filter arguments based on filter type
            filter_kwargs = {
                'max_upload_months': args.max_upload_months,
                'temp_dir': args.temp_dir
            }

            if filter_type == "yolo":
                filter_kwargs['yolo_model_path'] = args.yolo_model
            elif filter_type == "internvl3":
                filter_kwargs['model_name'] = args.internvl3_model
                filter_kwargs['threshold'] = args.threshold
                filter_kwargs['device'] = args.device

            quality_filter = create_quality_filter(filter_type, **filter_kwargs)
            print(f"Quality filter ({filter_type}) initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize quality filter ({filter_type}): {e}")
            print("Continuing without quality filtering...")

    # Seed the cross-run/cross-city dedup set from any prior CSVs.
    exclude_paths = list(args.exclude_existing or [])
    if args.append and os.path.exists(args.output):
        # Appending is idempotent: don't re-add IDs already in the target file.
        exclude_paths.append(args.output)
    global_seen = load_seen_ids(exclude_paths)
    if global_seen:
        print(f"Excluding {len(global_seen):,} already-known video ID(s) from prior CSV(s)")

    if args.verbose:
        print(f"Loaded {len(cities)} cities: {', '.join(cities)}")
        print(f"Searching for videos published after {args.since} ({since_date_iso})")
        print(f"Maximum {per_city} videos per city")
        print(f"Search mode: {'Multiple terms (5x quota)' if args.use_multiple_search else 'Single term (optimized)'}")
        print(f"Quota limit: {args.quota_limit:,} units")
        print(f"Quality filter: {'Enabled (' + filter_type + ')' if quality_filter else 'Disabled'}")
        if quality_filter and filter_type == "internvl3":
            print(f"InternVL3 threshold: {args.threshold}")
        print(f"Estimated minimum quota usage: {len(cities) * (500 if args.use_multiple_search else 100):,} units "
              f"(excludes pagination retries and per-video detail lookups)")

    # Initialize discovery tool
    discovery = YouTubeDiscovery(api_key, quota_tracker, quality_filter)

    # Discover videos for each city
    all_videos = []
    summary = []  # (city, kept, filtered)
    interrupted = False
    try:
        for i, city in enumerate(cities, 1):
            if args.verbose:
                print(f"Searching for videos in {city} ({i}/{len(cities)})...")
                print(f"  {quota_tracker.get_status()}")

            try:
                videos = discovery.search_videos(
                    city,
                    per_city,
                    since_date_iso,
                    use_single_search=not args.use_multiple_search,
                    region_code=country_map.get(city),
                    seen_ids=global_seen,
                )
                # Country code: prefer the explicit CC from cities.txt ("City,CC"),
                # falling back to the per-video region_code heuristic.
                for v in videos:
                    v['country_code'] = country_map.get(city) or v.get('region_code', 'UNKNOWN')
                all_videos.extend(videos)
                summary.append((city, len(videos), discovery.last_stats.get('filtered', 0)))
                if args.verbose:
                    print(f"  Found {len(videos)} videos for {city}")
                    print(f"  {quota_tracker.get_status()}")
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    print(f"QUOTA EXCEEDED: Stopping search. {quota_tracker.get_status()}")
                    print("Please wait 24 hours for quota reset or request quota increase.")
                    break
                else:
                    print(f"Error searching videos for {city}: {e}")
                    continue
            except Exception as e:
                print(f"Error searching videos for {city}: {e}")
                continue
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted (Ctrl-C) - saving the results gathered so far...")
    finally:
        # Always persist whatever was collected, so an interrupt / crash / quota
        # stop never discards quota-expensive results.
        actual_output_path = save_to_csv(all_videos, args.output, append=args.append)

        print()
        print("=== Run summary ===")
        for city, kept, filt in summary:
            extra = f" ({filt} filtered out)" if quality_filter else ""
            print(f"  {city}: {kept} videos{extra}")
        print(f"  Total: {len(all_videos)} videos across {len(summary)}/{len(cities)} cities")
        if interrupted and len(summary) < len(cities):
            print(f"  (interrupted; {len(cities) - len(summary)} cities not processed)")
        print(f"Results saved to: {actual_output_path}")
        print(f"Final quota usage: {quota_tracker.get_status()}")


if __name__ == '__main__':
    main()
