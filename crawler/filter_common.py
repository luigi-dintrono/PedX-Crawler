"""Shared quality-filter machinery (no ML dependencies).

* MetadataFilter - the cheap stage-1 checks on YouTube metadata. Always on in the
  crawler (unless --no-metadata-filter), independent of the visual backends.
* FrameSampler   - downloads 1-second sections spread across the whole video with
  yt-dlp and grabs one frame from each with ffmpeg.
* VisualFilter   - base class for the stage-2 backends (YOLO, InternVL3): sample
  frames, score each one, aggregate, and report a FilterResult that distinguishes
  "rejected" from "could not be checked" (error=True).

Everything fails closed: a video that cannot be checked is never accepted.
"""

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Rejected for every profile. Matched case-insensitively on whole words (substring
# for CJK/Thai, which do not separate words with spaces).
COMMON_NEGATIVE_KEYWORDS = (
    # not continuous real-world footage
    'compilation', 'fails', 'meme', 'memes', 'try not to laugh', 'prank', 'reaction',
    # video games / simulators that look like driving footage
    'gameplay', 'gta', 'gta 5', 'gta v', 'euro truck simulator', 'ets2', 'city car driving',
    'beamng', 'forza horizon', 'minecraft', 'roblox', 'simulator',
    # altered playback speed corrupts PedX-Insight's walking/crossing-speed estimates
    'timelapse', 'time lapse', 'time-lapse', 'hyperlapse', 'sped up', 'fast motion',
    # YouTube Shorts tag
    '#shorts',
)

# Altered playback speed written as a pattern, e.g. "3x Speed", "2X faster", "x4 speed",
# "fast forward". Keyword matching cannot express these (the digit is part of the word).
NEGATIVE_PATTERNS = (
    r'(?<![\w.])\d+(?:\.\d+)?\s*[x×]\s*(?:speed|faster|fast)\b',
    r'(?<![\w.])[x×]\s*\d+(?:\.\d+)?\s*(?:speed|faster|fast)\b',
    r'\bfast[- ]?forward',
    r'\b(?:speed(?:ed)?[- ]?up|sped[- ]?up)\b',
)
_NEGATIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]

# Title words for the time-of-day fallback (used only when frames were not analysed).
TITLE_TIME_OF_DAY = (
    ('night', ('night', 'nighttime', 'night drive', 'noche', 'nocturno', 'nuit', 'nacht',
               'notte', 'ночь', 'ночной', 'ночная', 'gece', 'malam', '夜', '밤')),
    ('evening', ('evening', 'dusk', 'sunset', 'atardecer', 'abend', 'вечер')),
    ('morning', ('morning', 'dawn', 'sunrise', 'mañana', 'morgen', 'утро', 'утренний')),
    ('afternoon', ('afternoon', 'midday', 'noon')),
)

# Frame-brightness day/night thresholds (median over frames of mean luma, 0-255).
# Calibrated on 111 YouTube city-driving videos labelled from unambiguous title cues
# ("night", "morning", ...; weak labels) using their thumbnail frames: night-titled videos
# had median brightness 49 (quartiles 36-64), day-titled 120 (110-140). Lit cities at night
# are not very dark, so the night cut-off is higher than intuition suggests. With these
# values 79% of night-titled videos were labelled night (4% mislabelled day) and 91% of
# day-titled day (3% mislabelled night); the rest fall in the 'unknown' band. Re-fit the
# two numbers from decisions.csv (`frame_brightness`) for other kinds of footage.
NIGHT_MAX_LUMA = 65.0
DAY_MIN_LUMA = 95.0


def _uses_word_boundaries(keyword: str) -> bool:
    # Scripts from Thai (U+0E00) upwards in this list (Thai, CJK, Kana, Hangul) do not
    # put spaces between words, so a word-boundary match would never fire there.
    return all(ord(ch) < 0x0E00 for ch in keyword)


def compile_keywords(keywords: Iterable[str]) -> List[Tuple[str, 're.Pattern']]:
    compiled = []
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        body = re.escape(kw)
        pattern = rf'(?<!\w){body}(?!\w)' if _uses_word_boundaries(kw) else body
        compiled.append((kw, re.compile(pattern, re.IGNORECASE)))
    return compiled


def first_match(compiled: Sequence[Tuple[str, 're.Pattern']], text: str) -> Optional[str]:
    for kw, pattern in compiled:
        if pattern.search(text):
            return kw
    return None


_TITLE_TOD = [(label, compile_keywords(words)) for label, words in TITLE_TIME_OF_DAY]


def time_of_day_from_title(title: str) -> str:
    """'Tokyo Night Drive 4K' -> 'night'; no time word -> 'unknown'."""
    for label, compiled in _TITLE_TOD:
        if first_match(compiled, title or ''):
            return label
    return 'unknown'


def time_of_day_from_brightness(lumas: Sequence[float]) -> str:
    """Median frame luma -> 'night' / 'day' / 'unknown' (see thresholds above)."""
    if not lumas:
        return 'unknown'
    m = median(lumas)
    if m <= NIGHT_MAX_LUMA:
        return 'night'
    if m >= DAY_MIN_LUMA:
        return 'day'
    return 'unknown'


BAR_MAX_LUMA = 6  # an edge row/column whose brightest pixel is <= this is a black bar


def strip_black_bars(gray):
    """Crop letterbox / pillarbox bars from the edges of a grayscale PIL image.

    A bar is an edge row/column that is near-pure black throughout (brightest pixel <=
    BAR_MAX_LUMA), and bars are only removed when the opposite side has one of similar
    thickness - real letterboxing is centred, whereas a dark night scene is rarely black
    to the same depth at both edges (sensor noise alone lifts a full row above the
    threshold). At most a quarter of each dimension per side is removed, so a dark frame
    is never cropped down to its few bright pixels."""
    w, h = gray.size

    def bar_row(y):
        return gray.crop((0, y, w, y + 1)).getextrema()[1] <= BAR_MAX_LUMA

    def bar_col(x):
        return gray.crop((x, 0, x + 1, h)).getextrema()[1] <= BAR_MAX_LUMA

    top = 0
    while top < h // 4 and bar_row(top):
        top += 1
    bottom = h
    while bottom > h - h // 4 and bar_row(bottom - 1):
        bottom -= 1
    left = 0
    while left < w // 4 and bar_col(left):
        left += 1
    right = w
    while right > w - w // 4 and bar_col(right - 1):
        right -= 1
    if abs(top - (h - bottom)) > max(2, h // 50):    # not symmetric: a dark scene, not bars
        top, bottom = 0, h
    if abs(left - (w - right)) > max(2, w // 50):
        left, right = 0, w
    if right - left < 1 or bottom - top < 1:
        return gray
    return gray.crop((left, top, right, bottom))


def frame_luma(path: str) -> float:
    """Mean luma (ITU-R 601, 0-255) of an image file, excluding letterbox bars."""
    from PIL import Image, ImageStat
    with Image.open(path) as img:
        return float(ImageStat.Stat(strip_black_bars(img.convert('L'))).mean[0])


def parse_hms(value: str) -> Optional[int]:
    """'H:MM:SS' / 'MM:SS' / 'SS' -> seconds, None when unparseable."""
    try:
        parts = [int(p) for p in str(value).strip().split(':')]
    except ValueError:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def video_duration_seconds(video: Dict[str, Any]) -> Optional[int]:
    """Duration from the crawler's integer 'duration_seconds', else its 'end_time'."""
    raw = video.get('duration_seconds')
    if raw not in (None, ''):
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    if video.get('end_time'):
        return parse_hms(video['end_time'])
    return None


@dataclass
class FilterResult:
    accepted: bool
    stage: str                    # 'metadata' or 'visual'
    reason: str
    score: Optional[float] = None
    error: bool = False           # the check could not be performed (download/model failure)
    details: Dict[str, Any] = field(default_factory=dict)


class MetadataFilter:
    """Stage 1: reject from YouTube metadata alone (no download, no ML)."""

    def __init__(self, min_duration: int = 240, max_duration: Optional[int] = 3 * 3600,
                 negative_keywords: Iterable[str] = COMMON_NEGATIVE_KEYWORDS,
                 required_keywords: Optional[Iterable[str]] = None,
                 max_upload_months: Optional[int] = None,
                 reference_date: Optional[datetime] = None):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.negative = compile_keywords(negative_keywords)
        self.required = compile_keywords(required_keywords) if required_keywords else None
        self.shorts_tag = compile_keywords(['#shorts'])
        self.max_upload_months = max_upload_months
        # A fixed reference date (run start, or --until) keeps the age check
        # reproducible instead of drifting with datetime.now() on every call.
        self.reference_date = reference_date or datetime.now(timezone.utc)

    def check(self, video: Dict[str, Any]) -> FilterResult:
        def reject(reason):
            return FilterResult(False, 'metadata', reason)

        live = (video.get('live_broadcast') or 'none').lower()
        if live in ('live', 'upcoming'):
            return reject(f"live broadcast ({live})")

        duration = video_duration_seconds(video)
        if not duration:
            return reject("duration unknown or zero (live stream or still processing)")
        if duration < self.min_duration:
            return reject(f"too short: {duration}s < {self.min_duration}s")
        if self.max_duration and duration > self.max_duration:
            return reject(f"too long: {duration}s > {self.max_duration}s")

        if (video.get('projection') or '').lower() == '360':
            return reject("360-degree video")

        title = video.get('name') or ''
        tags = ' '.join(video.get('tags') or [])
        searchable = f"{title} {tags} {video.get('description') or ''}"
        hit = first_match(self.negative, title + ' ' + tags)
        if hit:
            return reject(f"negative keyword: '{hit}'")
        for pattern in _NEGATIVE_PATTERNS:
            m = pattern.search(title + ' ' + tags)
            if m:
                return reject(f"altered playback speed: '{m.group(0)}'")
        if first_match(self.shorts_tag, searchable):
            return reject("YouTube Short (#shorts)")

        if self.max_upload_months:
            published = parse_published_at(video.get('published_at'))
            if published:
                months = (self.reference_date - published).days / 30.4375
                if months > self.max_upload_months:
                    return reject(f"too old: {months:.1f} months > {self.max_upload_months}")

        if self.required is not None and not first_match(self.required, title + ' ' + tags):
            return reject("no profile keyword in title/tags")

        return FilterResult(True, 'metadata', "passed metadata checks")


def parse_published_at(value: Optional[str]) -> Optional[datetime]:
    """YouTube publishedAt ('2024-05-01T12:00:00Z') -> aware UTC datetime."""
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


class SamplingError(RuntimeError):
    """Frames could not be obtained (download / ffmpeg failure)."""


class AccessDeniedError(SamplingError):
    """YouTube refused the media request. Without a PO token or login cookies, current
    player clients only serve the first ~6 MB of a stream (or nothing); see README,
    "Visual filter access"."""

    HINT = ("YouTube refused the media request (HTTP 403). Supply login cookies with "
            "--ytdlp-cookies, or install a yt-dlp PO-token provider plugin "
            "(e.g. bgutil-ytdlp-pot-provider); see README, 'Visual filter access'.")

    def __init__(self, status=403):
        super().__init__(f"{self.HINT} [HTTP {status}]")


@dataclass
class Frame:
    path: str
    timestamp: Optional[float]  # seconds into the video; None when the source does not say


def detect_js_runtime() -> Optional[str]:
    """yt-dlp needs a JavaScript runtime to solve YouTube's challenges; prefer deno
    (yt-dlp's default), then node."""
    for runtime in ('deno', 'node'):
        if shutil.which(runtime):
            return runtime
    return None


class FrameSampler:
    """Sample frames evenly across a video: one frame at the midpoint of each of n
    equal bins (e.g. 10%, 30%, 50%, 70%, 90% for n=5). The opening seconds are usually
    intros/title cards and are a poor proxy for the whole video.

    yt-dlp (Python API) resolves the stream URL; ffmpeg seeks to each timestamp through
    a localhost range-chunking proxy (see video_access.py for why ffmpeg cannot read
    YouTube's URLs directly). Only the index and the fragments around each timestamp
    are transferred."""

    source = 'stream'

    def __init__(self, temp_dir: str = 'tmp', n_frames: int = 5, max_height: int = 720,
                 cookies_file: Optional[str] = None, js_runtime: Optional[str] = 'auto',
                 timeout: int = 90):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.n_frames = max(1, n_frames)
        self.max_height = max_height
        self.cookies_file = cookies_file
        self.js_runtime = detect_js_runtime() if js_runtime == 'auto' else js_runtime
        self.timeout = timeout  # per ffmpeg call

    def timestamps(self, duration: Optional[float]) -> List[float]:
        if not duration or duration <= 2:
            return [0.0]
        last = max(0.0, duration - 2.0)
        return [round(min(last, (i + 0.5) * duration / self.n_frames), 1)
                for i in range(self.n_frames)]

    def _ffmpeg_command(self, source_url: str, t: float, frame_path: Path) -> List[str]:
        return ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-ss', str(t), '-i', source_url,
                '-frames:v', '1', '-q:v', '2', str(frame_path)]

    def sample(self, video_id: str, url: str, duration: Optional[float]) -> List[Frame]:
        from video_access import ChunkedRangeProxy, StreamRefused, resolve_stream
        try:
            stream = resolve_stream(url, self.max_height, self.cookies_file, self.js_runtime)
        except Exception as e:
            raise SamplingError(f"yt-dlp could not resolve a stream: {str(e).strip().splitlines()[-1]}")
        frames: List[Frame] = []
        last_error = ''
        try:
            with ChunkedRangeProxy(stream) as proxy:
                for t in self.timestamps(duration):
                    frame_path = self.temp_dir / f'{video_id}.{t:.1f}.jpg'
                    try:
                        r = subprocess.run(self._ffmpeg_command(proxy.url, t, frame_path),
                                           capture_output=True, text=True, timeout=self.timeout)
                    except subprocess.TimeoutExpired:
                        last_error = f"ffmpeg timed out at {t}s"
                        continue
                    if r.returncode == 0 and frame_path.exists():
                        frames.append(Frame(str(frame_path), t))
                    else:
                        lines = (r.stderr or '').strip().splitlines()
                        last_error = lines[-1] if lines else f"ffmpeg exit {r.returncode}"
                refused = proxy.upstream_status
        except StreamRefused as e:
            raise AccessDeniedError(e.status)
        if not frames:
            if refused in (401, 403):
                raise AccessDeniedError(refused)
            raise SamplingError(f"no frame could be extracted ({last_error})")
        if refused in (401, 403) and len(frames) < max(1, self.n_frames // 2):
            # Only the head of the stream was reachable: the frames would not be spread
            # across the video, so do not pretend they are.
            raise AccessDeniedError(refused)
        return frames

    def cleanup(self, video_id: str) -> None:
        for path in self.temp_dir.glob(f'{video_id}.*'):
            try:
                path.unlink()
            except OSError as e:
                print(f"    Warning: could not delete temp file {path}: {e}")


THUMBNAIL_NAMES = ('mq1', 'mq2', 'mq3')  # 320x180 auto-generated frames
THUMBNAIL_URL = 'https://i.ytimg.com/vi/{video_id}/{name}.jpg'


class ThumbnailSampler:
    """Frames from YouTube's auto-generated thumbnails (`mq1`, `mq2`, `mq3`).

    These are public 320x180 stills taken from the video itself (commonly documented as
    roughly the 25%, 50% and 75% marks; YouTube does not guarantee the positions), served
    from i.ytimg.com with no token, login, yt-dlp or ffmpeg. Trade-offs versus
    FrameSampler: only 3 frames, low resolution (small pedestrians can be missed),
    positions unknown (timestamps are None), but they work when the stream is refused.
    `decisions.csv` records `frame_source` for every verdict."""

    source = 'thumbnails'

    def __init__(self, temp_dir: str = 'tmp', names: Sequence[str] = THUMBNAIL_NAMES,
                 timeout: int = 20, min_frames: int = 2, session=None):
        import requests
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.names = tuple(names)
        self.timeout = timeout
        self.min_frames = min_frames
        self.session = session or requests.Session()

    def sample(self, video_id: str, url: str, duration: Optional[float]) -> List[Frame]:
        import requests
        frames: List[Frame] = []
        last_error = ''
        for name in self.names:
            try:
                r = self.session.get(THUMBNAIL_URL.format(video_id=video_id, name=name),
                                     timeout=self.timeout)
            except requests.RequestException as e:
                last_error = str(e)
                continue
            if r.status_code != 200 or not r.content:
                last_error = f"HTTP {r.status_code} for {name}"
                continue
            path = self.temp_dir / f'{video_id}.{name}.jpg'
            path.write_bytes(r.content)
            frames.append(Frame(str(path), None))
        if len(frames) < self.min_frames:
            raise SamplingError(f"only {len(frames)} of {len(self.names)} thumbnail frames "
                                f"available ({last_error or 'none returned'})")
        return frames

    def cleanup(self, video_id: str) -> None:
        for path in self.temp_dir.glob(f'{video_id}.*'):
            try:
                path.unlink()
            except OSError as e:
                print(f"    Warning: could not delete temp file {path}: {e}")


class AutoSampler:
    """Try the full-stream sampler; once YouTube refuses the media request, use thumbnails
    for the rest of the run (re-trying a blocked stream per video would cost ~15 s each).
    `last_source` says which one produced the most recent frames."""

    def __init__(self, stream: FrameSampler, thumbnails: ThumbnailSampler):
        self.stream = stream
        self.thumbnails = thumbnails
        self.stream_blocked = False
        self.last_source = ''

    def sample(self, video_id: str, url: str, duration: Optional[float]) -> List[Frame]:
        if not self.stream_blocked:
            try:
                frames = self.stream.sample(video_id, url, duration)
                self.last_source = self.stream.source
                return frames
            except AccessDeniedError:
                self.stream_blocked = True
                self.stream.cleanup(video_id)
                print("    Note: YouTube refused the media stream; using thumbnail frames "
                      "(3 low-res frames) for the rest of this run. For full-resolution frames "
                      "spread across the video, supply --ytdlp-cookies or a PO-token plugin.")
        frames = self.thumbnails.sample(video_id, url, duration)
        self.last_source = self.thumbnails.source
        return frames

    def cleanup(self, video_id: str) -> None:
        self.stream.cleanup(video_id)
        self.thumbnails.cleanup(video_id)


FRAME_SOURCES = ('auto', 'stream', 'thumbnails')


def make_sampler(frame_source: str, temp_dir: str = 'tmp', n_frames: int = 5, **stream_kwargs):
    """Build the frame sampler for --frame-source (auto | stream | thumbnails)."""
    if frame_source not in FRAME_SOURCES:
        raise ValueError(f"frame_source must be one of {FRAME_SOURCES}, got {frame_source!r}")
    if frame_source == 'thumbnails':
        return ThumbnailSampler(temp_dir=temp_dir)
    stream = FrameSampler(temp_dir=temp_dir, n_frames=n_frames, **stream_kwargs)
    if frame_source == 'stream':
        return stream
    return AutoSampler(stream, ThumbnailSampler(temp_dir=temp_dir))


class VisualFilter:
    """Base class for stage-2 backends. Subclasses implement score_frames()."""

    backend = 'base'
    model_name = ''

    def __init__(self, threshold: float = 0.5, sampler=None, temp_dir: str = 'tmp',
                 n_frames: int = 5, frame_source: str = 'auto', **sampler_kwargs):
        self.threshold = threshold
        self.sampler = sampler or make_sampler(frame_source, temp_dir, n_frames, **sampler_kwargs)

    def score_frames(self, paths: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Return a score in [0, 1] per frame plus per-frame audit details."""
        raise NotImplementedError

    def decide(self, scores: List[float], frame_details: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        score = sum(scores) / len(scores) if scores else 0.0
        verdict = score >= self.threshold
        return verdict, score, f"mean frame score {score:.3f} {'>=' if verdict else '<'} {self.threshold}"

    def check(self, video: Dict[str, Any]) -> FilterResult:
        video_id, url = video.get('id', ''), video.get('video_url', '')
        if not video_id or not url:
            return FilterResult(False, 'visual', "missing video id or URL", error=True)
        try:
            frames = self.sampler.sample(video_id, url, video_duration_seconds(video))
            paths = [f.path for f in frames]
            lumas = [round(frame_luma(p), 1) for p in paths]
            scores, frame_details = self.score_frames(paths)
        except SamplingError as e:
            return FilterResult(False, 'visual', f"frame sampling failed: {e}", error=True,
                                details={'access_denied': isinstance(e, AccessDeniedError)})
        except Exception as e:  # model/runtime failure: cannot judge -> do not accept
            return FilterResult(False, 'visual', f"{self.backend} error: {e}", error=True)
        finally:
            self.sampler.cleanup(video_id)
        accepted, score, reason = self.decide(scores, frame_details)
        details = {
            'frame_source': getattr(self.sampler, 'last_source', '') or getattr(self.sampler, 'source', ''),
            'frame_timestamps': [f.timestamp for f in frames],
            'frame_scores': [round(s, 3) for s in scores],
            'frame_brightness': lumas,
            'frame_details': frame_details,
            'time_of_day': time_of_day_from_brightness(lumas),
        }
        return FilterResult(accepted, 'visual', reason, score=round(score, 4), details=details)
