"""Shared pytest helpers for the PedX crawler tests.

The modules under test have hyphenated filenames and heavy optional deps
(googleapiclient, ultralytics, torch, ...). We load them via importlib and stub
the externals so the suite runs fast with only pytest + the stdlib installed.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

CRAWLER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CRAWLER_DIR.parent
# The crawler imports its sibling modules (filter_common, gazetteer, ...) by name, as
# it does when run as `python crawler/pedx-crawler.py`.
if str(CRAWLER_DIR) not in sys.path:
    sys.path.insert(0, str(CRAWLER_DIR))


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load(filename, module_name):
    return _load_path(CRAWLER_DIR / filename, module_name)


@pytest.fixture
def pedx():
    """The crawler module, with googleapiclient/dotenv stubbed out."""
    _stub('googleapiclient')
    _stub('googleapiclient.discovery', build=lambda *a, **k: None)
    _stub('googleapiclient.errors', HttpError=type('HttpError', (Exception,), {}))
    _stub('dotenv', load_dotenv=lambda *a, **k: None)
    return _load('pedx-crawler.py', 'pedx_under_test')


@pytest.fixture
def bridge():
    """The Crawler -> PedX-Insight converter (scripts/discovery_to_mapping.py).

    Pure stdlib, so nothing needs stubbing; it lives outside crawler/ hence the
    explicit path.
    """
    return _load_path(REPO_ROOT / 'scripts' / 'discovery_to_mapping.py', 'bridge_under_test')


class _FakeBoxes:
    def __init__(self, class_ids):
        self.cls = types.SimpleNamespace(tolist=lambda: list(class_ids))


class _FakeYOLOResult:
    names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
             9: 'traffic light', 11: 'stop sign', 56: 'chair'}

    def __init__(self, class_ids):
        self.boxes = _FakeBoxes(class_ids)


class FakeYOLO:
    """Returns pre-programmed detections (class ids) per frame, in call order."""
    script = []

    def __init__(self, *a, **k):
        pass

    def __call__(self, paths, **kwargs):
        return [_FakeYOLOResult(FakeYOLO.script[i] if i < len(FakeYOLO.script) else [])
                for i in range(len(paths))]


@pytest.fixture
def yolo_filter():
    """The YOLO filter module, with ultralytics stubbed by FakeYOLO."""
    FakeYOLO.script = []
    _stub('ultralytics', YOLO=FakeYOLO)
    return _load('video_quality_filter_yolo.py', 'yolo_under_test')


@pytest.fixture
def internvl3_filter():
    """The InternVL3 filter module, with torch/transformers stubbed (no weights)."""
    _stub('torch',
          bfloat16='bfloat16', float32='float32',
          cuda=types.SimpleNamespace(is_available=lambda: False),
          backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))

    class _FakeModel:
        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [len(text.strip())]  # deterministic: every 'yes' -> 3, every 'no' -> 2

    _stub('transformers',
          AutoProcessor=types.SimpleNamespace(
              from_pretrained=lambda *a, **k: types.SimpleNamespace(tokenizer=_FakeTokenizer())),
          AutoModelForImageTextToText=types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel()))
    return _load('video_quality_filter_internvl3.py', 'internvl3_under_test')


# --- Fake YouTube API client (shared by crawler tests) -----------------------

class _Req:
    def __init__(self, resp):
        self._resp = resp

    def execute(self):
        return self._resp


class _FakeSearch:
    """search.list keyed by (q, pageToken); every call's params are recorded."""

    def __init__(self, table, counter):
        self.table, self.counter = table, counter
        self.last_params = None
        self.calls = []

    def list(self, **params):
        self.counter['search'] += 1
        self.last_params = params
        self.calls.append(params)
        return _Req(self.table.get((params['q'], params.get('pageToken')), {'items': []}))


class _FakeVideos:
    def __init__(self, details, counter):
        self.details, self.counter = details, counter

    def list(self, part, id):
        self.counter['details_calls'] += 1
        ids = id.split(',')
        self.counter['details_ids'] += len(ids)
        return _Req({'items': [self.details[i] for i in ids if i in self.details]})


class FakeYouTube:
    def __init__(self, table, details):
        self.counter = {'search': 0, 'details_calls': 0, 'details_ids': 0}
        self._search = _FakeSearch(table, self.counter)
        self._videos = _FakeVideos(details, self.counter)

    def search(self):
        return self._search

    def videos(self):
        return self._videos


def make_search_item(vid, title="Driving in London 4K"):
    return {'id': {'videoId': vid},
            'snippet': {'title': title, 'channelTitle': f'ch-{vid}',
                        'channelId': f'cid-{vid}', 'publishedAt': '2026-06-01T00:00:00Z',
                        'liveBroadcastContent': 'none',
                        'thumbnails': {'high': {'url': f'https://i.ytimg.com/{vid}.jpg'}}}}


DISCOVERY_FIELDS = [
    'id', 'name', 'city', 'video', 'video_url', 'time_of_day', 'start_time', 'end_time',
    'region_code', 'channel_name', 'channel_url', 'published_at', 'country_code',
    'duration_seconds', 'view_count', 'like_count', 'comment_count', 'thumbnail_url',
    'latitude', 'longitude',
    'channel_id', 'definition', 'projection', 'default_audio_language', 'live_broadcast',
    'time_of_day_source', 'country_code_source',
    'search_profile', 'search_query', 'search_language', 'search_duration', 'search_rank',
    'filter_backend', 'filter_score', 'run_id',
]


def write_discovery_csv(path, rows, fields=None):
    """Write a discovery CSV (current schema by default) for the bridge tests."""
    import csv
    fields = fields or DISCOVERY_FIELDS
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})


def discovery_row(vid, city='London', cc='GB', time_of_day='unknown',
                  end_time='0:02:30', published_at='2023-08-29T10:00:00Z',
                  channel_name='ChanX'):
    """One discovery CSV row, keyword-overridable."""
    return {'id': vid, 'name': f'{city} pedestrian crossing', 'city': city, 'video': vid,
            'video_url': f'https://www.youtube.com/watch?v={vid}', 'time_of_day': time_of_day,
            'start_time': '0:00', 'end_time': end_time, 'region_code': cc,
            'channel_name': channel_name, 'channel_url': f'https://youtube.com/channel/{vid}',
            'published_at': published_at, 'country_code': cc}


def make_details(vid, duration='PT25M', views='1234', lat=None, lon=None, title=None,
                 live='none', projection='rectangular', definition='hd', tags=None,
                 description=''):
    rec = {}
    if lat is not None and lon is not None:
        rec = {'location': {'latitude': lat, 'longitude': lon}}
    snippet = {'title': title or 'Driving in London 4K', 'channelTitle': f'ch-{vid}',
               'channelId': f'UC{vid:0>22}'[:24], 'publishedAt': '2026-06-01T00:00:00Z',
               'liveBroadcastContent': live, 'description': description,
               'thumbnails': {'high': {'url': f'https://i.ytimg.com/{vid}.jpg'}}}
    if tags:
        snippet['tags'] = tags
    return {'id': vid,
            'snippet': snippet,
            'contentDetails': {'duration': duration, 'definition': definition,
                               'projection': projection},
            'statistics': {'viewCount': views, 'likeCount': '10', 'commentCount': '2'},
            'recordingDetails': rec}
