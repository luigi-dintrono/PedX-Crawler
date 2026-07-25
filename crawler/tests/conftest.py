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


@pytest.fixture
def yolo_filter():
    """The YOLO filter module, with ultralytics stubbed (model load fails -> None)."""
    def _raise(*a, **k):
        raise RuntimeError("stubbed YOLO: no model in tests")
    _stub('ultralytics', YOLO=_raise)
    return _load('video_quality_filter_yolo.py', 'yolo_under_test')


@pytest.fixture
def internvl3_filter():
    """The InternVL3 filter module, with torch/transformers/PIL/torchvision stubbed."""
    torch = _stub('torch',
                  bfloat16='bfloat16', float32='float32',
                  cuda=types.SimpleNamespace(is_available=lambda: False),
                  backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))

    class _FakeModel:
        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    _stub('transformers',
          AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
          AutoModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeModel()))
    _stub('PIL', Image=types.SimpleNamespace(open=lambda *a, **k: None))
    _stub('PIL.Image', open=lambda *a, **k: None)
    tv = _stub('torchvision')
    _stub('torchvision.transforms', Compose=lambda *a, **k: None, Resize=object,
          ToTensor=object, Normalize=object, Lambda=object)
    tv.transforms = sys.modules['torchvision.transforms']
    _stub('torchvision.transforms.functional', InterpolationMode=types.SimpleNamespace(BICUBIC='bicubic'))
    return _load('video_quality_filter_internvl3.py', 'internvl3_under_test')


# --- Fake YouTube API client (shared by crawler tests) -----------------------

class _Req:
    def __init__(self, resp):
        self._resp = resp

    def execute(self):
        return self._resp


class _FakeSearch:
    def __init__(self, table, counter):
        self.table, self.counter = table, counter
        self.last_params = None

    def list(self, **params):
        self.counter['search'] += 1
        self.last_params = params
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


def make_search_item(vid, title="pedestrian crossing at intersection"):
    return {'id': {'videoId': vid},
            'snippet': {'title': title, 'channelTitle': f'ch-{vid}',
                        'channelId': f'cid-{vid}', 'publishedAt': '2026-06-01T00:00:00Z',
                        'thumbnails': {'high': {'url': f'https://i.ytimg.com/{vid}.jpg'}}}}


DISCOVERY_FIELDS = [
    'id', 'name', 'city', 'video', 'video_url', 'time_of_day', 'start_time', 'end_time',
    'region_code', 'channel_name', 'channel_url', 'published_at', 'country_code',
    'duration_seconds', 'view_count', 'like_count', 'comment_count', 'thumbnail_url',
    'latitude', 'longitude',
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


def make_details(vid, duration='PT2M30S', views='1234', lat=None, lon=None):
    rec = {}
    if lat is not None and lon is not None:
        rec = {'location': {'latitude': lat, 'longitude': lon}}
    return {'id': vid,
            'contentDetails': {'duration': duration},
            'statistics': {'viewCount': views, 'likeCount': '10', 'commentCount': '2'},
            'recordingDetails': rec}
