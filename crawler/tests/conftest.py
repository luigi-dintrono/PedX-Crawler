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


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, CRAWLER_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pedx():
    """The crawler module, with googleapiclient/dotenv stubbed out."""
    _stub('googleapiclient')
    _stub('googleapiclient.discovery', build=lambda *a, **k: None)
    _stub('googleapiclient.errors', HttpError=type('HttpError', (Exception,), {}))
    _stub('dotenv', load_dotenv=lambda *a, **k: None)
    return _load('pedx-crawler.py', 'pedx_under_test')


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


def make_details(vid, duration='PT2M30S', views='1234', lat=None, lon=None):
    rec = {}
    if lat is not None and lon is not None:
        rec = {'location': {'latitude': lat, 'longitude': lon}}
    return {'id': vid,
            'contentDetails': {'duration': duration},
            'statistics': {'viewCount': views, 'likeCount': '10', 'commentCount': '2'},
            'recordingDetails': rec}
