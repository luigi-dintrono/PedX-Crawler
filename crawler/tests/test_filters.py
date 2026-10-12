"""Tests for the filter stack: metadata stage, keyword matching, time of day, frame
sampling schedule, the VisualFilter error handling, YOLO scoring and the InternVL3
score/device helpers. No model weights, network or ffmpeg needed."""
from datetime import datetime, timezone

import pytest

import filter_common as fc


def _video(**overrides):
    v = {'id': 'abc', 'name': 'Driving in Paris 4K', 'video_url': 'https://www.youtube.com/watch?v=abc',
         'duration_seconds': 1500, 'published_at': '2026-06-01T00:00:00Z', 'live_broadcast': 'none',
         'projection': 'rectangular', 'tags': [], 'description': ''}
    v.update(overrides)
    return v


# --- metadata stage ----------------------------------------------------------

@pytest.fixture
def meta():
    return fc.MetadataFilter(min_duration=240, max_duration=10800)


def test_metadata_accepts_typical_dashcam_video(meta):
    assert meta.check(_video()).accepted


@pytest.mark.parametrize("overrides,reason", [
    ({'live_broadcast': 'live'}, 'live broadcast'),
    ({'live_broadcast': 'upcoming'}, 'live broadcast'),
    ({'duration_seconds': 0}, 'duration unknown'),
    ({'duration_seconds': 45}, 'too short'),
    ({'duration_seconds': 4 * 3600}, 'too long'),
    ({'projection': '360'}, '360'),
    ({'name': 'Crash compilation 2025'}, "'compilation'"),
    ({'name': 'GTA 5 driving in Paris'}, "'gta'"),
    ({'name': 'Paris hyperlapse drive'}, "'hyperlapse'"),
    ({'name': 'Paris drive', 'tags': ['timelapse']}, "'timelapse'"),
    ({'description': 'quick one #shorts'}, 'YouTube Short'),
])
def test_metadata_rejections(meta, overrides, reason):
    r = meta.check(_video(**overrides))
    assert not r.accepted and r.stage == 'metadata' and reason in r.reason


@pytest.mark.parametrize("title", [
    '4K Night Drive on Tokyo Metropolitan EXPWY | 3x Speed | Music', 'Paris drive 2X faster',
    'Driving Rome (x4 speed)', 'Berlin drive fast forward', 'Lisbon speed up drive', '8× speed Lagos drive',
])
def test_speed_altered_titles_rejected(meta, title):
    r = meta.check(_video(name=title))
    assert not r.accepted and 'speed' in r.reason


@pytest.mark.parametrize("title", [
    'Driving in Paris at 30 km/h', 'Route 66 drive, 1.5x zoom lens test'.replace('zoom lens test', 'tour'),
    'Mach 2 Speedway drive', 'Driving the A1 - speed limit 130',
])
def test_speed_patterns_do_not_overreach(meta, title):
    assert meta.check(_video(name=title)).accepted


def test_negative_keywords_match_whole_words_only(meta):
    assert meta.check(_video(name='Driving to Niagara Falls')).accepted   # 'falls' is not 'fails'
    assert meta.check(_video(name='Driving past Agtan street')).accepted  # 'gta' inside a word


def test_duration_falls_back_to_end_time(meta):
    assert not meta.check(_video(duration_seconds='', end_time='0:00:45')).accepted
    assert meta.check(_video(duration_seconds='', end_time='0:25:00')).accepted


def test_max_duration_none_means_no_limit():
    assert fc.MetadataFilter(max_duration=None).check(_video(duration_seconds=7 * 3600)).accepted


def test_upload_age_uses_fixed_reference_date():
    ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
    f = fc.MetadataFilter(max_upload_months=12, reference_date=ref)
    assert not f.check(_video(published_at='2024-06-01T00:00:00Z')).accepted
    assert f.check(_video(published_at='2025-06-01T00:00:00Z')).accepted
    assert f.check(_video(published_at='')).accepted  # unknown date -> not rejected on age


def test_required_keywords_gate_is_multilingual():
    f = fc.MetadataFilter(required_keywords=['driving', '車載', 'manejando'])
    assert f.check(_video(name='東京 車載動画 首都高')).accepted   # CJK: substring match
    assert f.check(_video(name='Manejando en La Habana')).accepted
    r = f.check(_video(name='Paris at sunset'))
    assert not r.accepted and 'keyword' in r.reason


# --- time of day ----------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ('Tokyo Night Drive 4K', 'night'), ('Sunset Drive - Paris', 'evening'),
    ('Early morning drive through Bucharest', 'morning'), ('Driving in Lagos', 'unknown'),
    ('Darkness falls', 'unknown'),  # no substring false positive on 'dark'
    ('東京 夜 ドライブ', 'night'),
])
def test_time_of_day_from_title(title, expected):
    assert fc.time_of_day_from_title(title) == expected


@pytest.mark.parametrize("lumas,expected", [
    ([20, 30, 40], 'night'), ([120, 140, 90], 'day'), ([60, 70, 75], 'unknown'), ([], 'unknown'),
])
def test_time_of_day_from_brightness(lumas, expected):
    assert fc.time_of_day_from_brightness(lumas) == expected


def test_frame_luma(tmp_path):
    from PIL import Image
    path = tmp_path / 'f.jpg'
    Image.new('RGB', (8, 8), (200, 200, 200)).save(path)
    assert abs(fc.frame_luma(str(path)) - 200) < 2


# --- frame sampling --------------------------------------------------------------

def test_sampler_spreads_timestamps_across_video(tmp_path):
    s = fc.FrameSampler(temp_dir=str(tmp_path), n_frames=5, js_runtime=None)
    assert s.timestamps(1000) == [100.0, 300.0, 500.0, 700.0, 900.0]
    assert s.timestamps(None) == [0.0]
    assert all(t <= 8.0 for t in s.timestamps(10))  # never past the end


def test_sampler_ffmpeg_command_seeks_before_input(tmp_path):
    s = fc.FrameSampler(temp_dir=str(tmp_path), n_frames=2, js_runtime=None)
    cmd = s._ffmpeg_command('http://127.0.0.1:1/video', 12.5, tmp_path / 'f.jpg')
    assert cmd.index('-ss') < cmd.index('-i') and cmd[cmd.index('-ss') + 1] == '12.5'
    assert '-nostdin' in cmd and cmd[-1].endswith('f.jpg')


def test_sampler_cleanup_only_removes_that_video(tmp_path):
    s = fc.FrameSampler(temp_dir=str(tmp_path), js_runtime=None)
    for name in ['abc.10.0.jpg', 'abc.30.0.jpg', 'abcd.5.0.jpg']:
        (tmp_path / name).write_bytes(b'')
    s.cleanup('abc')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['abcd.5.0.jpg']


class _FakeSampler:
    def __init__(self, frames=None, error=None):
        self.frames, self.error, self.cleaned = frames or [], error, []

    def sample(self, video_id, url, duration):
        if self.error:
            raise self.error
        return self.frames

    def cleanup(self, video_id):
        self.cleaned.append(video_id)


class _ConstVisual(fc.VisualFilter):
    backend = 'const'

    def __init__(self, scores, **kw):
        super().__init__(**kw)
        self._scores = scores

    def score_frames(self, paths):
        return self._scores[:len(paths)], [{} for _ in paths]


def _frames(tmp_path, n, shade=30):
    from PIL import Image
    frames = []
    for i in range(n):
        p = tmp_path / f'abc.{i}.jpg'
        Image.new('RGB', (4, 4), (shade, shade, shade)).save(p)
        frames.append(fc.Frame(str(p), float(i)))
    return frames


def test_visual_filter_aggregates_and_cleans_up(tmp_path):
    sampler = _FakeSampler(_frames(tmp_path, 4))
    f = _ConstVisual([1, 1, 0, 1], threshold=0.5, sampler=sampler)
    r = f.check(_video())
    assert r.accepted and r.score == 0.75 and not r.error
    assert r.details['time_of_day'] == 'night'           # dark frames
    assert r.details['frame_timestamps'] == [0.0, 1.0, 2.0, 3.0]
    assert sampler.cleaned == ['abc']


def test_visual_filter_sampling_failure_is_an_error_not_a_pass(tmp_path):
    sampler = _FakeSampler(error=fc.SamplingError('403'))
    r = _ConstVisual([1], sampler=sampler).check(_video())
    assert not r.accepted and r.error and 'sampling failed' in r.reason
    assert sampler.cleaned == ['abc']


def test_visual_filter_access_denied_is_flagged(tmp_path):
    sampler = _FakeSampler(error=fc.AccessDeniedError(403))
    r = _ConstVisual([1], sampler=sampler).check(_video())
    assert not r.accepted and r.error and r.details['access_denied'] is True
    assert 'PO-token' in r.reason and '--ytdlp-cookies' in r.reason


def test_visual_filter_model_failure_is_an_error(tmp_path):
    class Broken(_ConstVisual):
        def score_frames(self, paths):
            raise RuntimeError('CUDA OOM')
    r = Broken([], sampler=_FakeSampler(_frames(tmp_path, 1))).check(_video())
    assert not r.accepted and r.error and 'CUDA OOM' in r.reason


# --- YOLO backend ------------------------------------------------------------------

@pytest.mark.parametrize("counts,expected", [
    ({'person': 1, 'car': 1}, (True, True)),
    ({'car': 2}, (True, False)),
    ({'traffic light': 1, 'person': 1}, (True, True)),
    ({'person': 3}, (False, False)),          # people indoors: no longer a street scene
    ({}, (False, False)),
])
def test_yolo_classify_frame(yolo_filter, counts, expected):
    assert yolo_filter.classify_frame(counts) == expected


def test_yolo_decision_requires_street_and_pedestrian(yolo_filter, tmp_path):
    from conftest import FakeYOLO
    frames = _frames(tmp_path, 4, shade=150)
    f = yolo_filter.VideoQualityFilter(sampler=_FakeSampler(frames))
    FakeYOLO.script = [[2, 2], [2, 0], [56], [7]]          # car, car+person, chair, truck
    r = f.check(_video())
    assert r.accepted and r.score == 0.75
    FakeYOLO.script = [[2], [2], [2], [2]]                  # street but nobody on it
    r = f.check(_video())
    assert not r.accepted and 'no person' in r.reason
    FakeYOLO.script = [[0, 0], [0], [56], []]               # people, no street
    assert not f.check(_video()).accepted


def test_yolo_model_load_failure_raises(yolo_filter, tmp_path, monkeypatch):
    """A filter whose model cannot load must raise, not silently pass every video."""
    import sys
    import types

    def _raise(*a, **k):
        raise OSError("no weights")
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=_raise))
    with pytest.raises(RuntimeError):
        yolo_filter.VideoQualityFilter(temp_dir=str(tmp_path))


# --- InternVL3 backend ---------------------------------------------------------------

@pytest.mark.parametrize("device", ['cpu', 'auto'])
def test_internvl3_dtype_matches_resolved_device(internvl3_filter, tmp_path, device):
    """On CPU (incl. auto->cpu with no GPU) weights and inputs must both be float32."""
    f = internvl3_filter.VideoQualityFilterInternVL3(device=device, temp_dir=str(tmp_path))
    assert f.device == 'cpu'
    assert f.model_dtype == 'float32'  # stubbed torch.float32 sentinel
    assert f.yes_ids and f.no_ids and not set(f.yes_ids) & set(f.no_ids)


@pytest.mark.parametrize("p_yes,p_no,expected", [
    (0.9, 0.1, 0.9), (0.2, 0.6, 0.25), (0.0, 0.0, 0.0),
])
def test_internvl3_yes_score(internvl3_filter, p_yes, p_no, expected):
    assert internvl3_filter.yes_score(p_yes, p_no) == pytest.approx(expected)


def test_internvl3_model_load_failure_raises(internvl3_filter, tmp_path, monkeypatch):
    import sys

    def _raise(*a, **k):
        raise OSError("no weights")
    monkeypatch.setattr(sys.modules['transformers'].AutoModelForImageTextToText, 'from_pretrained', _raise)
    with pytest.raises(RuntimeError):
        internvl3_filter.VideoQualityFilterInternVL3(device='cpu', temp_dir=str(tmp_path))
