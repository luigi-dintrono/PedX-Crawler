"""Tests for the thumbnail frame source, the auto fallback and letterbox-aware brightness."""
import pytest
from PIL import Image

import filter_common as fc


class _Resp:
    def __init__(self, status=200, content=b'jpg'):
        self.status_code, self.content = status, content


class _Session:
    """Maps thumbnail name -> (status, bytes); anything unlisted returns 200."""

    def __init__(self, table=None, raises=None):
        self.table, self.raises, self.urls = table or {}, raises, []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if self.raises:
            raise self.raises
        name = url.rsplit('/', 1)[1].split('.')[0]
        return _Resp(*self.table.get(name, (200, b'jpg')))


# --- ThumbnailSampler --------------------------------------------------------------

def test_thumbnail_sampler_fetches_three_public_frames(tmp_path):
    session = _Session()
    s = fc.ThumbnailSampler(temp_dir=str(tmp_path), session=session)
    frames = s.sample('abcDEF_-123', 'https://youtu.be/x', 900)
    assert session.urls == [f'https://i.ytimg.com/vi/abcDEF_-123/{n}.jpg' for n in ('mq1', 'mq2', 'mq3')]
    assert [f.timestamp for f in frames] == [None, None, None]   # positions are unknown, not invented
    assert all(open(f.path, 'rb').read() == b'jpg' for f in frames)
    s.cleanup('abcDEF_-123')
    assert list(tmp_path.iterdir()) == []


def test_thumbnail_sampler_tolerates_one_missing_but_not_two(tmp_path):
    ok = fc.ThumbnailSampler(temp_dir=str(tmp_path), session=_Session({'mq3': (404, b'')}))
    assert len(ok.sample('v', 'u', 600)) == 2
    bad = fc.ThumbnailSampler(temp_dir=str(tmp_path), session=_Session({'mq2': (404, b''), 'mq3': (404, b'')}))
    with pytest.raises(fc.SamplingError, match='only 1 of 3'):
        bad.sample('v', 'u', 600)


def test_thumbnail_sampler_network_error_is_a_sampling_error(tmp_path):
    import requests
    s = fc.ThumbnailSampler(temp_dir=str(tmp_path), session=_Session(raises=requests.ConnectionError('down')))
    with pytest.raises(fc.SamplingError, match='down'):
        s.sample('v', 'u', 600)


# --- AutoSampler -----------------------------------------------------------------------

class _Stub:
    def __init__(self, source, error=None):
        self.source, self.error, self.calls, self.cleaned = source, error, 0, []

    def sample(self, video_id, url, duration):
        self.calls += 1
        if self.error:
            raise self.error
        return [fc.Frame(f'{self.source}.jpg', 1.0)]

    def cleanup(self, video_id):
        self.cleaned.append(video_id)


def test_auto_sampler_prefers_the_stream():
    stream, thumbs = _Stub('stream'), _Stub('thumbnails')
    auto = fc.AutoSampler(stream, thumbs)
    assert auto.sample('v', 'u', 600)[0].path == 'stream.jpg'
    assert auto.last_source == 'stream' and thumbs.calls == 0


def test_auto_sampler_falls_back_once_and_stays_on_thumbnails(capsys):
    stream, thumbs = _Stub('stream', error=fc.AccessDeniedError(403)), _Stub('thumbnails')
    auto = fc.AutoSampler(stream, thumbs)
    assert auto.sample('v1', 'u', 600)[0].path == 'thumbnails.jpg'
    assert auto.sample('v2', 'u', 600)[0].path == 'thumbnails.jpg'
    assert stream.calls == 1                      # not retried for the second video
    assert auto.last_source == 'thumbnails' and auto.stream_blocked
    assert capsys.readouterr().out.count('Note: YouTube refused the media stream') == 1
    auto.cleanup('v1')
    assert stream.cleaned == ['v1', 'v1'] and thumbs.cleaned == ['v1']


def test_auto_sampler_does_not_mask_other_failures():
    stream = _Stub('stream', error=fc.SamplingError('ffmpeg not found'))
    auto = fc.AutoSampler(stream, _Stub('thumbnails'))
    with pytest.raises(fc.SamplingError, match='ffmpeg not found'):
        auto.sample('v', 'u', 600)
    assert not auto.stream_blocked


def test_make_sampler_variants(tmp_path):
    assert isinstance(fc.make_sampler('thumbnails', str(tmp_path)), fc.ThumbnailSampler)
    assert isinstance(fc.make_sampler('stream', str(tmp_path), js_runtime=None), fc.FrameSampler)
    auto = fc.make_sampler('auto', str(tmp_path), js_runtime=None)
    assert isinstance(auto, fc.AutoSampler)
    with pytest.raises(ValueError):
        fc.make_sampler('carrier-pigeon', str(tmp_path))


class _ConstVisual(fc.VisualFilter):
    backend = 'const'

    def score_frames(self, paths):
        return [1.0] * len(paths), [{} for _ in paths]


def test_visual_result_records_frame_source(tmp_path):
    shade = Image.new('RGB', (8, 8), (90, 90, 90))
    paths = []
    for i in range(2):
        p = tmp_path / f't{i}.jpg'
        shade.save(p)
        paths.append(fc.Frame(str(p), None))

    class Sampler:
        source = 'thumbnails'

        def sample(self, *a):
            return paths

        def cleanup(self, video_id):
            pass
    r = _ConstVisual(sampler=Sampler()).check({'id': 'abc', 'video_url': 'u', 'duration_seconds': 600})
    assert r.accepted and r.details['frame_source'] == 'thumbnails'
    assert r.details['frame_timestamps'] == [None, None]


# --- letterbox-aware brightness ---------------------------------------------------------

def _letterboxed(tmp_path, content_value, size=(160, 120), bar=20, name='lb.png'):
    img = Image.new('L', size, 0)
    img.paste(Image.new('L', (size[0], size[1] - 2 * bar), content_value), (0, bar))
    path = tmp_path / name
    img.save(path)
    return str(path)


def test_black_bars_do_not_dilute_brightness(tmp_path):
    assert fc.frame_luma(_letterboxed(tmp_path, 200)) == pytest.approx(200, abs=1)  # not 200 * 80/120


def test_pillarbox_bars_are_stripped_too(tmp_path):
    img = Image.new('L', (160, 120), 0)
    img.paste(Image.new('L', (100, 120), 180), (30, 0))
    path = tmp_path / 'pb.png'
    img.save(path)
    assert fc.frame_luma(str(path)) == pytest.approx(180, abs=1)


def _noisy_night(tmp_path, seed=0):
    """A dark scene with realistic sensor noise (0-12) and one streetlight."""
    import random
    rng = random.Random(seed)
    img = Image.new('L', (160, 120))
    img.putdata([rng.randint(0, 12) for _ in range(160 * 120)])
    img.paste(Image.new('L', (20, 20), 250), (70, 50))
    path = tmp_path / f'night{seed}.png'
    img.save(path)
    return str(path), img


def test_dark_night_frame_is_not_cropped_to_its_bright_spot(tmp_path):
    from PIL import ImageStat
    path, img = _noisy_night(tmp_path)
    assert fc.strip_black_bars(img).size == img.size           # nothing mistaken for bars
    luma = fc.frame_luma(path)
    assert luma == pytest.approx(ImageStat.Stat(img).mean[0], abs=0.01)
    assert fc.time_of_day_from_brightness([luma]) == 'night'


def test_perfectly_black_uniform_scene_is_not_misread_as_bars_asymmetrically(tmp_path):
    # black only at the top (e.g. a dark sky above a lit road): not letterboxing
    img = Image.new('L', (160, 120), 150)
    img.paste(Image.new('L', (160, 30), 0), (0, 0))
    path = tmp_path / 'sky.png'
    img.save(path)
    assert fc.strip_black_bars(img).size == (160, 120)
    assert fc.frame_luma(str(path)) == pytest.approx(150 * 90 / 120, abs=1)


def test_bright_frame_without_bars_is_untouched(tmp_path):
    path = tmp_path / 'day.png'
    Image.new('L', (160, 120), 140).save(path)
    assert fc.frame_luma(str(path)) == pytest.approx(140, abs=1)
