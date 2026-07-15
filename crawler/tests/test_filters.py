"""Tests for the two quality filters: Stage-1 metadata logic (no model/network),
YOLO scoring rules, and the InternVL3 device/dtype resolution."""
import pytest


# --- YOLO filter -------------------------------------------------------------

@pytest.fixture
def yolo(yolo_filter, tmp_path):
    return yolo_filter.VideoQualityFilter(max_upload_months=36, temp_dir=str(tmp_path))


def _video(name='pedestrian crossing at intersection', url='https://youtube.com/watch?v=x',
           end_time='0:02:00', published_at='2026-06-01T00:00:00Z'):
    return {'name': name, 'video_url': url, 'end_time': end_time, 'published_at': published_at}


def test_stage1_requires_positive_keyword(yolo):
    ok, _ = yolo._stage1_metadata_filter(_video(name='funny cat video'))
    assert ok is False


def test_stage1_rejects_negative_keyword(yolo):
    ok, reason = yolo._stage1_metadata_filter(_video(name='crosswalk fails compilation'))
    assert ok is False and 'negative keyword' in reason


def test_stage1_rejects_shorts_url(yolo):
    ok, reason = yolo._stage1_metadata_filter(_video(url='https://youtube.com/shorts/abc'))
    assert ok is False and 'Short' in reason


@pytest.mark.parametrize("end_time,ok_expected", [
    ('0:00:20', False),   # 20s < 30s -> reject
    ('0:00:45', True),    # 45s -> ok
    ('0:25:00', False),   # 25min > 20min -> reject
    ('0:10:00', True),    # 10min -> ok
])
def test_stage1_duration_bounds(yolo, end_time, ok_expected):
    ok, _ = yolo._stage1_metadata_filter(_video(end_time=end_time))
    assert ok is ok_expected


def test_stage1_upload_age(yolo):
    assert yolo._stage1_metadata_filter(_video(published_at='2019-01-01T00:00:00Z'))[0] is False  # too old
    assert yolo._stage1_metadata_filter(_video(published_at='2026-06-01T00:00:00Z'))[0] is True    # recent
    assert yolo._stage1_metadata_filter(_video(published_at=''))[0] is True                        # unknown -> pass


@pytest.mark.parametrize("counts,score", [
    ({'person': 1, 'car': 1}, 1.0),          # person + vehicle
    ({'person': 2}, 1.0),                    # multiple people
    ({'traffic light': 1, 'person': 1}, 1.0),  # light + activity
    ({}, 0.0),                               # empty scene
    ({'car': 1}, 0.0),                       # lone vehicle, no person/light
])
def test_yolo_frame_scoring(yolo, counts, score):
    assert yolo._calculate_frame_score(counts) == score


@pytest.mark.parametrize("end_time,secs", [
    ('1:02:03', 3723), ('2:30', 150), ('0:00:45', 45), ('', None), ('bad', None),
])
def test_extract_duration_seconds(yolo, end_time, secs):
    assert yolo._extract_duration_seconds({'end_time': end_time}) == secs


# --- InternVL3 filter --------------------------------------------------------

@pytest.mark.parametrize("device", ['cpu', 'auto'])
def test_internvl3_dtype_matches_resolved_device(internvl3_filter, tmp_path, device):
    """Regression guard: on CPU (incl. auto->cpu with no GPU) the weight dtype and
    the input-cast dtype must both be float32, or inference crashes."""
    f = internvl3_filter.VideoQualityFilterInternVL3(device=device, temp_dir=str(tmp_path))
    assert f.device == 'cpu'
    assert f.model_dtype == 'float32'  # stubbed torch.float32 sentinel


def test_internvl3_response_to_score(internvl3_filter, tmp_path):
    f = internvl3_filter.VideoQualityFilterInternVL3(device='cpu', temp_dir=str(tmp_path))
    assert f._response_to_score('Yes.') == 1.0
    assert f._response_to_score('no') == 0.0
    assert f._response_to_score('maybe?') == 0.0  # unclear -> reject
