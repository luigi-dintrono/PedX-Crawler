"""Tests for scripts/validate_filters.py: the pure report logic and helpers (no network)."""
import pytest

pytest.importorskip('scipy')
from conftest import REPO_ROOT, _load_path  # noqa: E402


@pytest.fixture
def vf():
    return _load_path(REPO_ROOT / 'scripts' / 'validate_filters.py', 'validate_under_test')


def _item(label, group, name='Driving in Paris 4K', duration=1200):
    return {'label': label, 'group': group, 'name': name, 'duration_seconds': duration, 'tags': [],
            'description': '', 'live_broadcast': 'none', 'projection': '', 'frames': ['a', 'b', 'c']}


def _score(street, person=True, internvl=None):
    s = {'street_fraction': street, 'pedestrian': person, 'person_frames': 1, 'luma': [100.0] * 3}
    if internvl is not None:
        s['internvl'] = internvl
    return s


def test_report_counts_and_combinations(vf):
    items = {
        'p1': _item(1, 'curated'), 'p2': _item(1, 'curated'),
        'p3': _item(1, 'curated', duration=6 * 3600),             # metadata: too long
        'n1': _item(0, 'easy:vlog', 'my morning routine'),         # fails YOLO
        'n2': _item(0, 'hard:gta', 'GTA 5 driving gameplay'),      # fails metadata
        'n3': _item(0, 'hard:car-interior', 'Porsche walkaround'),  # passes everything
    }
    scores = {'p1': _score(1.0), 'p2': _score(0.67, person=False), 'p3': _score(1.0),
              'n1': _score(0.0, person=False), 'n2': _score(1.0), 'n3': _score(1.0)}
    rep = vf.build_report(items, scores)
    assert (rep['n_positives'], rep['n_negatives']) == (3, 3)
    c = rep['configs']
    assert c['metadata only']['kept_curated'] == [2, 3]
    assert c['YOLO street-scene only']['kept_curated'] == [3, 3]
    assert c['metadata + YOLO (default)']['kept_curated'] == [2, 3]
    assert c['metadata + YOLO + person']['kept_curated'] == [1, 3]
    assert c['metadata + YOLO (default)']['rejected_negatives'] == [2, 3]
    assert rep['by_group']['hard:gta']['metadata + YOLO (default)'] == [1, 1]
    assert rep['by_group']['hard:car-interior']['metadata + YOLO (default)'] == [0, 1]
    assert 'InternVL3 only' not in c                                  # no InternVL scores -> no such rows


def test_report_includes_internvl_when_scored(vf):
    items = {'p': _item(1, 'curated'), 'n': _item(0, 'easy:vlog', 'vlog')}
    scores = {'p': _score(1.0, internvl=0.9), 'n': _score(1.0, internvl=0.1)}
    c = vf.build_report(items, scores)['configs']
    assert c['InternVL3 only'] == {'kept_curated': [1, 1], 'rejected_negatives': [1, 1]}
    assert c['metadata + InternVL3']['kept_curated'] == [1, 1]
    assert vf.build_report(items, scores, ivl_threshold=0.95)['configs']['InternVL3 only']['kept_curated'] == [0, 1]


def test_report_skips_unscored_videos(vf):
    items = {'a': _item(1, 'curated'), 'b': _item(0, 'easy:vlog')}
    rep = vf.build_report(items, {'a': _score(1.0)})
    assert (rep['n_positives'], rep['n_negatives']) == (1, 0)


@pytest.mark.parametrize("duration,seconds", [('PT1H2M3S', 3723), ('PT45S', 45), ('P1DT1H', 90000), ('junk', 0)])
def test_iso_seconds(vf, duration, seconds):
    assert vf._iso_seconds(duration) == seconds


def test_wilson_and_fmt(vf):
    p, lo, hi = vf.wilson(50, 100)
    assert p == 0.5 and 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert vf.fmt(0, 0) == 'n/a' and '(3/4)' in vf.fmt(3, 4)
