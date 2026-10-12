"""End-to-end tests for main(): full crawl flow with a mocked YouTube client,
covering country/language resolution, YouTube search filters, enrichment, provenance
(decisions.csv + manifest.json), always-save, --append dedup and fail-closed filters."""
import csv
import json
import sys
import types

import pytest
from conftest import FakeYouTube, make_search_item, make_details

TERM = "London driving 4K"  # dashcam profile, English (GB's primary language)


def _run_main(pedx, monkeypatch, fake, argv):
    monkeypatch.setattr(pedx, 'build', lambda *a, **k: fake)
    monkeypatch.setattr(pedx.sys, 'argv', argv)
    pedx.main()


def _read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _cities(tmp_path, text="London,GB\n"):
    cities = tmp_path / "cities.txt"
    cities.write_text(text, encoding='utf-8')
    return cities


def test_main_end_to_end(pedx, tmp_path, monkeypatch):
    out = tmp_path / "out.csv"
    fake = FakeYouTube({(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')]}},
                       {v: make_details(v, lat=51.5, lon=-0.12) for v in ['v1', 'v2']})

    _run_main(pedx, monkeypatch, fake,
              ['prog', '--api-key', 'x', '--cities-file', str(_cities(tmp_path)),
               '--output', str(out), '--per-city', '10', '--runs-dir', str(tmp_path / 'runs')])

    rows = _read(out)
    assert [r['id'] for r in rows] == ['v1', 'v2']
    assert rows[0]['country_code'] == 'GB' and rows[0]['country_code_source'] == 'cities_file'
    assert rows[0]['latitude'] == '51.5'                   # enrichment column populated
    assert rows[0]['search_profile'] == 'dashcam'
    params = fake._search.calls
    assert {p['videoDuration'] for p in params} == {'medium', 'long'}  # balanced: both streams
    assert all(p['videoDefinition'] == 'high' and p['regionCode'] == 'GB' for p in params)
    assert all(p.get('relevanceLanguage') == 'en' for p in params)

    run_dirs = list((tmp_path / 'runs').iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'
    assert manifest['parameters']['profile'] == 'dashcam'
    assert manifest['cities'][0]['kept'] == 2
    assert manifest['quota_used'] >= 200
    decisions = _read(run_dirs[0] / 'decisions.csv')
    assert [d['decision'] for d in decisions] == ['accept', 'accept']
    assert rows[0]['run_id'] == manifest['run_id']


def test_main_append_is_idempotent(pedx, tmp_path, monkeypatch):
    out = tmp_path / "out.csv"
    table = {(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')]}}
    details = {v: make_details(v) for v in ['v1', 'v2']}
    argv = ['prog', '--api-key', 'x', '--cities-file', str(_cities(tmp_path)),
            '--output', str(out), '--per-city', '10', '--append', '--runs-dir', str(tmp_path / 'runs')]

    _run_main(pedx, monkeypatch, FakeYouTube(table, details), argv)
    _run_main(pedx, monkeypatch, FakeYouTube(table, details), argv)  # second run: all excluded

    rows = _read(out)
    assert [r['id'] for r in rows] == ['v1', 'v2']  # no duplicates from the second run
    assert open(out, encoding='utf-8').read().count('id,name,city') == 1  # header once


def test_dry_run_spends_nothing(pedx, tmp_path, monkeypatch, capsys):
    fake = FakeYouTube({}, {})
    _run_main(pedx, monkeypatch, fake, ['prog', '--dry-run', '--cities-file',
                                        str(_cities(tmp_path, "Tokyo,JP\nLondon,GB\n"))])
    out = capsys.readouterr().out
    assert fake.counter['search'] == 0
    assert 'Tokyo 車載動画' in out and 'London driving 4K' in out
    assert 'Estimated search quota: 600' in out  # balanced: JP 4 streams + GB 2 streams


def test_filter_init_failure_aborts(pedx, tmp_path, monkeypatch):
    """--filter yolo with an unloadable model must exit non-zero, not crawl unfiltered."""
    def _raise(*a, **k):
        raise OSError("no weights")
    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=_raise))
    monkeypatch.delitem(sys.modules, 'video_quality_filter_yolo', raising=False)
    fake = FakeYouTube({}, {})
    with pytest.raises(SystemExit) as exc:
        _run_main(pedx, monkeypatch, fake, ['prog', '--api-key', 'x', '--filter', 'yolo',
                                            '--cities-file', str(_cities(tmp_path)),
                                            '--output', str(tmp_path / 'o.csv'),
                                            '--temp-dir', str(tmp_path / 'tmp')])
    assert exc.value.code == 1
    assert fake.counter['search'] == 0
