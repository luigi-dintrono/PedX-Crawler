"""Tests for pedx-crawler.py: pure helpers, the search loop, dedup, and CSV I/O."""
import csv

import pytest
from conftest import FakeYouTube, make_search_item, make_details

TERM = "London street crossing pedestrian"


def _discovery(pedx, table, details, quality_filter=None):
    d = pedx.YouTubeDiscovery.__new__(pedx.YouTubeDiscovery)
    d.youtube = FakeYouTube(table, details)
    d.quota_tracker = pedx.QuotaTracker()
    d.quality_filter = quality_filter
    d.last_stats = {}
    return d


# --- pure helpers ------------------------------------------------------------

@pytest.mark.parametrize("iso,expected", [
    ('PT1H2M3S', '1:02:03'), ('PT2M30S', '0:02:30'), ('PT6S', '0:00:06'),
    ('PT1H', '1:00:00'), ('PT1H30S', '1:00:30'), ('PT0S', '0:00:00'), ('junk', '0:00:00'),
])
def test_format_duration(pedx, iso, expected):
    d = pedx.YouTubeDiscovery.__new__(pedx.YouTubeDiscovery)
    assert d._format_duration(iso) == expected


@pytest.mark.parametrize("iso,secs", [
    ('PT1H2M3S', 3723), ('PT2M30S', 150), ('PT6S', 6), ('PT0S', 0), ('junk', 0),
])
def test_iso_duration_seconds(pedx, iso, secs):
    d = pedx.YouTubeDiscovery.__new__(pedx.YouTubeDiscovery)
    assert d._iso_duration_seconds(iso) == secs


def test_time_of_day(pedx):
    d = pedx.YouTubeDiscovery.__new__(pedx.YouTubeDiscovery)
    assert d._extract_time_of_day('crossing at night') == 'night'
    assert d._extract_time_of_day('sunrise walk') == 'morning'
    assert d._extract_time_of_day('random title') == 'unknown'


def test_region_code(pedx):
    d = pedx.YouTubeDiscovery.__new__(pedx.YouTubeDiscovery)
    assert d._extract_region_code('New York') == 'US'
    assert d._extract_region_code('Nowhere') == 'UNKNOWN'


def test_convert_date_to_iso(pedx):
    assert pedx.convert_date_to_iso('2024-01-15') == '2024-01-15T00:00:00Z'
    with pytest.raises(SystemExit):
        pedx.convert_date_to_iso('15/01/2024')


def test_quota_percentage_zero_guard(pedx):
    qt = pedx.QuotaTracker(daily_limit=0)
    qt.add_search_request()
    assert qt.get_quota_percentage() == 0.0  # no ZeroDivisionError


def test_load_cities_country_map(pedx, tmp_path):
    f = tmp_path / "cities.txt"
    f.write_text("# comment\nLondon,GB\nTokyo\n  \nParis, fr\n", encoding='utf-8')
    cities, cc = pedx.load_cities(str(f))
    assert cities == ['London', 'Tokyo', 'Paris']
    assert cc == {'London': 'GB', 'Paris': 'FR'}  # CC upper-cased, Tokyo absent


# --- search loop -------------------------------------------------------------

def test_batched_details_and_quota(pedx):
    table = {(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2'), make_search_item('v3')]}}
    details = {v: make_details(v) for v in ['v1', 'v2', 'v3']}
    d = _discovery(pedx, table, details)
    res = d.search_videos('London', max_results=10, use_single_search=True)
    assert [r['id'] for r in res] == ['v1', 'v2', 'v3']
    # ONE batched details call for the whole page, not one per video.
    assert d.youtube.counter['details_calls'] == 1
    assert d.youtube.counter['details_ids'] == 3
    # 100 (search) + 1 (one details batch) = 101, NOT 100 + 3.
    assert d.quota_tracker.used_quota == 101


def test_pagination_dedup_and_exhaustion(pedx):
    table = {
        (TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')], 'nextPageToken': 'p2'},
        (TERM, 'p2'): {'items': [make_search_item('v2'), make_search_item('v3')]},  # v2 dup, no next
    }
    details = {v: make_details(v) for v in ['v1', 'v2', 'v3']}
    d = _discovery(pedx, table, details)
    res = d.search_videos('London', max_results=10, use_single_search=True)
    assert [r['id'] for r in res] == ['v1', 'v2', 'v3']
    assert d.youtube.counter['search'] == 2  # stops after exhaustion, no re-crawl


def test_shared_seen_ids_dedups_across_runs(pedx):
    table = {(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')]}}
    details = {v: make_details(v) for v in ['v1', 'v2']}
    d = _discovery(pedx, table, details)
    seen = {'v1'}  # pretend v1 came from a prior run / --exclude-existing
    res = d.search_videos('London', max_results=10, use_single_search=True, seen_ids=seen)
    assert [r['id'] for r in res] == ['v2']
    assert 'v2' in seen  # the shared set is updated in place


def test_malformed_item_skipped(pedx):
    bad = {'snippet': {'title': 'x'}}  # missing id.videoId
    table = {(TERM, None): {'items': [make_search_item('g1'), bad, make_search_item('g2')]}}
    details = {v: make_details(v) for v in ['g1', 'g2']}
    d = _discovery(pedx, table, details)
    res = d.search_videos('London', max_results=10, use_single_search=True)
    assert [r['id'] for r in res] == ['g1', 'g2']


def test_region_code_passed_to_search(pedx):
    table = {(TERM, None): {'items': [make_search_item('v1')]}}
    d = _discovery(pedx, table, {'v1': make_details('v1')})
    d.search_videos('London', max_results=5, use_single_search=True, region_code='GB')
    assert d.youtube._search.last_params.get('regionCode') == 'GB'


def test_enrichment_columns_and_html_unescape(pedx):
    item = make_search_item('v1', title='Ben &amp; Jerry&#39;s crossing')
    table = {(TERM, None): {'items': [item]}}
    details = {'v1': make_details('v1', duration='PT2M30S', views='4242', lat=51.5, lon=-0.1)}
    d = _discovery(pedx, table, details)
    res = d.search_videos('London', max_results=5, use_single_search=True)
    row = res[0]
    assert row['name'] == "Ben & Jerry's crossing"  # HTML entities decoded
    assert row['duration_seconds'] == 150
    assert row['view_count'] == '4242'
    assert row['latitude'] == 51.5 and row['longitude'] == -0.1
    assert row['thumbnail_url'].endswith('v1.jpg')


def test_quality_filter_integration_and_stats(pedx):
    class StubFilter:
        def filter_video(self, vd):
            return ('bad' not in vd['name'], 'stub reject')
    table = {(TERM, None): {'items': [
        make_search_item('q1', 'good crossing'),
        make_search_item('q2', 'bad crossing'),
        make_search_item('q3', 'good crossing')]}}
    details = {v: make_details(v) for v in ['q1', 'q2', 'q3']}
    d = _discovery(pedx, table, details, quality_filter=StubFilter())
    res = d.search_videos('London', max_results=10, use_single_search=True)
    assert [r['id'] for r in res] == ['q1', 'q3']
    assert d.last_stats['filtered'] == 1
    assert d.last_stats['kept'] == 2


def test_retry_on_transient_error(pedx, monkeypatch):
    monkeypatch.setattr(pedx.time, 'sleep', lambda *_: None)  # don't actually wait
    calls = {'n': 0}

    class FlakyReq:
        def execute(self):
            calls['n'] += 1
            if calls['n'] < 3:
                err = pedx.HttpError()
                err.resp = type('R', (), {'status': 503})()
                raise err
            return {'ok': True}

    d = pedx.YouTubeDiscovery.__new__(pedx.YouTubeDiscovery)
    assert d._execute_with_retry(FlakyReq(), 'test') == {'ok': True}
    assert calls['n'] == 3  # two failures then success


# --- CSV I/O -----------------------------------------------------------------

def _row(vid, city='London'):
    return {'id': vid, 'name': 'n', 'city': city, 'video': vid, 'video_url': 'u',
            'time_of_day': 'unknown', 'start_time': '0:00', 'end_time': '0:02:30',
            'region_code': 'GB', 'channel_name': 'ch', 'channel_url': 'cu',
            'published_at': '2026-06-01T00:00:00Z', 'country_code': 'GB',
            'duration_seconds': 150, 'view_count': '10', 'like_count': '1',
            'comment_count': '0', 'thumbnail_url': 't', 'latitude': '', 'longitude': ''}


def test_save_to_csv_new_file(pedx, tmp_path):
    out = tmp_path / "sub" / "discovery.csv"
    path = pedx.save_to_csv([_row('v1')], str(out))
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['id'] == 'v1'
    assert 'duration_seconds' in rows[0] and 'latitude' in rows[0]


def test_save_to_csv_bare_filename(pedx, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = pedx.save_to_csv([_row('v1')], 'discovery.csv')  # no directory component
    assert (tmp_path / path).exists()


def test_save_to_csv_append(pedx, tmp_path):
    out = str(tmp_path / "d.csv")
    pedx.save_to_csv([_row('v1')], out, append=True)
    pedx.save_to_csv([_row('v2')], out, append=True)
    with open(out, newline='', encoding='utf-8') as f:
        content = f.read()
    rows = list(csv.DictReader(content.splitlines()))
    assert [r['id'] for r in rows] == ['v1', 'v2']
    assert content.count('id,name,city') == 1  # header written exactly once


def test_load_seen_ids(pedx, tmp_path):
    out = str(tmp_path / "prior.csv")
    pedx.save_to_csv([_row('v1'), _row('v2')], out, append=True)
    seen = pedx.load_seen_ids([out, str(tmp_path / "missing.csv")])
    assert seen == {'v1', 'v2'}  # missing file is tolerated


def test_get_unique_filename(pedx, tmp_path):
    out = str(tmp_path / "d.csv")
    open(out, 'w').close()
    assert pedx.get_unique_filename(out) == str(tmp_path / "d_1.csv")
