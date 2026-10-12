"""Tests for pedx-crawler.py: pure helpers, the search loop, sampling, dedup, CSV I/O."""
import csv

import pytest
from conftest import FakeYouTube, make_search_item, make_details

TERM = "London driving 4K"


def _stream(pedx, query=TERM, language='en', duration='any'):
    return pedx.SearchStream(query=query, language=language, duration=duration)


def _discovery(pedx, table, details, **kwargs):
    return pedx.YouTubeDiscovery(youtube=FakeYouTube(table, details), **kwargs)


def _search(d, pedx, streams=None, max_results=10, **kwargs):
    return d.search_city('London', streams or [_stream(pedx)], max_results, '2020-01-01T00:00:00Z', **kwargs)


# --- pure helpers ------------------------------------------------------------

@pytest.mark.parametrize("iso,expected", [
    ('PT1H2M3S', '1:02:03'), ('PT2M30S', '0:02:30'), ('PT6S', '0:00:06'),
    ('PT1H', '1:00:00'), ('PT1H30S', '1:00:30'), ('PT0S', '0:00:00'), ('junk', '0:00:00'),
    ('P1DT2H', '26:00:00'),
])
def test_format_duration(pedx, iso, expected):
    assert pedx.YouTubeDiscovery._format_duration(iso) == expected


@pytest.mark.parametrize("iso,secs", [
    ('PT1H2M3S', 3723), ('PT2M30S', 150), ('PT6S', 6), ('PT0S', 0), ('junk', 0), ('P0D', 0),
])
def test_iso_duration_seconds(pedx, iso, secs):
    assert pedx.YouTubeDiscovery._iso_duration_seconds(iso) == secs


def test_convert_date_to_iso(pedx):
    assert pedx.convert_date_to_iso('2024-01-15') == '2024-01-15T00:00:00Z'
    with pytest.raises(SystemExit):
        pedx.convert_date_to_iso('15/01/2024')


def test_quota_percentage_zero_guard(pedx):
    qt = pedx.QuotaTracker(daily_limit=0)
    qt.add_search_request()
    assert qt.get_quota_percentage() == 0.0  # no ZeroDivisionError


def test_load_cities_formats(pedx, tmp_path):
    f = tmp_path / "cities.txt"
    f.write_text("# comment\nLondon,GB\nTokyo\n  \nParis, fr\nMoscow,RU,Москва\n", encoding='utf-8-sig')
    entries = pedx.load_cities(str(f))
    assert [(e.name, e.country_code, e.local_name) for e in entries] == [
        ('London', 'GB', None), ('Tokyo', None, None), ('Paris', 'FR', None), ('Moscow', 'RU', 'Москва')]


def test_interleave_round_robin(pedx):
    assert pedx.interleave([[1, 2, 3], ['a'], ['x', 'y']]) == [1, 'a', 'x', 2, 'y', 3]


# --- city planning -------------------------------------------------------------

def test_plan_city_uses_gazetteer_country_and_language(pedx):
    gz = pedx.Gazetteer()
    if not gz.available:
        pytest.skip('geonamescache not installed')
    profile = pedx.PROFILES['dashcam']
    plan = pedx.plan_city(pedx.CityEntry('Tokyo'), gz, profile, 'auto', ['medium', 'long'], False)
    assert plan.country_code == 'JP' and plan.country_code_source == 'gazetteer'
    assert plan.languages == ['ja', 'en']
    assert [s.query for s in plan.streams] == ['Tokyo 車載動画', 'Tokyo 車載動画',
                                               'Tokyo driving 4K', 'Tokyo driving 4K']
    explicit = pedx.plan_city(pedx.CityEntry('London', 'GB'), gz, profile, 'auto', ['any'], False)
    assert explicit.country_code_source == 'cities_file' and explicit.languages == ['en']


# --- search loop -------------------------------------------------------------

def test_batched_details_and_quota(pedx):
    table = {(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2'), make_search_item('v3')]}}
    details = {v: make_details(v) for v in ['v1', 'v2', 'v3']}
    d = _discovery(pedx, table, details)
    res = _search(d, pedx)
    assert [r['id'] for r in res] == ['v1', 'v2', 'v3']
    # ONE batched details call for the whole page, not one per video.
    assert d.youtube.counter['details_calls'] == 1
    assert d.youtube.counter['details_ids'] == 3
    assert d.quota_tracker.used_quota == 101  # 100 (search) + 1 (one details batch)


def test_search_parameters(pedx):
    table = {(TERM, None): {'items': [make_search_item('v1')]}}
    d = _discovery(pedx, table, {'v1': make_details('v1')}, definition='high')
    d.search_city('London', [_stream(pedx, language='en', duration='long')], 5,
                  '2020-01-01T00:00:00Z', '2025-01-01T00:00:00Z', region_code='GB')
    params = d.youtube._search.last_params
    assert params['videoDuration'] == 'long'
    assert params['videoDefinition'] == 'high'
    assert params['relevanceLanguage'] == 'en'
    assert params['regionCode'] == 'GB'
    assert params['publishedBefore'] == '2025-01-01T00:00:00Z'


def test_any_filters_are_omitted(pedx):
    table = {(TERM, None): {'items': []}}
    d = _discovery(pedx, table, {}, definition='any')
    _search(d, pedx, streams=[_stream(pedx, language=None, duration='any')])
    params = d.youtube._search.last_params
    assert not {'videoDuration', 'videoDefinition', 'relevanceLanguage'} & set(params)


def test_pagination_dedup_and_exhaustion(pedx):
    table = {
        (TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')], 'nextPageToken': 'p2'},
        (TERM, 'p2'): {'items': [make_search_item('v2'), make_search_item('v3')]},  # v2 dup, no next
    }
    details = {v: make_details(v) for v in ['v1', 'v2', 'v3']}
    d = _discovery(pedx, table, details)
    res = _search(d, pedx)
    assert [r['id'] for r in res] == ['v1', 'v2', 'v3']
    assert d.youtube.counter['search'] == 2  # stops after exhaustion, no re-crawl
    assert [r['search_rank'] for r in res] == [1, 2, 4]  # rank counts every returned item


def test_repeated_page_token_stops(pedx):
    table = {(TERM, None): {'items': [make_search_item('v1')], 'nextPageToken': 'p2'},
             (TERM, 'p2'): {'items': [make_search_item('v2')], 'nextPageToken': 'p2'}}
    d = _discovery(pedx, table, {v: make_details(v) for v in ['v1', 'v2']})
    _search(d, pedx)
    assert d.youtube.counter['search'] == 2  # the looping token does not cost 10 pages


def test_max_pages_per_city(pedx):
    table = {(TERM, None if i == 0 else f'p{i}'): {'items': [], 'nextPageToken': f'p{i + 1}'}
             for i in range(20)}
    d = _discovery(pedx, table, {}, max_pages=3)
    _search(d, pedx)
    assert d.youtube.counter['search'] == 3


def test_balanced_sampling_interleaves_streams(pedx):
    a, b = 'q-medium', 'q-long'
    table = {(a, None): {'items': [make_search_item('a1'), make_search_item('a2')]},
             (b, None): {'items': [make_search_item('b1'), make_search_item('b2')]}}
    details = {v: make_details(v) for v in ['a1', 'a2', 'b1', 'b2']}
    d = _discovery(pedx, table, details, sampling='balanced')
    res = _search(d, pedx, streams=[_stream(pedx, a), _stream(pedx, b)], max_results=2)
    assert [r['id'] for r in res] == ['a1', 'b1']  # one from each stream, not a1, a2
    assert d.youtube.counter['search'] == 2


def test_greedy_sampling_stops_early(pedx):
    a, b = 'q-medium', 'q-long'
    table = {(a, None): {'items': [make_search_item('a1'), make_search_item('a2')]},
             (b, None): {'items': [make_search_item('b1')]}}
    details = {v: make_details(v) for v in ['a1', 'a2', 'b1']}
    d = _discovery(pedx, table, details, sampling='greedy')
    res = _search(d, pedx, streams=[_stream(pedx, a), _stream(pedx, b)], max_results=2)
    assert [r['id'] for r in res] == ['a1', 'a2']
    assert d.youtube.counter['search'] == 1  # second stream never paid for


def test_shared_seen_ids_dedups_across_runs(pedx):
    table = {(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')]}}
    details = {v: make_details(v) for v in ['v1', 'v2']}
    d = _discovery(pedx, table, details)
    seen = {'v1'}  # pretend v1 came from a prior run / --exclude-existing
    res = _search(d, pedx, seen_ids=seen)
    assert [r['id'] for r in res] == ['v2']
    assert 'v2' in seen  # the shared set is updated in place


def test_malformed_item_skipped(pedx):
    bad = {'snippet': {'title': 'x'}}  # missing id.videoId
    table = {(TERM, None): {'items': [make_search_item('g1'), bad, make_search_item('g2')]}}
    details = {v: make_details(v) for v in ['g1', 'g2']}
    d = _discovery(pedx, table, details)
    assert [r['id'] for r in _search(d, pedx)] == ['g1', 'g2']


def test_enrichment_columns_and_html_unescape(pedx):
    item = make_search_item('v1', title='Ben &amp; Jerry&#39;s drive')
    table = {(TERM, None): {'items': [item]}}
    details = {'v1': make_details('v1', duration='PT2M30S', views='4242', lat=51.5, lon=-0.1,
                                  title='Ben &amp; Jerry&#39;s night drive')}
    d = _discovery(pedx, table, details, run_id='r1', profile='dashcam')
    row = _search(d, pedx)[0]
    assert row['name'] == "Ben & Jerry's night drive"  # HTML entities decoded
    assert row['duration_seconds'] == 150
    assert row['view_count'] == '4242'
    assert row['latitude'] == 51.5 and row['longitude'] == -0.1
    assert row['thumbnail_url'].endswith('v1.jpg')
    assert row['channel_id'].startswith('UC') and row['definition'] == 'hd'
    assert row['time_of_day'] == 'night' and row['time_of_day_source'] == 'title'
    assert (row['search_query'], row['search_rank'], row['run_id']) == (TERM, 1, 'r1')


def test_metadata_filter_rejects_and_logs(pedx, tmp_path):
    from provenance import DecisionLog
    table = {(TERM, None): {'items': [make_search_item(v) for v in ['ok', 'short', 'live', 'game']]}}
    details = {'ok': make_details('ok'), 'short': make_details('short', duration='PT45S'),
               'live': make_details('live', duration='PT0S', live='live'),
               'game': make_details('game', title='GTA 5 driving in London')}
    log = DecisionLog(str(tmp_path / 'decisions.csv'))
    d = _discovery(pedx, table, details, metadata_filter=pedx.MetadataFilter(), decision_log=log,
                   run_id='r1')
    res = _search(d, pedx)
    log.close()
    assert [r['id'] for r in res] == ['ok']
    assert d.last_stats['rejected_metadata'] == 3
    rows = list(csv.DictReader(open(tmp_path / 'decisions.csv', encoding='utf-8')))
    assert [(r['video_id'], r['decision'], r['stage']) for r in rows] == [
        ('ok', 'accept', 'metadata'), ('short', 'reject', 'metadata'),
        ('live', 'reject', 'metadata'), ('game', 'reject', 'metadata')]
    assert rows[0]['search_query'] == TERM and rows[0]['search_rank'] == '1'
    assert 'too short' in rows[1]['reason'] and 'live' in rows[2]['reason'] and 'gta' in rows[3]['reason']


class _StubVisual:
    backend = 'stub'
    model_name = 'stub-model'
    threshold = 0.5

    def check(self, video):
        from filter_common import FilterResult
        if 'error' in video['name']:
            return FilterResult(False, 'visual', 'download failed', error=True)
        ok = 'bad' not in video['name']
        return FilterResult(ok, 'visual', 'stub', score=1.0 if ok else 0.0,
                            details={'time_of_day': 'night', 'frame_brightness': [20.0]})


def test_visual_filter_integration_and_stats(pedx):
    table = {(TERM, None): {'items': [make_search_item(v) for v in ['q1', 'q2', 'q3']]}}
    details = {'q1': make_details('q1', title='good drive'), 'q2': make_details('q2', title='bad drive'),
               'q3': make_details('q3', title='error drive')}
    d = _discovery(pedx, table, details, visual_filter=_StubVisual())
    res = _search(d, pedx)
    assert [r['id'] for r in res] == ['q1']
    assert d.last_stats['rejected_visual'] == 1 and d.last_stats['errors'] == 1
    # frame-based time of day overrides the (absent) title cue
    assert res[0]['time_of_day'] == 'night' and res[0]['time_of_day_source'] == 'frames'
    assert res[0]['filter_backend'] == 'stub' and res[0]['filter_score'] == 1.0


def test_circuit_breaker_aborts_when_visual_checks_keep_failing(pedx, tmp_path):
    from provenance import DecisionLog

    class Failing:
        backend, model_name, threshold = 'stub', 'm', 0.5

        def check(self, video):
            from filter_common import FilterResult
            return FilterResult(False, 'visual', 'access denied', error=True)

    ids = [f'v{i}' for i in range(8)]
    table = {(TERM, None): {'items': [make_search_item(v) for v in ids]}}
    log = DecisionLog(str(tmp_path / 'd.csv'))
    d = _discovery(pedx, table, {v: make_details(v) for v in ids}, visual_filter=Failing(),
                   decision_log=log, max_consecutive_errors=3)
    with pytest.raises(pedx.VisualFilterBroken):
        _search(d, pedx)
    log.close()
    rows = list(csv.DictReader(open(tmp_path / 'd.csv', encoding='utf-8')))
    assert len(rows) == 3 and all(r['decision'] == 'error' for r in rows)  # stopped at the 3rd


def test_circuit_breaker_resets_on_any_verdict(pedx):
    seq = iter([True, True, False, True, True])   # error, error, reject(ok), error, error

    class Flaky:
        backend, model_name, threshold = 'stub', 'm', 0.5

        def check(self, video):
            from filter_common import FilterResult
            return FilterResult(False, 'visual', 'x', error=next(seq))

    ids = [f'v{i}' for i in range(5)]
    table = {(TERM, None): {'items': [make_search_item(v) for v in ids]}}
    d = _discovery(pedx, table, {v: make_details(v) for v in ids}, visual_filter=Flaky(),
                   max_consecutive_errors=3)
    assert _search(d, pedx) == []  # never 3 errors in a row -> no abort


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

    d = pedx.YouTubeDiscovery(youtube=object())
    assert d._execute_with_retry(FlakyReq(), 'test') == {'ok': True}
    assert calls['n'] == 3  # two failures then success


# --- CSV I/O -----------------------------------------------------------------

def _row(vid, city='London'):
    return {'id': vid, 'name': 'n', 'city': city, 'video': vid, 'video_url': 'u',
            'time_of_day': 'unknown', 'start_time': '0:00', 'end_time': '0:02:30',
            'region_code': 'GB', 'channel_name': 'ch', 'channel_url': 'cu',
            'published_at': '2026-06-01T00:00:00Z', 'country_code': 'GB',
            'duration_seconds': 150, 'view_count': '10', 'like_count': '1',
            'comment_count': '0', 'thumbnail_url': 't', 'latitude': '', 'longitude': '',
            'description': 'not a column'}


def test_save_to_csv_new_file(pedx, tmp_path):
    out = tmp_path / "sub" / "discovery.csv"
    path = pedx.save_to_csv([_row('v1')], str(out))
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['id'] == 'v1'
    assert 'duration_seconds' in rows[0] and 'search_query' in rows[0]
    assert 'description' not in rows[0]  # helper fields are not written


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
