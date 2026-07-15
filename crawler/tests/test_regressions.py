"""Regression tests for issues found in adversarial review of the QoL rewrite."""
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


def _row(vid):
    return {'id': vid, 'name': 'n', 'city': 'London', 'video': vid, 'video_url': 'u',
            'time_of_day': 'unknown', 'start_time': '0:00', 'end_time': '0:02:30',
            'region_code': 'GB', 'channel_name': 'ch', 'channel_url': 'cu',
            'published_at': '', 'country_code': 'GB', 'duration_seconds': 150,
            'view_count': '', 'like_count': '', 'comment_count': '', 'thumbnail_url': '',
            'latitude': '', 'longitude': ''}


def test_missing_details_leaves_videos_unseen(pedx):
    """A failed/empty details batch must NOT burn IDs in the shared seen set."""
    table = {(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')]}}
    d = _discovery(pedx, table, {})  # no details for anyone
    seen = set()
    res = d.search_videos('London', max_results=10, use_single_search=True, seen_ids=seen)
    assert res == []
    assert seen == set()  # nothing burned -> discoverable on a later run


def test_surplus_page_ids_not_marked_seen(pedx):
    """IDs past the per-city cap on a page must stay discoverable for later cities."""
    table = {(TERM, None): {'items': [make_search_item(f'v{i}') for i in range(1, 6)]}}
    details = {f'v{i}': make_details(f'v{i}') for i in range(1, 6)}
    d = _discovery(pedx, table, details)
    seen = set()
    res = d.search_videos('London', max_results=2, use_single_search=True, seen_ids=seen)
    assert [r['id'] for r in res] == ['v1', 'v2']
    assert seen == {'v1', 'v2'}  # v3..v5 NOT burned


def test_quota_exceeded_in_details_propagates(pedx):
    """quotaExceeded from the details call must stop the run, not be swallowed."""
    class _QuotaVideos:
        def list(self, part, id):
            class _R:
                def execute(self_inner):
                    raise pedx.HttpError("quotaExceeded")
            return _R()

    d = _discovery(pedx, {(TERM, None): {'items': [make_search_item('v1')]}}, {})
    d.youtube._videos = _QuotaVideos()
    with pytest.raises(pedx.HttpError):
        d.search_videos('London', max_results=5, use_single_search=True)


def test_append_incompatible_header_falls_back(pedx, tmp_path):
    out = str(tmp_path / "old.csv")
    with open(out, 'w', newline='', encoding='utf-8') as f:
        f.write("id,name,city\nx,old,London\n")  # narrow legacy schema
    path = pedx.save_to_csv([_row('v1')], out, append=True)
    assert path != out                                   # diverted to a new file
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['id'] == 'v1' and 'duration_seconds' in rows[0]
    assert len(open(out, encoding='utf-8').read().splitlines()) == 2  # original untouched


def test_append_empty_file_writes_header(pedx, tmp_path):
    out = str(tmp_path / "empty.csv")
    open(out, 'w').close()  # pre-existing 0-byte file
    pedx.save_to_csv([_row('v1')], out, append=True)
    content = open(out, encoding='utf-8').read()
    assert content.startswith('id,name,city')  # header written despite file existing
    assert 'v1' in content


def test_load_seen_ids_handles_bom(pedx, tmp_path):
    out = tmp_path / "bom.csv"
    out.write_text('id,name\nv1,foo\nv2,bar\n', encoding='utf-8-sig')  # writes a BOM
    assert pedx.load_seen_ids([str(out)]) == {'v1', 'v2'}
