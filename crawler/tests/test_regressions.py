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
        # Not a prefix of CSV_FIELDNAMES (column order differs), so it cannot be
        # migrated and must be diverted instead.
        f.write("id,city,name\nx,London,old\n")
    path = pedx.save_to_csv([_row('v1')], out, append=True)
    assert path != out                                   # diverted to a new file
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['id'] == 'v1' and 'duration_seconds' in rows[0]
    assert len(open(out, encoding='utf-8').read().splitlines()) == 2  # original untouched


@pytest.mark.parametrize("n_cols", [11, 13])
def test_append_upgrades_prefix_schema_in_place(pedx, tmp_path, n_cols):
    """Older discovery CSVs have a header that is a strict PREFIX of the current one.
    Those must be widened in place and appended to — not diverted to a new file, which
    left the target never growing and the diverted IDs outside the next run's dedup set.
    """
    out = str(tmp_path / "legacy.csv")
    old_fields = pedx.CSV_FIELDNAMES[:n_cols]
    legacy = {k: f"old-{k}" for k in old_fields}
    legacy['id'] = 'old1'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow(legacy)

    path = pedx.save_to_csv([_row('v1')], out, append=True)

    assert path == out                                   # same file, no fallback
    assert list(tmp_path.glob('legacy_*.csv')) == []      # nothing diverted
    assert list(tmp_path.glob('*.upgrading')) == []       # temp file cleaned up
    with open(out, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        assert next(reader) == pedx.CSV_FIELDNAMES       # full header
        widths = {len(r) for r in reader}
    assert widths == {len(pedx.CSV_FIELDNAMES)}          # no ragged rows
    rows = _read_rows(out)
    assert [r['id'] for r in rows] == ['old1', 'v1']
    # migrated row keeps its old values and gets '' for the new columns
    assert all(rows[0][k] == legacy[k] for k in old_fields)
    assert all(rows[0][k] == '' for k in pedx.CSV_FIELDNAMES[n_cols:])
    assert rows[1]['duration_seconds'] == '150'          # appended row is complete


def test_upgrade_then_append_is_idempotent(pedx, tmp_path):
    """A second append must not re-upgrade or duplicate the header."""
    out = str(tmp_path / "legacy.csv")
    old_fields = pedx.CSV_FIELDNAMES[:11]
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow({k: '' for k in old_fields} | {'id': 'old1'})
    pedx.save_to_csv([_row('v1')], out, append=True)
    pedx.save_to_csv([_row('v2')], out, append=True)
    content = open(out, encoding='utf-8').read()
    assert content.count('id,name,city') == 1
    assert [r['id'] for r in _read_rows(out)] == ['old1', 'v1', 'v2']


def _read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


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
