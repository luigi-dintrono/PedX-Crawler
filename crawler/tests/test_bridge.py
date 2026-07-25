"""Tests for scripts/discovery_to_mapping.py — the Crawler -> PedX-Insight seam.

This converter writes into PedX-Insight's real mapping.csv / mapping_one_each.csv, so
the properties that matter most are (a) the naming convention run.py depends on and
(b) that --append never corrupts rows it did not add.
"""
import csv
import re

import pytest
from conftest import DISCOVERY_FIELDS, discovery_row, write_discovery_csv


def _run(bridge, monkeypatch, tmp_path, discovery, append=False,
         one_each='oe.csv', mapping='map.csv'):
    argv = ['prog', '--discovery', str(discovery),
            '--out-mapping-one-each', str(tmp_path / one_each),
            '--out-mapping', str(tmp_path / mapping)]
    if append:
        argv.append('--append')
    monkeypatch.setattr(bridge.sys, 'argv', argv)
    bridge.main()
    return _read(tmp_path / one_each), _read(tmp_path / mapping)


def _read(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


# --- pure helpers ------------------------------------------------------------

@pytest.mark.parametrize("hms,secs", [
    ('1:02:03', 3723), ('2:30', 150), ('45', 45), ('0:00:06', 6),
    ('', None), ('bad', None), (None, None),
])
def test_parse_duration_seconds(bridge, hms, secs):
    assert bridge.parse_duration_seconds(hms) == secs


@pytest.mark.parametrize("value,flag", [
    ('night', 1), ('evening', 1), ('morning', 0), ('afternoon', 0),
    ('unknown', 0), ('', 0), ('NIGHT', 1),
])
def test_time_of_day_to_flag(bridge, value, flag):
    assert bridge.time_of_day_to_flag(value) == flag


@pytest.mark.parametrize("published,expected", [
    ('2023-08-29T10:00:00Z', 20230829), ('2024-04-03T00:00:00Z', 20240403),
    ('', 0), ('garbage', 0), (None, 0),
])
def test_published_to_upload_date(bridge, published, expected):
    """YYYYMMDD, the unambiguous form (see the module docstring on DDMMYYYY)."""
    assert bridge.published_to_upload_date(published) == expected


@pytest.mark.parametrize("city,slug", [
    ('New York', 'NewYork'), ('São Paulo', 'SãoPaulo'), ('Ulm', 'Ulm'),
    ('Some_City', 'SomeCity'), ('  Spaced  Out  ', 'SpacedOut'),
])
def test_clean_city_slug_has_no_underscore_or_space(bridge, city, slug):
    assert bridge.clean_city_slug(city) == slug
    assert '_' not in bridge.clean_city_slug(city)


def test_parse_nested_preserves_multi_element_inner_lists(bridge):
    """Regression guard: a strip('[]')+split(',') approach shreds '[0,0,0]' into
    three items, silently desynchronising the column from 'videos'."""
    assert bridge.parse_nested('[[0],[1]]') == ['[0]', '[1]']
    assert bridge.parse_nested('[[6177,7514,7737],[41,1990],[0]]') == \
        ['[6177,7514,7737]', '[41,1990]', '[0]']
    # inner content is re-joined verbatim, so spaced input keeps its spacing
    assert bridge.parse_nested('[[0], [12, 30]]') == ['[0]', '[12, 30]']
    assert bridge.parse_nested('') == []
    assert bridge.parse_nested('[]') == []


def test_parse_bracketed(bridge):
    assert bridge.parse_bracketed('[a,b]') == ['a', 'b']
    assert bridge.parse_bracketed("['a', \"b\"]") == ['a', 'b']
    assert bridge.parse_bracketed('') == []
    assert bridge.parse_bracketed(None) == []


def test_conftest_schema_matches_crawler(pedx):
    """The fixtures' discovery schema must not drift from the crawler's."""
    assert DISCOVERY_FIELDS == pedx.CSV_FIELDNAMES


# --- conversion --------------------------------------------------------------

def test_basic_conversion(bridge, monkeypatch, tmp_path):
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [
        discovery_row('aaa_111', 'New York', 'US', time_of_day='night', end_time='0:02:30'),
        discovery_row('bbb222', 'New York', 'US', time_of_day='morning', end_time='0:01:05'),
        discovery_row('ccc333', 'Ulm', 'DE', end_time='1:02:03'),
    ])
    oe, mp = _run(bridge, monkeypatch, tmp_path, disc)

    # mapping_one_each: sequential ids, underscore-free names, seconds durations
    assert [r['name'] for r in oe] == ['NewYork1', 'NewYork2', 'Ulm1']
    assert [r['id'] for r in oe] == ['1', '2', '3']
    assert [r['end_time'] for r in oe] == ['150', '65', '3723']
    assert [r['time_of_day'] for r in oe] == ['1', '0', '0']
    assert all('_' not in r['name'] for r in oe)
    # a video id containing '_' must survive intact while its name stays underscore-free
    assert oe[0]['video'] == 'aaa_111'

    # mapping: one row per city, ISO2 expanded
    assert [r['city'] for r in mp] == ['New York', 'Ulm']
    assert [r['country'] for r in mp] == ['United States', 'Germany']
    assert [r['id'] for r in mp] == ['1', '2']


def test_mapping_list_shapes(bridge, monkeypatch, tmp_path):
    """videos/upload_date/channel are FLAT; time_of_day/start_time/end_time NESTED."""
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [
        discovery_row('v1', 'Ulm', 'DE', time_of_day='night', end_time='0:00:30',
                      published_at='2023-08-29T10:00:00Z', channel_name='ChanA'),
        discovery_row('v2', 'Ulm', 'DE', time_of_day='morning', end_time='0:01:00',
                      published_at='2024-04-03T10:00:00Z', channel_name='ChanB'),
    ])
    _, mp = _run(bridge, monkeypatch, tmp_path, disc)
    row = mp[0]
    assert row['videos'] == '[v1,v2]'
    assert row['time_of_day'] == '[[1],[0]]'
    assert row['start_time'] == '[[0],[0]]'
    assert row['end_time'] == '[[30],[60]]'
    assert row['upload_date'] == '[20230829,20240403]'
    assert row['channel'] == '[ChanA,ChanB]'


def test_unknown_country_left_blank_and_unmapped_code_passes_through(bridge, monkeypatch, tmp_path):
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('v1', 'Nowhere', 'UNKNOWN'),
                              discovery_row('v2', 'Elsewhere', 'ZZ')])
    _, mp = _run(bridge, monkeypatch, tmp_path, disc)
    by_city = {r['city']: r for r in mp}
    assert by_city['Nowhere']['country'] == ''
    assert by_city['Elsewhere']['country'] == 'ZZ'


def test_bad_rows_skipped_not_fatal(bridge, monkeypatch, tmp_path, capsys):
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [
        discovery_row('good1', 'Ulm', 'DE'),
        {'id': '', 'video': '', 'city': 'Ulm'},        # no video id
        {'id': 'x1', 'video': 'x1', 'city': ''},       # no city
    ])
    oe, _ = _run(bridge, monkeypatch, tmp_path, disc)
    assert [r['video'] for r in oe] == ['good1']
    assert '2 bad rows' in capsys.readouterr().out


def test_duplicate_ids_within_one_run_deduped(bridge, monkeypatch, tmp_path):
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('v1', 'Ulm', 'DE'),
                              discovery_row('v1', 'Ulm', 'DE'),
                              discovery_row('v2', 'Ulm', 'DE')])
    oe, mp = _run(bridge, monkeypatch, tmp_path, disc)
    assert [r['video'] for r in oe] == ['v1', 'v2']
    assert mp[0]['videos'] == '[v1,v2]'


def test_unparseable_duration_blank_in_one_each_zero_in_mapping(bridge, monkeypatch, tmp_path):
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('v1', 'Ulm', 'DE', end_time='n/a')])
    oe, mp = _run(bridge, monkeypatch, tmp_path, disc)
    assert oe[0]['end_time'] == ''        # unknown stays blank for run.py
    assert mp[0]['end_time'] == '[[0]]'   # but the aligned list keeps its slot


# --- --append ----------------------------------------------------------------

def test_append_is_idempotent(bridge, monkeypatch, tmp_path):
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('v1', 'Ulm', 'DE'),
                               discovery_row('v2', 'Ulm', 'DE')])
    first_oe, first_mp = _run(bridge, monkeypatch, tmp_path, disc, append=True)
    again_oe, again_mp = _run(bridge, monkeypatch, tmp_path, disc, append=True)
    assert first_oe == again_oe
    assert first_mp == again_mp


def test_append_adds_new_videos_to_existing_city_row(bridge, monkeypatch, tmp_path):
    disc1, disc2 = tmp_path / 'd1.csv', tmp_path / 'd2.csv'
    write_discovery_csv(disc1, [discovery_row('v1', 'Ulm', 'DE', time_of_day='night')])
    write_discovery_csv(disc2, [discovery_row('v2', 'Ulm', 'DE', time_of_day='morning')])
    _run(bridge, monkeypatch, tmp_path, disc1, append=True)
    oe, mp = _run(bridge, monkeypatch, tmp_path, disc2, append=True)
    assert len(mp) == 1                       # same city -> same row, not a new one
    assert mp[0]['videos'] == '[v1,v2]'
    assert mp[0]['time_of_day'] == '[[1],[0]]'
    assert [r['name'] for r in oe] == ['Ulm1', 'Ulm2']   # counter continues
    assert [r['id'] for r in oe] == ['1', '2']


def test_append_preserves_untouched_rows_and_nested_lists_verbatim(bridge, monkeypatch, tmp_path):
    """The critical safety property: appending must not rewrite other cities' rows,
    and must round-trip multi-element inner lists without shredding them."""
    mapping = tmp_path / 'map.csv'
    keep = {c: '' for c in bridge.MAPPING_HEADER}
    keep.update(id='7', city='Toronto', country='Canada', lat='43.65', population_city='2600000',
                videos='[aaa,bbb,ccc]',
                time_of_day='[[0],[0,0,0],[1]]',
                start_time='[[27],[6177,7514,7737],[0]]',
                end_time='[[3250],[7045,7589,9275],[600]]',
                vehicle_type='[0,0,0]', upload_date='[29082023,3042024,13042014]',
                fps_list='[30,30,30]', channel='[UCa,UCb,UCc]')
    with open(mapping, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=bridge.MAPPING_HEADER)
        w.writeheader()
        w.writerow(keep)
    before = _read(mapping)[0]

    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('newvid', 'Ulm', 'DE')])
    _, mp = _run(bridge, monkeypatch, tmp_path, disc, append=True)

    assert [r['city'] for r in mp] == ['Toronto', 'Ulm']
    assert mp[0] == before          # untouched row byte-for-byte identical
    assert mp[1]['videos'] == '[newvid]'


def test_append_keeps_aligned_columns_in_step(bridge, monkeypatch, tmp_path):
    """vehicle_type/fps_list are aligned with 'videos': extend them when populated,
    leave them blank when they were blank."""
    mapping = tmp_path / 'map.csv'
    rows = []
    populated = {c: '' for c in bridge.MAPPING_HEADER}
    populated.update(id='1', city='Toronto', videos='[a,b]', time_of_day='[[0],[0]]',
                     start_time='[[0],[0]]', end_time='[[10],[20]]', vehicle_type='[0,0]',
                     upload_date='[20200101,20200102]', fps_list='[30,30]', channel='[x,y]')
    blank = {c: '' for c in bridge.MAPPING_HEADER}
    blank.update(id='2', city='Ulm', videos='[c]', time_of_day='[[0]]', start_time='[[0]]',
                 end_time='[[5]]', upload_date='[20200103]', channel='[z]')
    rows += [populated, blank]
    with open(mapping, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=bridge.MAPPING_HEADER)
        w.writeheader()
        w.writerows(rows)

    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('newT', 'Toronto', 'CA'),
                               discovery_row('newU', 'Ulm', 'DE')])
    _, mp = _run(bridge, monkeypatch, tmp_path, disc, append=True)

    def n_flat(s):
        return len([x for x in s.strip('[]').split(',') if x.strip()]) if s.strip('[]') else 0

    tor, ulm = mp[0], mp[1]
    assert tor['videos'] == '[a,b,newT]'
    assert n_flat(tor['vehicle_type']) == n_flat(tor['videos']) == 3
    assert n_flat(tor['fps_list']) == 3
    assert tor['fps_list'] == '[30,30,0]'          # 0 = unknown, not a fabricated 30
    assert ulm['videos'] == '[c,newU]'
    assert ulm['vehicle_type'] == '' and ulm['fps_list'] == ''   # stayed blank


def test_append_assigns_ids_to_rows_missing_them(bridge, monkeypatch, tmp_path):
    mapping = tmp_path / 'map.csv'
    row = {c: '' for c in bridge.MAPPING_HEADER}
    row.update(city='Toronto', videos='[a]')     # no id
    with open(mapping, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=bridge.MAPPING_HEADER)
        w.writeheader()
        w.writerow(row)
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('v1', 'Ulm', 'DE')])
    _, mp = _run(bridge, monkeypatch, tmp_path, disc, append=True)
    assert all(str(r['id']).isdigit() for r in mp)
    assert len({r['id'] for r in mp}) == len(mp)   # unique


def test_missing_discovery_file_exits(bridge, monkeypatch, tmp_path):
    monkeypatch.setattr(bridge.sys, 'argv',
                        ['prog', '--discovery', str(tmp_path / 'nope.csv'),
                         '--out-mapping-one-each', str(tmp_path / 'oe.csv'),
                         '--out-mapping', str(tmp_path / 'map.csv')])
    with pytest.raises(SystemExit):
        bridge.main()


def test_one_each_names_split_at_first_underscore(bridge, monkeypatch, tmp_path):
    """run.py builds '<name>_<video_id>' and everything downstream splits at the FIRST
    underscore, so name must contribute none even for underscore-bearing video ids."""
    disc = tmp_path / 'disc.csv'
    write_discovery_csv(disc, [discovery_row('_e0RQL7NJZg', 'New York', 'US'),
                               discovery_row('50Uf_T12OGY', 'New York', 'US')])
    oe, _ = _run(bridge, monkeypatch, tmp_path, disc)
    for row in oe:
        folder = f"{row['name']}_{row['video']}"
        assert folder.split('_', 1)[0] == row['name']
        assert folder.split('_', 1)[1] == row['video']
