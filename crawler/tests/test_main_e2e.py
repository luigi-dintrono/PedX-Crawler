"""End-to-end tests for main(): full crawl flow with a mocked YouTube client,
covering country-code assignment, regionCode, enrichment, always-save, and
--append cross-run dedup."""
import csv

from conftest import FakeYouTube, make_search_item, make_details

TERM = "London street crossing pedestrian"


def _run_main(pedx, monkeypatch, fake, argv):
    monkeypatch.setattr(pedx, 'build', lambda *a, **k: fake)
    monkeypatch.setattr(pedx.sys, 'argv', argv)
    pedx.main()


def _read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_main_end_to_end(pedx, tmp_path, monkeypatch):
    cities = tmp_path / "cities.txt"
    cities.write_text("London,GB\n", encoding='utf-8')
    out = tmp_path / "out.csv"
    fake = FakeYouTube({(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')]}},
                       {v: make_details(v, lat=51.5, lon=-0.12) for v in ['v1', 'v2']})

    _run_main(pedx, monkeypatch, fake,
              ['prog', '--api-key', 'x', '--cities-file', str(cities),
               '--output', str(out), '--per-city', '10'])

    rows = _read(out)
    assert [r['id'] for r in rows] == ['v1', 'v2']
    assert rows[0]['country_code'] == 'GB'                 # from "London,GB" in cities.txt
    assert rows[0]['latitude'] == '51.5'                   # enrichment column populated
    assert fake._search.last_params.get('regionCode') == 'GB'  # CC drives search localization


def test_main_append_is_idempotent(pedx, tmp_path, monkeypatch):
    cities = tmp_path / "cities.txt"
    cities.write_text("London,GB\n", encoding='utf-8')
    out = tmp_path / "out.csv"
    table = {(TERM, None): {'items': [make_search_item('v1'), make_search_item('v2')]}}
    details = {v: make_details(v) for v in ['v1', 'v2']}
    argv = ['prog', '--api-key', 'x', '--cities-file', str(cities),
            '--output', str(out), '--per-city', '10', '--append']

    _run_main(pedx, monkeypatch, FakeYouTube(table, details), argv)
    _run_main(pedx, monkeypatch, FakeYouTube(table, details), argv)  # second run: all excluded

    rows = _read(out)
    assert [r['id'] for r in rows] == ['v1', 'v2']  # no duplicates from the second run
    assert open(out, encoding='utf-8').read().count('id,name,city') == 1  # header once
