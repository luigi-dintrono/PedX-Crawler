"""Failures at the Crawler/Insight seam must be repairable without data loss."""

import csv
import json

import pytest

from conftest import FakeYouTube, discovery_row, make_details, make_search_item, write_discovery_csv
from contracts import city_identity
from filter_common import FilterResult
from provenance import redact_argv


def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def bridge_run(bridge, monkeypatch, root, discovery):
    monkeypatch.setattr(bridge.sys, 'argv', ['bridge', '--append', '--discovery', str(discovery),
                        '--out-mapping-one-each', str(root / 'queue.csv'),
                        '--out-mapping', str(root / 'mapping.csv')])
    bridge.main()
    return read(root / 'queue.csv'), read(root / 'mapping.csv')


def test_fallback_identity_is_country_and_admin_aware():
    assert city_identity(' São-Paulo ', 'BR') == 'place:br:unknown:sao-paulo'
    assert city_identity('London', 'GB') != city_identity('London', 'CA')
    assert city_identity('Portland', 'US', 'OR') != city_identity('Portland', 'US', 'ME')
    assert city_identity('any', 'XX', geonameid='2988507') == 'geonames:2988507'
    assert city_identity('any', explicit_id='my-canonical-id') == 'my-canonical-id'


def test_bridge_keeps_country_admin_identity_and_unknown_recording_date(bridge, monkeypatch, tmp_path):
    rows = [discovery_row('gb', 'London', 'GB'), discovery_row('ca', 'London', 'CA'),
            discovery_row('or', 'Portland', 'US'), discovery_row('me', 'Portland', 'US')]
    rows[2]['admin1_code'], rows[3]['admin1_code'] = 'OR', 'ME'
    rows[0].update(run_id='run1', discovered_at='2026-10-27T12:00:00Z', start_time='2:00',
                   end_time='3:00', search_query='London driving')
    discovery = tmp_path / 'discovery.csv'
    write_discovery_csv(discovery, rows)
    queue, mapping = bridge_run(bridge, monkeypatch, tmp_path, discovery)
    assert len(mapping) == 4 and len({r['city_id'] for r in mapping}) == 4
    assert [r['name'] for r in queue] == ['London1', 'London2', 'Portland1', 'Portland2']
    assert queue[0]['start_time'] == '120' and queue[0]['end_time'] == '180'
    assert queue[0]['discovery_run_id'] == 'run1'
    assert queue[0]['recording_date'] == '' and queue[0]['recording_date_source'] == 'unknown'
    assert all(r['eligibility_status'] == 'unverified' and not r['verified_city_id'] for r in queue)
    evidence = json.loads(mapping[0]['video_provenance'])['gb']
    assert evidence['search_query'] == 'London driving'
    assert evidence['discovered_at'] != evidence['published_at']
    assert evidence['clip_start_seconds'] == 120


def test_bridge_retry_repairs_half_replaced_outputs(bridge, monkeypatch, tmp_path):
    discovery = tmp_path / 'discovery.csv'
    write_discovery_csv(discovery, [discovery_row('v1', 'Ulm', 'DE')])
    original = bridge.write_csv_atomic

    def fail_mapping(path, header, rows):
        if path == str(tmp_path / 'mapping.csv'):
            raise OSError('simulated second-file failure')
        original(path, header, rows)

    monkeypatch.setattr(bridge, 'write_csv_atomic', fail_mapping)
    with pytest.raises(OSError):
        bridge_run(bridge, monkeypatch, tmp_path, discovery)
    assert [r['video'] for r in read(tmp_path / 'queue.csv')] == ['v1']
    monkeypatch.setattr(bridge, 'write_csv_atomic', original)
    queue, mapping = bridge_run(bridge, monkeypatch, tmp_path, discovery)
    assert len(queue) == len(mapping) == 1 and mapping[0]['videos'] == '[v1]'
    assert bridge_run(bridge, monkeypatch, tmp_path, discovery) == (queue, mapping)


def test_bridge_repairs_prior_queue_with_a_different_discovery_input(bridge, monkeypatch, tmp_path):
    first, second = tmp_path / 'first.csv', tmp_path / 'second.csv'
    write_discovery_csv(first, [discovery_row('old', 'Ulm', 'DE')])
    write_discovery_csv(second, [discovery_row('new', 'London', 'GB')])
    original = bridge.write_csv_atomic

    def fail_mapping(path, header, rows):
        if path == str(tmp_path / 'mapping.csv'):
            raise OSError('interrupted after queue replacement')
        original(path, header, rows)

    monkeypatch.setattr(bridge, 'write_csv_atomic', fail_mapping)
    with pytest.raises(OSError):
        bridge_run(bridge, monkeypatch, tmp_path, first)
    monkeypatch.setattr(bridge, 'write_csv_atomic', original)
    queue, mapping = bridge_run(bridge, monkeypatch, tmp_path, second)
    assert {r['video'] for r in queue} == {'old', 'new'}
    assert {vid for row in mapping for vid in bridge.parse_bracketed(row['videos'])} == {'old', 'new'}


def test_append_preserves_existing_verification_window_and_extension(bridge, monkeypatch, tmp_path):
    discovery = tmp_path / 'discovery.csv'
    write_discovery_csv(discovery, [discovery_row('v1', 'Ulm', 'DE')])
    queue, _ = bridge_run(bridge, monkeypatch, tmp_path, discovery)
    queue[0].update(start_time='12', end_time='34', finished='TRUE',
                    clip_start_seconds='12', clip_end_seconds='34',
                    eligibility_status='verified', verified_city_id='geonames:2820256',
                    location_verification_source='human:review42', operator_note='retain me')
    with open(tmp_path / 'queue.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    after, _ = bridge_run(bridge, monkeypatch, tmp_path, discovery)
    for key in ('name', 'start_time', 'end_time', 'finished', 'verified_city_id',
                'eligibility_status', 'location_verification_source', 'operator_note'):
        assert after[0][key] == queue[0][key]


def test_explicit_review_enriches_unverified_queue_without_automatic_reversal(bridge, monkeypatch, tmp_path):
    discovery = tmp_path / 'discovery.csv'
    automatic = discovery_row('v1', 'Ulm', 'DE')
    write_discovery_csv(discovery, [automatic])
    bridge_run(bridge, monkeypatch, tmp_path, discovery)
    reviewed = {**automatic, 'verified_city_id': 'geonames:2820256',
                'eligibility_status': 'verified', 'location_verification_source': 'human:review42'}
    write_discovery_csv(discovery, [reviewed])
    queue, mapping = bridge_run(bridge, monkeypatch, tmp_path, discovery)
    assert queue[0]['eligibility_status'] == 'verified'
    assert json.loads(mapping[0]['video_provenance'])['v1']['verified_city_id'] == 'geonames:2820256'
    write_discovery_csv(discovery, [automatic])
    queue, mapping = bridge_run(bridge, monkeypatch, tmp_path, discovery)
    assert queue[0]['location_verification_source'] == 'human:review42'
    assert json.loads(mapping[0]['video_provenance'])['v1']['eligibility_status'] == 'verified'


@pytest.mark.parametrize('stop', ['interrupt', 'breaker', 'quota'])
def test_accepted_rows_survive_mid_city_abort(pedx, monkeypatch, tmp_path, stop):
    ids = ['first', 'second']
    fake = FakeYouTube({('London driving 4K', None): {'items': [make_search_item(v) for v in ids]}},
                       {v: make_details(v) for v in ids})

    class Visual:
        backend, model_name, threshold = 'mock', 'mock-v1', 0.5

        def check(self, video):
            if video['id'] == 'second':
                if stop == 'interrupt':
                    raise KeyboardInterrupt()
                if stop == 'quota':
                    raise pedx.HttpError('quotaExceeded')
                return FilterResult(False, 'visual', 'network unavailable', error=True)
            return FilterResult(True, 'visual', 'street frame', score=1.0)

    monkeypatch.setattr(pedx, 'build', lambda *a, **k: fake)
    monkeypatch.setattr(pedx, 'create_quality_filter', lambda *a, **k: Visual())
    monkeypatch.setattr(pedx, 'environment_info', lambda *a: {})
    cities = tmp_path / 'cities.txt'
    cities.write_text('London,GB\n', encoding='utf-8')
    monkeypatch.setattr(pedx.sys, 'argv', ['crawler', '--api-key=DO-NOT-STORE', '--filter', 'yolo',
                        '--cities-file', str(cities), '--output', str(tmp_path / 'out.csv'),
                        '--runs-dir', str(tmp_path / 'runs'), '--max-consecutive-errors', '1'])
    with pytest.raises(SystemExit) as exc:
        pedx.main()
    assert exc.value.code == (130 if stop == 'interrupt' else 1)
    result = read(tmp_path / 'out.csv')
    assert [r['id'] for r in result] == ['first']
    assert result[0]['discovered_at'] and result[0]['clip_end_seconds'] == '1500'
    manifest = json.loads(next((tmp_path / 'runs').glob('*/manifest.json')).read_text(encoding='utf-8'))
    assert manifest['cities'][0]['kept'] == 1
    assert 'DO-NOT-STORE' not in json.dumps(manifest)


def test_credential_redaction_covers_both_cli_forms():
    assert redact_argv(['--api-key', 'secret1', '--token=secret2', '--ytdlp-cookies', 'secret3',
                        '--profile', 'dashcam']) == [
        '--api-key', '[REDACTED]', '--token=[REDACTED]', '--ytdlp-cookies', '[REDACTED]',
        '--profile', 'dashcam']
