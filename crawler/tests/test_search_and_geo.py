"""Tests for search_profiles.py, gazetteer.py and provenance.py."""
import csv
import json

import pytest

import gazetteer as gz_mod
import provenance
import search_profiles as sp


# --- search profiles ---------------------------------------------------------------

def test_profiles_are_well_formed():
    for profile in sp.PROFILES.values():
        assert profile.terms['en'], profile.name
        assert set(profile.video_durations) <= set(sp.VALID_DURATIONS)
        assert profile.definition in sp.VALID_DEFINITIONS
        for templates in profile.terms.values():
            assert all('{city}' in t for t in templates)


def test_dashcam_defaults_match_curated_dataset():
    p = sp.PROFILES['dashcam']
    assert (p.min_duration, p.max_duration) == (240, 10800)
    assert p.video_durations == ('medium', 'long') and p.definition == 'high'


def test_templates_fall_back_to_base_language():
    p = sp.PROFILES['dashcam']
    assert p.templates('de') == p.terms['de']
    assert p.templates('pt-BR') == p.terms['pt']
    assert p.templates('zh-Hant') == p.terms['zh-Hant']
    assert p.templates('xx') == ()
    assert p.templates(None) == p.terms['en']


@pytest.mark.parametrize("spec,country_langs,expected", [
    ('auto', ['ja'], ['ja', 'en']),
    ('auto', ['en'], ['en']),
    ('auto', [], ['en']),
    ('auto', ['sw'], ['en']),          # no Swahili templates -> English only
    ('none', ['ja'], [None]),
    ('de,en', [], ['de', 'en']),
    ('xx', [], ['en']),
])
def test_resolve_languages(spec, country_langs, expected):
    assert sp.resolve_languages(spec, sp.PROFILES['dashcam'], country_langs) == expected


def test_build_streams_order_and_local_name():
    p = sp.PROFILES['dashcam']
    streams = sp.build_streams(p, 'Moscow', ['ru', 'en'], ['medium', 'long'], local_name='Москва')
    assert [(s.query, s.language, s.duration) for s in streams] == [
        ('Москва на машине', 'ru', 'medium'), ('Москва на машине', 'ru', 'long'),
        ('Moscow driving 4K', 'en', 'medium'), ('Moscow driving 4K', 'en', 'long')]
    all_terms = sp.build_streams(p, 'Moscow', ['ru', 'en'], ['any'], all_terms=True)
    assert len(all_terms) == len(p.terms['ru']) + len(p.terms['en'])
    assert all_terms[0].language == 'ru' and all_terms[1].language == 'en'  # interleaved by rank


def test_crossing_profile_keeps_original_queries():
    streams = sp.build_streams(sp.PROFILES['crossing'], 'London', [None], ['any'], all_terms=True)
    assert streams[0].query == 'London street crossing pedestrian' and len(streams) == 5


# --- gazetteer ------------------------------------------------------------------------

@pytest.fixture(scope='module')
def gz():
    g = gz_mod.Gazetteer()
    if not g.available:
        pytest.skip('geonamescache not installed')
    return g


def test_normalize_name():
    assert gz_mod.normalize_name('  São-Paulo ') == 'sao paulo'
    assert gz_mod.normalize_name('Köln') == 'koln'


@pytest.mark.parametrize("tag,code", [
    ('de-DE', 'de'), ('ja', 'ja'), ('zh-TW', 'zh-Hant'), ('zh-CN', 'zh-Hans'), ('zh', 'zh-Hans'),
    ('fil', None), ('', None),
])
def test_to_relevance_language(tag, code):
    assert gz_mod.to_relevance_language(tag) == code


def test_resolve_prefers_population_and_respects_country(gz):
    london = gz.resolve_city('London')
    assert london.country_code == 'GB' and london.ambiguous is False
    assert gz.resolve_city('London', 'CA').country_code == 'CA'
    assert abs(london.latitude - 51.5) < 0.2


def test_resolve_alternate_name_and_accents(gz):
    nyc = gz.resolve_city('New York', 'US')
    assert nyc.match == 'alternate' and abs(nyc.latitude - 40.7) < 0.2
    assert gz.resolve_city('sao paulo', 'BR').name == 'São Paulo'


def test_primary_name_beats_bigger_city_nickname(gz):
    """'Crescent City' is an alternate name of New Orleans; the real (small) Crescent
    City, CA must win."""
    cc = gz.resolve_city('Crescent City', 'US')
    assert cc.match == 'name' and cc.longitude < -120


def test_ambiguous_flag(gz):
    assert gz.resolve_city('Portland', 'US').ambiguous is True


def test_admin_and_explicit_geonames_disambiguate_city(gz):
    oregon = gz.resolve_city('Portland', 'US', 'OR')
    maine = gz.resolve_city('Portland', 'US', 'ME')
    assert oregon.geonameid != maine.geonameid
    assert oregon.admin1_code == 'OR' and maine.admin1_code == 'ME'
    assert gz.resolve_city('Portland', 'US', 'ME', maine.geonameid).geonameid == maine.geonameid
    assert gz.resolve_city('Portland', 'US', 'OR', maine.geonameid) is None


def test_unknown_city(gz):
    assert gz.resolve_city('Nowhereville Xyzzy', 'GB') is None
    assert gz.resolve_city('', 'GB') is None


def test_country_info(gz):
    jp = gz.country('jp')
    assert (jp.iso3, jp.continent, jp.primary_language) == ('JPN', 'Asia', 'ja')
    assert gz.country('TW').primary_language == 'zh-Hant'
    assert gz.country('ZZ') is None and gz.country(None) is None


# --- provenance ----------------------------------------------------------------------

def test_decision_log_appends_with_single_header(tmp_path):
    path = tmp_path / 'runs' / 'r1' / 'decisions.csv'
    log = provenance.DecisionLog(str(path))
    log.write({'video_id': 'a', 'decision': 'accept', 'frame_scores': [1.0, 0.0]})
    log.close()
    log = provenance.DecisionLog(str(path))
    log.write({'video_id': 'b', 'decision': 'reject'})
    log.close()
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    assert [r['video_id'] for r in rows] == ['a', 'b']
    assert json.loads(rows[0]['frame_scores']) == [1.0, 0.0]
    assert rows[0]['timestamp_utc'].endswith('Z')
    with open(path, newline='', encoding='utf-8') as f:
        assert sum(row[0] == 'run_id' for row in csv.reader(f)) == 1


def test_manifest_and_environment(tmp_path):
    path = tmp_path / 'm' / 'manifest.json'
    env = provenance.environment_info(str(tmp_path), ['pytest', 'definitely-not-installed'])
    provenance.write_manifest(str(path), {'run_id': 'x', 'environment': env})
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['environment']['packages']['pytest']
    assert data['environment']['packages']['definitely-not-installed'] is None
    assert not (tmp_path / 'm' / 'manifest.json.tmp').exists()


def test_run_id_format():
    rid = provenance.new_run_id()
    assert len(rid) == len('20261010T120000Z-abcd') and rid[8] == 'T'
