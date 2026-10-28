"""Additive identity and evidence fields shared by discovery and the Insight bridge.

City identifiers describe the search location, never proof of a video's location.
Explicit IDs are authoritative. Fallback keys use NFKD, remove combining marks,
casefold, replace non-alphanumeric runs by '-', and include country and admin1.
"""

import unicodedata


IDENTITY_FIELDS = [
    'city_id', 'searched_city_id', 'verified_city_id', 'eligibility_status',
    'location_verification_source', 'admin1_code', 'geonameid', 'city_identity_source',
]
EVIDENCE_FIELDS = [
    'discovery_run_id', 'discovered_at', 'recording_date', 'recording_date_source',
    'clip_start_seconds', 'clip_end_seconds', 'sampling_strategy', 'search_order',
]
CONTRACT_FIELDS = IDENTITY_FIELDS + EVIDENCE_FIELDS


def identity_token(value):
    """Portable Unicode normalization; do not transliterate non-Latin names."""
    text = unicodedata.normalize('NFKD', str(value or ''))
    text = ''.join(c for c in text if not unicodedata.combining(c)).casefold()
    return '-'.join(''.join(c if c.isalnum() else ' ' for c in text).split()) or 'unknown'


def city_identity(city, country_code='', admin1_code='', geonameid=None, explicit_id=''):
    if str(explicit_id or '').strip():
        return str(explicit_id).strip()
    raw_geo = str(geonameid or '').strip()
    if raw_geo.isdigit() and int(raw_geo) > 0:
        return f'geonames:{int(raw_geo)}'
    cc = str(country_code or '').strip()
    if cc.upper() == 'UNKNOWN':
        cc = ''
    return f'place:{identity_token(cc)}:{identity_token(admin1_code)}:{identity_token(city)}'


def contract_values(row, city=None, country_code=None, admin1_code=None, geonameid=None):
    """Preserve supplied evidence, with conservative defaults for legacy inputs."""
    cc = country_code if country_code is not None else row.get('country_code', '')
    admin = admin1_code if admin1_code is not None else row.get('admin1_code', '')
    geo = geonameid if geonameid is not None else row.get('geonameid', '')
    explicit = row.get('searched_city_id') or row.get('city_id')
    searched = city_identity(city or row.get('city', ''), cc, admin, geo, explicit)
    values = {key: row.get(key, '') or '' for key in CONTRACT_FIELDS}
    values.update(city_id=searched, searched_city_id=searched, admin1_code=admin or '',
                  geonameid=str(geo) if geo else '',
                  city_identity_source=row.get('city_identity_source') or
                  ('geonames' if searched.startswith('geonames:') else 'country_admin_name'),
                  eligibility_status=row.get('eligibility_status') or 'unverified',
                  recording_date_source=row.get('recording_date_source') or 'unknown',
                  discovery_run_id=row.get('discovery_run_id') or row.get('run_id') or '')
    values['location_verification_source'] = str(values['location_verification_source']).strip()
    values['verified_city_id'] = str(values['verified_city_id']).strip()
    # A key alone is not a verified assertion. Keep verification only with its evidence.
    if not values['location_verification_source']:
        values['verified_city_id'] = ''
    if values['eligibility_status'] in ('verified', 'eligible', 'approved') and (
            not values['location_verification_source'] or not values['verified_city_id']):
        values['eligibility_status'] = 'unverified'
    return values
