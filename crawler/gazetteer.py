"""Offline city / country lookup backed by GeoNames (via the `geonamescache` package).

Replaces the crawler's hand-maintained city->country table and the bridge's ISO2->name
table. Everything runs offline from the data bundled with geonamescache; nothing is
sent to a geocoding service.

City resolution is deliberately conservative and transparent:
  * exact (accent- and case-insensitive) match on the GeoNames primary name, searched
    through the 15k-, 1k- and 500-population tables in turn; only if no table has a
    primary-name match are alternate names tried. (Alternate names include nicknames:
    "Crescent City" is an alternate name of New Orleans, so letting a big-city alternate
    beat a small-town primary name misplaces Crescent City, CA by 3,000 km.)
  * restricted to the given ISO2 country code when one is known;
  * ties broken by population (so "London" -> London, GB rather than London, CA);
  * `ambiguous` is set when a rival candidate has at least 10% of the chosen city's
    population (e.g. Portland, OR vs Portland, ME) — such names need a human check.
The match kind and GeoNames id are returned so callers can record provenance.

Validated against the 2,346 hand-geocoded cities of PedX-Insight's mapping.csv
(see README, "Geocoding").
"""

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# GeoNames language tags -> the codes YouTube's relevanceLanguage accepts (ISO 639-1,
# plus zh-Hans / zh-Hant for Chinese scripts).
_CHINESE_SCRIPT = {'zh-CN': 'zh-Hans', 'zh-SG': 'zh-Hans', 'zh-TW': 'zh-Hant',
                   'zh-HK': 'zh-Hant', 'zh-MO': 'zh-Hant'}


_SHARED_CACHES: Dict[int, object] = {}
_SHARED_INDEX: Dict[int, Dict[str, List[Tuple[str, dict]]]] = {}


@dataclass(frozen=True)
class CityInfo:
    name: str
    geonameid: int
    country_code: str
    latitude: float
    longitude: float
    population: int
    match: str  # 'name' or 'alternate'
    ambiguous: bool = False  # a rival candidate has >= 10% of this city's population


@dataclass(frozen=True)
class CountryInfo:
    iso2: str
    iso3: str
    name: str
    continent_code: str
    continent: str
    population: int
    languages: Tuple[str, ...]  # GeoNames order, most widely spoken first

    @property
    def primary_language(self) -> Optional[str]:
        """Primary language as a YouTube relevanceLanguage code, e.g. 'ja', 'zh-Hant'."""
        return to_relevance_language(self.languages[0]) if self.languages else None


def to_relevance_language(tag: str) -> Optional[str]:
    """'de-DE' -> 'de', 'zh-TW' -> 'zh-Hant', 'fil' -> None (not a 2-letter code)."""
    tag = (tag or '').strip()
    if tag in _CHINESE_SCRIPT:
        return _CHINESE_SCRIPT[tag]
    base = tag.split('-')[0].lower()
    if base == 'zh':
        return 'zh-Hans'
    return base if len(base) == 2 else None


def normalize_name(name: str) -> str:
    """Case-, accent- and punctuation-insensitive key: 'São  Paulo' -> 'sao paulo'."""
    decomposed = unicodedata.normalize('NFKD', str(name))
    stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = ''.join(c if c.isalnum() else ' ' for c in stripped.casefold())
    return ' '.join(cleaned.split())


class Gazetteer:
    """Lazy wrapper around geonamescache. Construct once and reuse (data loads on demand)."""

    POPULATION_TIERS = (15000, 1000, 500)
    AMBIGUITY_RATIO = 0.1

    def __init__(self):
        try:
            import geonamescache
        except ImportError:  # core dependency, but degrade gracefully (e.g. minimal test envs)
            geonamescache = None
        self._module = geonamescache
        # GeoNames tables are large (the 500-population table is ~80 MB of JSON), so
        # loaded tables and name indexes are shared by every instance in the process.
        self._caches = _SHARED_CACHES
        self._index = _SHARED_INDEX
        self._countries: Optional[dict] = None
        self._continents: Optional[dict] = None
        self._city_cache: Dict[Tuple[str, Optional[str]], Optional[CityInfo]] = {}

    @property
    def available(self) -> bool:
        return self._module is not None

    def _cache(self, tier: int):
        if tier not in self._caches:
            self._caches[tier] = self._module.GeonamesCache(min_city_population=tier)
        return self._caches[tier]

    def _city_index(self, tier: int) -> Dict[str, List[Tuple[str, dict]]]:
        """normalized name -> [(match kind, city dict)], built once per tier."""
        if tier not in self._index:
            index: Dict[str, List[Tuple[str, dict]]] = {}
            for city in self._cache(tier).get_cities().values():
                primary = normalize_name(city['name'])
                index.setdefault(primary, []).append(('name', city))
                for alt in set(normalize_name(a) for a in city.get('alternatenames') or []):
                    if alt and alt != primary:
                        index.setdefault(alt, []).append(('alternate', city))
            self._index[tier] = index
        return self._index[tier]

    def country(self, iso2: Optional[str]) -> Optional[CountryInfo]:
        if not self.available or not iso2:
            return None
        if self._countries is None:
            cache = self._cache(self.POPULATION_TIERS[0])
            self._countries = cache.get_countries()
            self._continents = {code: c.get('name', code) for code, c in cache.get_continents().items()}
        raw = self._countries.get(iso2.strip().upper())
        if not raw:
            return None
        languages = tuple(l for l in (raw.get('languages') or '').split(',') if l)
        continent_code = raw.get('continentcode', '')
        return CountryInfo(iso2=raw['iso'], iso3=raw.get('iso3', ''), name=raw.get('name', ''),
                           continent_code=continent_code,
                           continent=self._continents.get(continent_code, continent_code),
                           population=int(raw.get('population') or 0), languages=languages)

    def resolve_city(self, name: str, country_code: Optional[str] = None) -> Optional[CityInfo]:
        """Best GeoNames match for a city name (optionally within one country), or None."""
        if not self.available or not name or not name.strip():
            return None
        cc = country_code.strip().upper() if country_code and country_code.strip() else None
        if cc == 'UNKNOWN':
            cc = None
        key = (normalize_name(name), cc)
        if key in self._city_cache:
            return self._city_cache[key]
        result = None
        for kind in ('name', 'alternate'):
            for tier in self.POPULATION_TIERS:
                pool = [c for k, c in self._city_index(tier).get(key[0], [])
                        if k == kind and (cc is None or c.get('countrycode') == cc)]
                if pool:
                    result = self._pick(pool, kind)
                    break
            if result:
                break
        self._city_cache[key] = result
        return result

    def _pick(self, pool: List[dict], kind: str) -> CityInfo:
        # Higher tiers are supersets of lower ones, so dedupe by GeoNames id.
        unique = {int(c['geonameid']): c for c in pool}
        ranked = sorted(unique.values(), key=lambda c: int(c.get('population') or 0), reverse=True)
        best = ranked[0]
        best_pop = int(best.get('population') or 0)
        ambiguous = len(ranked) > 1 and int(ranked[1].get('population') or 0) >= self.AMBIGUITY_RATIO * best_pop
        return CityInfo(name=best['name'], geonameid=int(best['geonameid']),
                        country_code=best['countrycode'],
                        latitude=float(best['latitude']), longitude=float(best['longitude']),
                        population=best_pop, match=kind, ambiguous=ambiguous)
