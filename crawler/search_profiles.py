"""Search profiles: what kind of footage to look for, and how to ask YouTube for it.

A profile bundles query templates (per language), the YouTube `videoDuration` /
`videoDefinition` search filters, the metadata-stage duration window and an optional
title-keyword list. The default, `dashcam`, matches PedX-Insight's curated dataset:
in a random sample of 98 of its videos, 56 titles contain "driving" and 29 "drive",
none mention walking or crossings, and the analysed segments run from 155 s (p1) to
14,000 s (p99), median 1,499 s. The 240 s - 3 h window keeps ~96% of those videos.

Localized templates matter for a cross-cultural dataset: English-only queries mostly
return English-speaking (often tourist) uploaders. Templates put `{city}` where no
grammatical inflection is needed, and native speakers should review them; edit
LOCALIZED_TERMS freely - every query actually sent is recorded in the run's decision
log, so changes stay auditable.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VALID_DURATIONS = ('any', 'short', 'medium', 'long')  # YouTube search.list videoDuration
VALID_DEFINITIONS = ('any', 'high', 'standard')        # YouTube search.list videoDefinition


@dataclass(frozen=True)
class SearchProfile:
    name: str
    description: str
    terms: Dict[str, Tuple[str, ...]]      # relevance language -> query templates ({city})
    title_keywords: Tuple[str, ...]        # for the optional --require-title-keywords gate
    video_durations: Tuple[str, ...]       # default videoDuration filters (one stream each)
    definition: str                        # default videoDefinition filter
    min_duration: int                      # metadata stage, seconds
    max_duration: Optional[int]            # metadata stage, seconds (None = no limit)
    negative_keywords: Tuple[str, ...] = ()  # profile-specific additions to the blocklist

    def templates(self, language: Optional[str]) -> Tuple[str, ...]:
        """Templates for a relevance language, falling back to its base language."""
        if language is None:
            return self.terms['en']
        if language in self.terms:
            return self.terms[language]
        return self.terms.get(language.split('-')[0], ())


DASHCAM = SearchProfile(
    name='dashcam',
    description='In-vehicle driving footage (matches the curated PedX-Insight dataset)',
    terms={
        'en': ('{city} driving 4K', 'driving in {city}', '{city} dashcam', '{city} drive',
               '{city} driving tour'),
        'es': ('manejando en {city}', 'conduciendo por {city}'),
        'pt': ('dirigindo em {city}', '{city} de carro'),
        'fr': ('{city} en voiture', 'conduite à {city}'),
        'de': ('Autofahrt durch {city}', '{city} Dashcam'),
        'it': ('{city} in auto', 'guidando a {city}'),
        'nl': ('rijden door {city}', '{city} autorit'),
        'sv': ('körning genom {city}', '{city} bilfärd'),
        'pl': ('{city} jazda samochodem', '{city} przejazd'),
        'cs': ('{city} jízda autem',),
        'hu': ('{city} autózás',),
        'ro': ('{city} cu mașina',),
        'el': ('{city} οδήγηση',),
        'tr': ('{city} arabayla şehir turu', '{city} araç kamerası'),
        'ru': ('{city} на машине', '{city} видеорегистратор'),
        'uk': ('{city} на машині',),
        'ar': ('قيادة في {city}', 'جولة بالسيارة في {city}'),
        'fa': ('رانندگی در {city}',),
        'ja': ('{city} 車載動画', '{city} ドライブ'),
        'ko': ('{city} 드라이브', '{city} 블랙박스'),
        'zh-Hans': ('{city} 开车', '{city} 行车记录仪'),
        'zh-Hant': ('{city} 開車', '{city} 行車記錄器'),
        'th': ('ขับรถ {city}',),
        'vi': ('lái xe {city}',),
        'id': ('berkendara di {city}', 'keliling {city} naik mobil'),
        'ms': ('memandu di {city}',),
    },
    title_keywords=(
        'driving', 'drive', 'dashcam', 'dash cam', 'road trip', 'roadtrip', 'car ride',
        'manejando', 'conduciendo', 'dirigindo', 'de carro', 'en voiture', 'conduite',
        'autofahrt', 'fahrt', 'in auto', 'guidando', 'rijden', 'autorit', 'körning', 'bilfärd',
        'jazda', 'jízda', 'autózás', 'mașina', 'οδήγηση', 'arabayla', 'araç kamerası',
        'на машине', 'видеорегистратор', 'поездка', 'на машині', 'قيادة', 'رانندگی',
        '車載', 'ドライブ', '드라이브', '블랙박스', '开车', '開車', '行车', '行車',
        'ขับรถ', 'lái xe', 'berkendara', 'naik mobil', 'memandu',
    ),
    video_durations=('medium', 'long'),
    definition='high',
    min_duration=240,
    max_duration=3 * 3600,
    negative_keywords=('drone', 'cab ride', 'cockpit', 'train ride'),
)

WALKING = SearchProfile(
    name='walking',
    description='Street-level walking footage (handheld / gimbal walking tours)',
    terms={
        'en': ('{city} walking tour 4K', 'walking in {city}', '{city} walk'),
        'es': ('caminando por {city}', '{city} paseo a pie'),
        'pt': ('caminhando por {city}', '{city} a pé'),
        'fr': ('{city} à pied', 'balade à {city}'),
        'de': ('Spaziergang durch {city}', '{city} zu Fuß'),
        'it': ('passeggiata a {city}', '{city} a piedi'),
        'nl': ('wandelen door {city}',),
        'ru': ('{city} прогулка', '{city} пешком'),
        'ja': ('{city} 散歩', '{city} 街歩き'),
        'ko': ('{city} 산책', '{city} 걷기'),
        'zh-Hans': ('{city} 漫步', '{city} 步行'),
        'zh-Hant': ('{city} 散步', '{city} 漫步'),
        'tr': ('{city} yürüyüş turu',),
        'ar': ('جولة مشي في {city}',),
        'th': ('เดินเล่น {city}',),
        'vi': ('đi bộ {city}',),
        'id': ('jalan kaki di {city}',),
    },
    title_keywords=(
        'walk', 'walking', 'caminando', 'caminhando', 'à pied', 'balade', 'spaziergang',
        'zu fuß', 'passeggiata', 'a piedi', 'wandelen', 'прогулка', 'пешком', '散歩', '街歩き',
        '산책', '걷기', '漫步', '步行', 'yürüyüş', 'مشي', 'เดินเล่น', 'đi bộ', 'jalan kaki',
    ),
    video_durations=('medium', 'long'),
    definition='high',
    min_duration=240,
    max_duration=3 * 3600,
    negative_keywords=('drone',),
)

# The crawler's original behaviour (English crossing queries, 1-20 min clips), kept for
# reproducing earlier outputs; localized crossing terms are added on top.
CROSSING = SearchProfile(
    name='crossing',
    description='Short static clips of street crossings (original crawler behaviour)',
    terms={
        'en': ('{city} street crossing pedestrian', '{city} crosswalk pedestrian',
               '{city} traffic light pedestrian', '{city} walking street crossing',
               '{city} pedestrian crossing street'),
        'ja': ('{city} 横断歩道',),
        'de': ('{city} Zebrastreifen', '{city} Fußgängerüberweg'),
        'fr': ('{city} passage piéton',),
        'es': ('{city} paso de peatones',),
        'pt': ('{city} faixa de pedestres',),
        'it': ('{city} strisce pedonali',),
        'nl': ('{city} zebrapad',),
        'ru': ('{city} пешеходный переход',),
        'ko': ('{city} 횡단보도',),
        'zh-Hans': ('{city} 斑马线',),
        'zh-Hant': ('{city} 斑馬線',),
        'tr': ('{city} yaya geçidi',),
        'pl': ('{city} przejście dla pieszych',),
        'sv': ('{city} övergångsställe',),
    },
    title_keywords=(
        'crosswalk', 'zebra crossing', 'pedestrian crossing', 'street crossing', 'jaywalking',
        'intersection', '横断歩道', 'zebrastreifen', 'fußgängerüberweg', 'passage piéton',
        'paso de peatones', 'faixa de pedestres', 'strisce pedonali', 'zebrapad',
        'пешеходный переход', '횡단보도', '斑马线', '斑馬線', 'yaya geçidi',
        'przejście dla pieszych', 'övergångsställe',
    ),
    video_durations=('any',),
    definition='any',
    min_duration=60,
    max_duration=20 * 60,
)

PROFILES: Dict[str, SearchProfile] = {p.name: p for p in (DASHCAM, WALKING, CROSSING)}


@dataclass
class SearchStream:
    """One paginated search.list query: a query string + language + duration filter."""
    query: str
    language: Optional[str]
    duration: str
    page_token: Optional[str] = None
    pages: int = 0
    items_seen: int = 0  # results returned so far -> 1-based search rank of the next item
    exhausted: bool = False
    seen_page_tokens: set = field(default_factory=set)

    def label(self) -> str:
        return f"'{self.query}' [lang={self.language or '-'}, duration={self.duration}]"


def resolve_languages(spec: str, profile: SearchProfile,
                      country_languages: Sequence[str] = ()) -> List[Optional[str]]:
    """Relevance languages to search in.

    spec 'auto': the country's primary language (when the profile has templates for it),
                 then English. Without a known country: English only.
    spec 'none': English templates and no relevanceLanguage parameter (legacy behaviour).
    otherwise:   comma-separated codes, e.g. 'ja,en'. Codes without templates are dropped.
    """
    spec = (spec or 'auto').strip()
    if spec == 'none':
        return [None]
    if spec == 'auto':
        wanted = list(country_languages[:1]) + ['en']
    else:
        wanted = [c.strip() for c in spec.split(',') if c.strip()]
    languages: List[Optional[str]] = []
    for code in wanted:
        if code and profile.templates(code) and code not in languages:
            languages.append(code)
    return languages or ['en']


def build_streams(profile: SearchProfile, city: str, languages: Iterable[Optional[str]],
                  durations: Iterable[str], all_terms: bool = False,
                  local_name: Optional[str] = None) -> List[SearchStream]:
    """All (query, language, duration) streams for one city, in priority order.

    Ordered by term rank, then language, then duration, so with --sampling greedy the
    primary query in the local language is tried first. `local_name` (third field in
    cities.txt) replaces the English city name in non-English templates.
    """
    languages = list(languages)
    durations = list(durations)
    per_language = {lang: profile.templates(lang) if all_terms else profile.templates(lang)[:1]
                    for lang in languages}
    depth = max((len(t) for t in per_language.values()), default=0)
    streams: List[SearchStream] = []
    seen = set()
    for rank in range(depth):
        for lang in languages:
            templates = per_language[lang]
            if rank >= len(templates):
                continue
            name = local_name if (local_name and lang not in (None, 'en')) else city
            query = templates[rank].format(city=name)
            for duration in durations:
                key = (query, lang, duration)
                if key not in seen:
                    seen.add(key)
                    streams.append(SearchStream(query=query, language=lang, duration=duration))
    return streams
