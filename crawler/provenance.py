"""Run provenance: a per-candidate decision log and a run manifest.

Every video the crawler examines gets one row in `decisions.csv` (accepted, rejected
or could-not-check, with the stage, reason, score, frame-level evidence and the exact
search stream/rank it came from). Together with `manifest.json` (parameters, tool
versions, git commit, per-city counts, quota) this gives a PRISMA-style record of how
the dataset was assembled, and the input for scripts/evaluate_filter.py.
"""

import csv
import json
import os
import platform
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict, Iterable, Optional

DECISION_FIELDS = [
    'run_id', 'timestamp_utc', 'city', 'country_code', 'profile',
    'search_query', 'search_language', 'search_duration', 'search_page', 'search_rank',
    'video_id', 'title', 'channel_id', 'published_at', 'duration_seconds', 'definition',
    'projection', 'live_broadcast', 'default_audio_language',
    'stage', 'decision', 'reason', 'score', 'backend', 'model', 'threshold',
    'frame_source', 'frame_timestamps', 'frame_scores', 'frame_brightness', 'time_of_day',
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(2)


class DecisionLog:
    """Append-only CSV, flushed per row so an interrupted run keeps its audit trail."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, 'a', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=DECISION_FIELDS, extrasaction='ignore')
        if new:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: Dict[str, Any]) -> None:
        out = {}
        for key in DECISION_FIELDS:
            value = row.get(key, '')
            out[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        out['timestamp_utc'] = out['timestamp_utc'] or utc_now()
        self._writer.writerow(out)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_state(repo_dir: str) -> Dict[str, Any]:
    def git(*args):
        return subprocess.run(['git', '-C', repo_dir, *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        return {'commit': git('rev-parse', 'HEAD') or None,
                'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def environment_info(repo_dir: str, packages: Iterable[str]) -> Dict[str, Any]:
    return {'python': sys.version.split()[0], 'platform': platform.platform(),
            'packages': package_versions(packages), 'git': git_state(repo_dir)}


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    """Atomic JSON write (temp file + replace), safe to call repeatedly during a run."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)
