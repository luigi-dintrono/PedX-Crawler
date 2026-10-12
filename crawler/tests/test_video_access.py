"""Tests for video_access.py: the localhost range-chunking proxy and FrameSampler.sample().

The fake upstream behaves like googlevideo's android_vr URLs: only *bounded* range
requests are served (206); open-ended / absent ranges and ranges ending past a limit get 403.
"""
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

import filter_common as fc
import video_access as va

DATA = bytes(i % 251 for i in range(300_000))


def _fake_upstream(limit=None):
    requests_seen = []

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get('Range', '')
            requests_seen.append(rng)
            m = re.fullmatch(r'bytes=(\d+)-(\d+)', rng)
            if not m:
                self.send_error(403)
                return
            start, end = int(m.group(1)), min(int(m.group(2)), len(DATA) - 1)
            if limit is not None and end >= limit:
                self.send_error(403)
                return
            body = DATA[start:end + 1]
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(DATA)}')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests_seen


@pytest.fixture
def upstream():
    servers = []

    def make(limit=None):
        server, seen = _fake_upstream(limit)
        servers.append(server)
        return va.Stream(url=f'http://127.0.0.1:{server.server_address[1]}/v'), seen
    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


def test_upstream_really_rejects_open_ended_ranges(upstream):
    stream, _ = upstream()
    assert requests.get(stream.url, headers={'Range': 'bytes=10-'}).status_code == 403
    assert requests.get(stream.url).status_code == 403


def test_proxy_serves_open_ended_ranges_with_bounded_upstream_requests(upstream, monkeypatch):
    monkeypatch.setattr(va, 'CHUNK_BYTES', 100_000)
    stream, seen = upstream()
    with va.ChunkedRangeProxy(stream) as proxy:
        r = requests.get(proxy.url, headers={'Range': 'bytes=50000-'})
        assert r.status_code == 206 and r.content == DATA[50000:]
        assert r.headers['Content-Range'] == f'bytes 50000-{len(DATA) - 1}/{len(DATA)}'
        full = requests.get(proxy.url)  # no Range: whole body, 200
        assert full.status_code == 200 and full.content == DATA
        tail = requests.get(proxy.url, headers={'Range': 'bytes=-1000'})
        assert tail.content == DATA[-1000:]
        assert requests.head(proxy.url).headers['Content-Length'] == str(len(DATA))
    assert seen and all(re.fullmatch(r'bytes=\d+-\d+', r) for r in seen)  # every upstream range bounded
    sizes = [int(r.split('-')[1]) - int(r.split('=')[1].split('-')[0]) + 1 for r in seen]
    assert max(sizes) <= 100_000


def test_proxy_records_upstream_refusal(upstream, monkeypatch):
    monkeypatch.setattr(va, 'CHUNK_BYTES', 50_000)
    stream, _ = upstream(limit=120_000)         # nothing past byte ~120k is served
    with va.ChunkedRangeProxy(stream) as proxy:
        with requests.get(proxy.url, stream=True) as r:
            try:
                body = b''.join(r.iter_content(65536))
            except requests.RequestException:
                body = b''                      # truncated response is also acceptable
        assert len(body) < len(DATA)
        assert proxy.upstream_status == 403


def test_proxy_probe_refusal_raises(upstream):
    stream, _ = upstream(limit=0)
    with pytest.raises(va.StreamRefused) as exc:
        with va.ChunkedRangeProxy(stream):
            pass
    assert exc.value.status == 403


# --- FrameSampler.sample() ------------------------------------------------------------

class _Proc:
    def __init__(self, rc=0, err=''):
        self.returncode, self.stderr = rc, err


def test_sample_extracts_one_frame_per_timestamp(tmp_path, upstream, monkeypatch):
    stream, _ = upstream()
    calls = []

    def fake_ffmpeg(cmd, **kw):
        calls.append(cmd)
        with open(cmd[-1], 'wb') as f:
            f.write(b'jpg')
        return _Proc()
    monkeypatch.setattr(va, 'resolve_stream', lambda *a, **k: stream)
    monkeypatch.setattr(fc.subprocess, 'run', fake_ffmpeg)
    s = fc.FrameSampler(temp_dir=str(tmp_path), n_frames=3, js_runtime=None)
    frames = s.sample('abc', 'https://youtu.be/abc', 900)
    assert [f.timestamp for f in frames] == [150.0, 450.0, 750.0]
    assert all(c[c.index('-i') + 1].startswith('http://127.0.0.1:') for c in calls)


def test_sample_access_denied_when_probe_refused(tmp_path, upstream, monkeypatch):
    stream, _ = upstream(limit=0)
    monkeypatch.setattr(va, 'resolve_stream', lambda *a, **k: stream)
    s = fc.FrameSampler(temp_dir=str(tmp_path), n_frames=2, js_runtime=None)
    with pytest.raises(fc.AccessDeniedError):
        s.sample('abc', 'u', 600)


def test_sample_resolve_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('Sign in to confirm you are not a bot')
    monkeypatch.setattr(va, 'resolve_stream', boom)
    s = fc.FrameSampler(temp_dir=str(tmp_path), n_frames=2, js_runtime=None)
    with pytest.raises(fc.SamplingError, match='could not resolve a stream'):
        s.sample('abc', 'u', 600)


def test_sample_ffmpeg_failure_reports_last_error(tmp_path, upstream, monkeypatch):
    stream, _ = upstream()
    monkeypatch.setattr(va, 'resolve_stream', lambda *a, **k: stream)
    monkeypatch.setattr(fc.subprocess, 'run', lambda cmd, **k: _Proc(1, 'moov atom not found'))
    s = fc.FrameSampler(temp_dir=str(tmp_path), n_frames=2, js_runtime=None)
    with pytest.raises(fc.SamplingError, match='moov atom not found'):
        s.sample('abc', 'u', 600)
