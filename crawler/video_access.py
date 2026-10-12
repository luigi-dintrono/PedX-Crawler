"""Reaching a YouTube video's bytes so ffmpeg can seek into it.

YouTube's current player clients (notably android_vr, the one yt-dlp falls back to
without a PO token) serve googlevideo URLs that only accept *bounded* range requests:
`Range: bytes=a-b` returns 206, but the open-ended `Range: bytes=a-` that ffmpeg sends
(and a request with no Range at all) returns 403. yt-dlp's own HTTP downloader copes by
fetching ~10 MiB chunks, but `--download-sections` hands the URL to ffmpeg, which fails
with "Server returned 403 Forbidden".

`ChunkedRangeProxy` closes that gap: a tiny localhost HTTP server that ffmpeg reads
from. It answers any (open-ended) range request by streaming the bytes from upstream
in bounded chunks. ffmpeg seeks to a timestamp, reads what it needs and disconnects, so
only the index plus the fragments around the timestamp are transferred.

`resolve_stream` uses the yt_dlp Python API to pick a format and return its URL + headers.
"""

import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import requests

# android_vr URLs reject single ranged requests above ~6 MiB (measured: 6.1 MiB); web
# clients allow ~10 MiB. 4 MiB is safe for both.
CHUNK_BYTES = 4 * 1024 * 1024
_RANGE_RE = re.compile(r'bytes=(\d*)-(\d*)')

# Prefer H.264 (decodes everywhere), then any video-only stream <= max_height, then a
# combined stream. Video-only is enough: only frames are needed, never audio.
FORMAT_SELECTOR = ('bv*[height<={h}][vcodec^=avc1]/bv*[height<={h}]/b[height<={h}]/bv*/b')


class StreamRefused(RuntimeError):
    """googlevideo refused the media request (typically HTTP 403: no PO token / cookies)."""

    def __init__(self, status: Optional[int]):
        super().__init__(f"media server refused the request (HTTP {status})")
        self.status = status


@dataclass
class Stream:
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    filesize: Optional[int] = None
    format_id: str = ''
    height: Optional[int] = None


def resolve_stream(video_url: str, max_height: int = 720, cookies_file: Optional[str] = None,
                   js_runtime: Optional[str] = None) -> Stream:
    """Pick a format with yt-dlp and return its direct URL (no download)."""
    import yt_dlp
    opts = {'quiet': True, 'no_warnings': True, 'noplaylist': True, 'skip_download': True,
            'format': FORMAT_SELECTOR.format(h=max_height), 'socket_timeout': 30, 'retries': 2}
    if cookies_file:
        opts['cookiefile'] = cookies_file
    if js_runtime:
        opts['js_runtimes'] = {js_runtime: {}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    chosen = (info.get('requested_formats') or [info])[0]  # the video part of a merge
    if not chosen.get('url'):
        raise RuntimeError('yt-dlp returned no direct URL for the selected format')
    return Stream(url=chosen['url'], headers=dict(chosen.get('http_headers') or {}),
                  filesize=chosen.get('filesize') or chosen.get('filesize_approx'),
                  format_id=str(chosen.get('format_id', '')), height=chosen.get('height'))


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):  # keep the crawl's console clean
        pass

    def _bounds(self):
        total = self.server.total
        start, end = 0, total - 1
        m = _RANGE_RE.fullmatch((self.headers.get('Range') or '').strip())
        if m:
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), total - 1)
            elif m.group(2):  # suffix range: the last N bytes
                start = max(0, total - int(m.group(2)))
        return start, end, bool(m)

    def _send_head(self, start, end, partial):
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{self.server.total}')
        self.end_headers()

    def do_HEAD(self):
        start, end, partial = self._bounds()
        self._send_head(start, end, partial)

    def do_GET(self):
        # One request per connection: if upstream refuses mid-stream the body is short,
        # and closing the socket lets the client see the truncation immediately instead
        # of waiting for the missing bytes.
        self.close_connection = True
        start, end, partial = self._bounds()
        if start > end:
            self.send_error(416)
            return
        self._send_head(start, end, partial)
        pos = start
        try:
            while pos <= end:
                stop = min(pos + CHUNK_BYTES - 1, end)
                headers = {**self.server.headers, 'Range': f'bytes={pos}-{stop}'}
                with self.server.session.get(self.server.url, headers=headers, stream=True,
                                             timeout=30) as upstream:
                    if upstream.status_code != 206:
                        self.server.upstream_status = upstream.status_code
                        return  # truncated body; ffmpeg reports the short read
                    for block in upstream.iter_content(64 * 1024):
                        self.wfile.write(block)
                pos = stop + 1
        except (BrokenPipeError, ConnectionError, requests.RequestException):
            pass  # ffmpeg got what it needed and hung up (or the network dropped)


class ChunkedRangeProxy:
    """Context manager: `with ChunkedRangeProxy(stream) as proxy:`; `proxy.url` is an
    http://127.0.0.1 URL that ffmpeg can open and seek in. Bound to localhost only."""

    def __init__(self, stream: Stream):
        self.stream = stream
        self._server: Optional[ThreadingHTTPServer] = None

    def _probe_total(self, session) -> int:
        headers = {**self.stream.headers, 'Range': 'bytes=0-0'}
        with session.get(self.stream.url, headers=headers, timeout=30, stream=True) as r:
            m = re.fullmatch(r'bytes \d+-\d+/(\d+)', r.headers.get('Content-Range', ''))
            if r.status_code != 206 or not m:
                raise StreamRefused(r.status_code)
            return int(m.group(1))

    def __enter__(self) -> 'ChunkedRangeProxy':
        session = requests.Session()
        total = self._probe_total(session)
        self._server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        self._server.daemon_threads = True
        self._server.total = total
        self._server.url = self.stream.url
        self._server.headers = self.stream.headers
        self._server.session = session
        self._server.upstream_status = None
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self._server.server_address[1]}/video'

    @property
    def upstream_status(self) -> Optional[int]:
        """HTTP status of the first refused upstream chunk request, if any (e.g. 403)."""
        return self._server.upstream_status if self._server else None

    def __exit__(self, *exc):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server.session.close()
        return False
