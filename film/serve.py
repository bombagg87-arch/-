#!/usr/bin/env python3
"""
Мини-сервер для превью фильма «Грозовой клинок».

Зачем не `python -m http.server`:
  * поддерживает Range-запросы (206) — без них браузер не может перематывать mp3
    и субтитры рассинхронизируются;
  * не кеширует HTML, чтобы правки сразу попадали в превью;
  * не ставит X-Frame-Options — страницу можно показывать в iframe превью;
  * слушает 0.0.0.0 и игнорирует Host, поэтому работает за прокси e2b.

Запуск:  python3 film/serve.py [порт]
"""
import os
import posixpath
import re
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class FilmHandler(SimpleHTTPRequestHandler):
    server_version = "GrozovoyKlinok/1.0"
    protocol_version = "HTTP/1.1"   # keep-alive: превью грузится заметно быстрее

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def translate_path(self, path):
        path = urllib.parse.urlsplit(path).path
        path = posixpath.normpath(urllib.parse.unquote(path))
        parts = [p for p in path.split("/") if p and p not in (".", "..")]
        full = os.path.join(ROOT, *parts)
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        return full

    def do_GET(self):
        if self.headers.get("Range"):
            return self.send_partial(with_body=True)
        super().do_GET()

    def do_HEAD(self):
        if self.headers.get("Range"):
            return self.send_partial(with_body=False)
        super().do_HEAD()

    def send_partial(self, with_body=True):
        """Отдаём кусок файла: без этого браузер не умеет seek в mp3."""
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return

        size = os.path.getsize(path)
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not m:
            self.send_error(416, "Bad Range")
            return

        start, end = m.group(1), m.group(2)
        if start == "":                       # суффиксный range: bytes=-N
            length = int(end or 0)
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(start)
            end = min(int(end) if end else size - 1, size - 1)

        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        if not with_body:
            return
        remaining = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        return {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".md": "text/markdown; charset=utf-8",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".mp3": "audio/mpeg",
            ".zip": "application/zip",
        }.get(ext, super().guess_type(path))

    def log_message(self, fmt, *args):
        sys.stdout.write("%s — %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()


if __name__ == "__main__":
    os.chdir(ROOT)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), FilmHandler)
    print(f"Фильм «Грозовой клинок»: http://0.0.0.0:{PORT}/  (root: {ROOT})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
