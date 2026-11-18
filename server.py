"""Tiny web server: serves templates/ + static/ and exposes POST /ask."""

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config


def make_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = (config.TEMPLATES_DIR / "index.html").read_bytes()
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path.startswith("/static/"):
                target = (config.STATIC_DIR / self.path[len("/static/"):]).resolve()
                # block path traversal outside the static folder
                if config.STATIC_DIR.resolve() in target.parents and target.is_file():
                    ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                    return self._send(200, target.read_bytes(), ctype)
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path not in ("/ask", "/ask/stream"):
                return self._send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/ask/stream":
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                for event in pipeline.stream_answer(
                    data.get("question", ""), data.get("history", [])
                ):
                    line = json.dumps(event, separators=(",", ":")).encode() + b"\n"
                    self.wfile.write(line)
                    self.wfile.flush()
                return

            reply, hits = pipeline.answer(data.get("question", ""), data.get("history", []))
            out = {"answer": reply, "sources": sorted({c["source"] for _, c in hits})}
            self._send(200, json.dumps(out).encode(), "application/json")

        def log_message(self, *args):
            pass

    return Handler


def run_web(pipeline, port):
    print(f"Web UI running at http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(pipeline)).serve_forever()
