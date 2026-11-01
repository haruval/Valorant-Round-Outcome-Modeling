"""Serve the replay viewer on loopback using only Python's standard library."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from replay_data import build_catalog

WEB_ROOT = Path(__file__).resolve().parent / "viewer"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/comparisons.js": ("comparisons.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"),
          "/maps/manifest.json": ("maps/manifest.json", "application/json; charset=utf-8"),
          "/maps/breeze.png": ("maps/breeze.png", "image/png"),
          "/maps/ascent.png": ("maps/ascent.png", "image/png")}


def make_handler(catalog: dict, rounds: dict):
    catalog_bytes = json.dumps(catalog, allow_nan=False).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "RoundReplay"
        sys_version = ""

        def respond(self, status, body, content_type="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                             "style-src 'self'; connect-src 'self'; img-src 'self' data:; "
                             "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            port = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed:
                return self.respond(403, b"Loopback host required")
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}:
                return self.respond(403, b"Cross-origin requests are not allowed")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                return self.respond(403, b"Cross-site requests are not allowed")
            path = urlsplit(self.path).path
            if path == "/api/catalog":
                return self.respond(200, catalog_bytes, "application/json; charset=utf-8")
            if path.startswith("/api/round/"):
                body = rounds.get(path.removeprefix("/api/round/"))
                return self.respond(200, body, "application/json; charset=utf-8") if body else self.respond(404, b"Round not found")
            if path in ASSETS:
                filename, content_type = ASSETS[path]
                return self.respond(200, (WEB_ROOT / filename).read_bytes(), content_type)
            return self.respond(404, b"Not found")

        def log_message(self, message, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--predictions", type=Path, help="Optional JSONL output from predict.py; no models are loaded")
    parser.add_argument("--run-manifest", type=Path, help="Matching run.json for prediction provenance")
    parser.add_argument("--rolling-predictions", type=Path, help="Time-indexed rolling prediction JSONL")
    parser.add_argument("--rolling-run-manifest", type=Path, help="Matching rolling run.json")
    parser.add_argument("--comparison-config", type=Path, help="Pinned validation runs and utility/movement review; JSON only")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    if args.comparison_config and any((args.predictions, args.run_manifest, args.rolling_predictions, args.rolling_run_manifest)):
        parser.error("Comparison config supplies its own separate prediction runs")
    try:
        catalog, payloads = build_catalog(args.data_dir, args.predictions, args.run_manifest,
                                         args.rolling_predictions, args.rolling_run_manifest)
        if args.comparison_config:
            from comparison_data import load_comparisons, attach_comparisons, add_entry_review
            config, comparisons = load_comparisons(args.comparison_config, Path(__file__).resolve().parent)
            attach_comparisons(catalog, payloads, config, comparisons)
            add_entry_review(args.data_dir, payloads)
            catalog["entry_review_cases"] = config.get("review_cases", [])
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(catalog, payloads))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Local: http://127.0.0.1:{args.port} | {len(payloads)} rounds | no model runtime", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
