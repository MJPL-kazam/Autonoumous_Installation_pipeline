"""
Step 2 — Manual Review Web Application
=======================================
Browser-based UI for reviewing and correcting DINO candidate labels.
Allows manual box adjustment, LCD value entry, and approve/reject workflow.

Usage:
  python review_app.py
  python review_app.py --manifest path/to/review_manifest.json --port 7862
  python review_app.py --show-all   # include approved/rejected in queue
"""

import argparse
import csv
import json
import mimetypes
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Any, Optional, Union

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "dino_auto_label" / "review_manifest.json"
DEFAULT_PORT = int(os.environ.get("DINO_REVIEW_PORT", "7862"))

REGIONS: dict[str, dict[str, str]] = {
    "kazam_box":  {"label": "0 Kazam box",   "color": "#ff5028"},
    "multimeter": {"label": "1 Multimeter",  "color": "#ffff00"},
    "lcd_screen": {"label": "2 LCD screen",  "color": "#2828ff"},
    "kazam_box_black_cover": {"label": "3 Black cover", "color": "#000000"},
    "qr_code": {"label": "4 QR code", "color": "#ff00ff"},
}

class ManifestManager:
    """Manages reading, writing, and filtering the review manifest."""
    def __init__(self, manifest_path: Path, show_all: bool = False):
        self.manifest_path = manifest_path
        self.show_all = show_all
        
        # Security: allowed roots for serving images
        self.allowed_roots = [
            SCRIPT_DIR.resolve(),
            SCRIPT_DIR.parent.resolve(),
        ]

    def load_rows(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        csv_path = self.manifest_path.with_suffix(".csv")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

        preferred = [
            "image_id", "installation_id", "measurement_type",
            "relative_image_path", "original_image_path",
            "copied_image_path", "candidate_preview_path",
            "image_width", "image_height",
            "review_status", "lcd_screen_value",
            "created_at", "updated_at",
        ]
        fields = preferred + sorted({k for r in rows for k in r if k not in preferred})
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({f: r.get(f, "") for f in fields})

    @staticmethod
    def parse_box(value: Any) -> Optional[list[int]]:
        if not value:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return None
        if not isinstance(value, list) or len(value) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
        return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None

    @staticmethod
    def normalize_box(box_dict: Optional[dict[str, Any]]) -> Union[list[int], str]:
        if not box_dict:
            return ""
        x1 = int(round(min(box_dict.get("x1", 0), box_dict.get("x2", 0))))
        y1 = int(round(min(box_dict.get("y1", 0), box_dict.get("y2", 0))))
        x2 = int(round(max(box_dict.get("x1", 0), box_dict.get("x2", 0))))
        y2 = int(round(max(box_dict.get("y1", 0), box_dict.get("y2", 0))))
        return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else ""

    def has_all_boxes(self, row: dict[str, Any]) -> bool:
        required = ["kazam_box", "multimeter", "lcd_screen"]
        return all(self.parse_box(row.get(f"{k}_bbox_xyxy")) for k in required)

    def is_final(self, row: dict[str, Any]) -> bool:
        return (
            row.get("review_status") == "approved"
            and self.has_all_boxes(row)
            and bool(row.get("lcd_screen_value"))
        )

    def review_queue(self, rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
        if self.show_all:
            return list(enumerate(rows))
        return [
            (i, r) for i, r in enumerate(rows)
            if r.get("review_status") != "rejected" and not self.is_final(r)
        ]

    def stats(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        approved = sum(1 for r in rows if r.get("review_status") == "approved")
        rejected = sum(1 for r in rows if r.get("review_status") == "rejected")
        pending = len(rows) - approved - rejected
        return {"total": len(rows), "approved": approved, "rejected": rejected, "pending": pending}

    def row_payload(self, row: dict[str, Any], row_index: Optional[int] = None) -> dict[str, Any]:
        boxes = {}
        for k in REGIONS:
            b = self.parse_box(row.get(f"{k}_bbox_xyxy"))
            if b:
                boxes[k] = {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
        return {
            "row_index": row_index,
            "image_id": row.get("image_id", ""),
            "installation_id": row.get("installation_id", ""),
            "measurement_type": row.get("measurement_type", ""),
            "relative_image_path": row.get("relative_image_path", ""),
            "copied_image_path": row.get("copied_image_path", ""),
            "candidate_preview_path": row.get("candidate_preview_path", ""),
            "image_width": row.get("image_width", ""),
            "image_height": row.get("image_height", ""),
            "review_status": row.get("review_status", "pending"),
            "lcd_screen_value": row.get("lcd_screen_value", ""),
            "boxes": boxes,
            "confidences": {k: row.get(f"{k}_confidence", "") for k in REGIONS},
            "labels": {k: row.get(f"{k}_dino_label", "") for k in REGIONS},
        }

    def resolve_file(self, path_text: str) -> Optional[Path]:
        if not path_text:
            return None
        candidate = Path(path_text).resolve()
        for root in self.allowed_roots:
            try:
                candidate.relative_to(root)
                if candidate.exists() and candidate.is_file():
                    return candidate
            except ValueError:
                pass
        return None

class ReviewServer(ThreadingHTTPServer):
    """Custom HTTP server that holds a reference to the ManifestManager."""
    def __init__(self, server_address, RequestHandlerClass, manifest_manager: ManifestManager):
        super().__init__(server_address, RequestHandlerClass)
        self.manifest_manager = manifest_manager

class ReviewHandler(BaseHTTPRequestHandler):
    """Handles API and static file requests."""
    @property
    def manager(self) -> ManifestManager:
        return self.server.manifest_manager  # type: ignore

    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, payload: Any, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str) -> None:
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            index_path = SCRIPT_DIR / "static" / "index.html"
            if not index_path.exists():
                self.send_error(500, "static/index.html not found. Please ensure the UI template exists.")
                return
            html = index_path.read_text(encoding="utf-8")
            self._html(html.replace("__REGIONS__", json.dumps(REGIONS)))
            return

        if path == "/api/manifest":
            rows = self.manager.load_rows()
            queue = self.manager.review_queue(rows)
            self._json({
                "queue": [self.manager.row_payload(r, i) for i, r in queue],
                "stats": self.manager.stats(rows),
            })
            return

        if path == "/api/item":
            rows = self.manager.load_rows()
            queue = self.manager.review_queue(rows)
            idx = int(qs.get("index", ["0"])[0])
            if idx < 0 or idx >= len(queue):
                self._json({"error": "index out of range"}, 404)
                return
            row_idx, row = queue[idx]
            self._json(self.manager.row_payload(row, row_idx))
            return

        if path == "/file":
            rows = self.manager.load_rows()
            queue = self.manager.review_queue(rows)
            idx = int(qs.get("index", ["0"])[0])
            mode = unquote(qs.get("mode", ["raw"])[0])
            if idx < 0 or idx >= len(queue):
                self.send_error(404)
                return
            _, row = queue[idx]
            
            if mode == "preview":
                path_text = row.get("candidate_preview_path", "")
            else:
                path_text = row.get("copied_image_path") or row.get("original_image_path", "")
                
            resolved = self.manager.resolve_file(path_text)
            if resolved is None:
                self.send_error(404)
                return
                
            data = resolved.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save":
            self.send_error(404)
            return
            
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            rows = self.manager.load_rows()
            idx = int(payload.get("row_index", payload.get("index")))
            row = rows[idx]

            review_status = payload.get("review_status", "pending")
            row["review_status"] = review_status
            row["lcd_screen_value"] = payload.get("lcd_screen_value", "")
            row["updated_at"] = datetime.now().isoformat(timespec="seconds")

            for key in REGIONS:
                box = self.manager.normalize_box(payload.get("boxes", {}).get(key))
                row[f"{key}_bbox_xyxy"] = json.dumps(box) if box else ""

            if review_status == "approved":
                if not row["lcd_screen_value"]:
                    self._json({"ok": False, "error": "Approved rows require LCD value."}, 400)
                    return
                missing = [k for k in ["kazam_box", "multimeter", "lcd_screen"] if not self.manager.parse_box(row.get(f"{k}_bbox_xyxy"))]
                if missing:
                    self._json({"ok": False, "error": f"Missing required boxes: {', '.join(missing)}"}, 400)
                    return
            elif review_status == "rejected":
                bad_txt = SCRIPT_DIR.parent / "bad_images.txt"
                rel_path = row.get("relative_image_path", "")
                inst_id = row.get("installation_id", "")
                line = rel_path
                
                existing = bad_txt.read_text(encoding="utf-8") if bad_txt.exists() else ""
                if line not in existing:
                    with bad_txt.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")

            rows[idx] = row
            self.manager.write_rows(rows)
            self._json({"ok": True})
        except Exception as exc:
            self._json({"ok": False, "error": repr(exc)}, 400)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review and correct DINO candidate labels.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--show-all", action="store_true", help="Include approved/rejected in review queue.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    
    manager = ManifestManager(manifest_path=args.manifest.resolve(), show_all=args.show_all)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        print("Run run_dino.py first to generate candidates.")
        return

    rows = manager.load_rows()
    stats = manager.stats(rows)
    print(f"Manifest : {args.manifest}")
    print(f"Records  : {stats['total']} total, {stats['approved']} approved, "
          f"{stats['rejected']} rejected, {stats['pending']} pending")

    host = "127.0.0.1"
    server = ReviewServer((host, args.port), ReviewHandler, manager)
    url = f"http://{host}:{args.port}"
    print(f"Open     : {url}")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()

if __name__ == "__main__":
    main()
