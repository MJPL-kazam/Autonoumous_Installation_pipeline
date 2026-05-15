from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import posixpath
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "data"
MANIFEST_JSON = DATA_ROOT / "review_manifest.json"
MANIFEST_CSV = DATA_ROOT / "review_manifest.csv"

REGIONS = {
    "kazam_box": {"label": "1 Kazam box", "color": "#2f80ff"},
    "multimeter": {"label": "2 Multimeter", "color": "#28d957"},
    "lcd_screen": {"label": "3 LCD screen", "color": "#ff3f3f"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and correct DINO ROI candidates before YOLO export.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_JSON)
    parser.add_argument("--port", type=int, default=int(os.environ.get("YOLO_ROI_REVIEW_PORT", "7862")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--show-all", action="store_true", help="Show approved and rejected records too.")
    return parser.parse_args()


def manifest_csv_path(manifest_json: Path) -> Path:
    return manifest_json.with_suffix(".csv")


def load_rows() -> list[dict]:
    if not SERVER_STATE["manifest_json"].exists():
        return []
    try:
        return json.loads(SERVER_STATE["manifest_json"].read_text(encoding="utf-8"))
    except Exception:
        return []


def write_rows(rows: list[dict]) -> None:
    manifest_json = SERVER_STATE["manifest_json"]
    manifest_csv = manifest_csv_path(manifest_json)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "image_id",
        "relative_image_path",
        "original_image_path",
        "copied_image_path",
        "candidate_preview_path",
        "image_width",
        "image_height",
        "review_status",
        "lcd_screen_value",
        "created_at",
        "updated_at",
    ]
    fields = preferred + [field for field in fields if field not in preferred]
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_box(value):
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
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_box(box):
    if not box:
        return ""
    x1 = int(round(min(box.get("x1", 0), box.get("x2", 0))))
    y1 = int(round(min(box.get("y1", 0), box.get("y2", 0))))
    x2 = int(round(max(box.get("x1", 0), box.get("x2", 0))))
    y2 = int(round(max(box.get("y1", 0), box.get("y2", 0))))
    if x2 <= x1 or y2 <= y1:
        return ""
    return [x1, y1, x2, y2]


def has_required_boxes(row: dict) -> bool:
    return all(parse_box(row.get(f"{key}_bbox_xyxy")) for key in REGIONS)


def is_final_ready(row: dict) -> bool:
    return row.get("review_status") == "approved" and has_required_boxes(row) and bool(row.get("lcd_screen_value"))


def review_queue(rows: list[dict]) -> list[tuple[int, dict]]:
    if SERVER_STATE["show_all"]:
        return list(enumerate(rows))
    queue = []
    for idx, row in enumerate(rows):
        if row.get("review_status") == "rejected":
            continue
        if is_final_ready(row):
            continue
        queue.append((idx, row))
    return queue


def delete_candidate_preview(row: dict) -> None:
    path_text = row.get("candidate_preview_path")
    path = resolve_allowed_file(path_text) if path_text else None
    if path is not None and path.exists():
        path.unlink()
    row["candidate_preview_path"] = ""


def row_to_payload(row: dict, row_index: int | None = None) -> dict:
    boxes = {}
    for key in REGIONS:
        box = parse_box(row.get(f"{key}_bbox_xyxy"))
        if box:
            boxes[key] = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
    return {
        "row_index": row_index,
        "image_id": row.get("image_id", ""),
        "relative_image_path": row.get("relative_image_path", ""),
        "copied_image_path": row.get("copied_image_path", ""),
        "candidate_preview_path": row.get("candidate_preview_path", ""),
        "image_width": row.get("image_width", ""),
        "image_height": row.get("image_height", ""),
        "review_status": row.get("review_status", "needs_recheck"),
        "lcd_screen_value": row.get("lcd_screen_value", ""),
        "boxes": boxes,
        "confidences": {key: row.get(f"{key}_confidence", "") for key in REGIONS},
        "labels": {key: row.get(f"{key}_dino_label", "") for key in REGIONS},
    }


def resolve_allowed_file(path_text: str) -> Path | None:
    if not path_text:
        return None
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = (APP_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    allowed_roots = [APP_ROOT.resolve(), APP_ROOT.parent.resolve()]
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            if candidate.exists() and candidate.is_file():
                return candidate
        except ValueError:
            pass
    return None


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YOLO ROI Review</title>
  <style>
    :root { --bg: #101217; --panel: #181c23; --line: #303642; --text: #f6f8fc; --muted: #a8b0bd; --accent: #36c5f0; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: Segoe UI, Arial, sans-serif; }
    .app { display: grid; grid-template-columns: minmax(0, 1fr) 380px; min-height: 100vh; }
    .stage { display: flex; align-items: center; justify-content: center; padding: 16px; overflow: auto; }
    canvas { display: block; max-width: 100%; max-height: calc(100vh - 32px); border: 1px solid var(--line); background: #07080b; cursor: crosshair; }
    aside { background: var(--panel); border-left: 1px solid var(--line); padding: 16px; overflow: auto; }
    h1 { font-size: 20px; margin: 0 0 10px; }
    .path, .meta, .status { color: var(--muted); font-size: 13px; line-height: 1.35; overflow-wrap: anywhere; }
    .region-list { display: grid; gap: 10px; margin: 14px 0; }
    .region { display: grid; grid-template-columns: 18px 1fr; gap: 10px; align-items: center; border: 1px solid var(--line); padding: 10px; cursor: pointer; }
    .region.active { border-color: var(--accent); background: #222833; }
    .swatch { width: 14px; height: 14px; border-radius: 3px; }
    input, select { width: 100%; border: 1px solid var(--line); background: #0d1016; color: var(--text); padding: 10px; font-size: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
    button { border: 1px solid var(--line); background: #222832; color: var(--text); padding: 10px 12px; cursor: pointer; font-weight: 600; }
    button.primary { background: #145d7a; border-color: #237fa3; }
    .section { margin-top: 16px; }
    .small { font-size: 12px; color: var(--muted); margin-top: 6px; }
  </style>
</head>
<body>
  <div class="app">
    <main class="stage"><canvas id="canvas"></canvas></main>
    <aside>
      <h1>YOLO ROI Review</h1>
      <div id="counter" class="meta"></div>
      <div id="path" class="path"></div>

      <div class="section">
        <label class="meta">Review status</label>
        <select id="reviewStatus">
          <option value="needs_recheck">needs_recheck</option>
          <option value="approved">approved</option>
          <option value="rejected">rejected</option>
        </select>
      </div>

      <div class="section">
        <label class="meta">LCD screen value</label>
        <input id="lcdValue" type="text" placeholder="Example: 002, 217, 0.01" />
      </div>

      <div id="regions" class="region-list"></div>

      <div class="grid">
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
        <button id="clearActiveBtn">Clear ROI</button>
        <button id="clearAllBtn">Clear All</button>
        <button id="rawBtn">Raw</button>
        <button id="previewBtn">DINO Preview</button>
      </div>
      <div class="grid">
        <button id="saveBtn" class="primary">Save</button>
        <button id="saveNextBtn" class="primary">Save + Next</button>
      </div>
      <div class="small">Keys: 1 Kazam, 2 Meter, 3 LCD, arrows navigate. Draw on image to replace selected ROI.</div>
      <pre id="status" class="status"></pre>
    </aside>
  </div>

  <script>
    const REGIONS = __REGIONS__;
    let manifest = [];
    let index = 0;
    let item = null;
    let boxes = {};
    let activeRegion = "kazam_box";
    let img = new Image();
    let showPreview = false;
    let drawing = false;
    let start = null;
    let current = null;

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const counterEl = document.getElementById("counter");
    const pathEl = document.getElementById("path");
    const statusEl = document.getElementById("status");
    const lcdValueEl = document.getElementById("lcdValue");
    const reviewStatusEl = document.getElementById("reviewStatus");

    function status(text) { statusEl.textContent = text; }
    function normBox(b) {
      const x1 = Math.round(Math.min(b.x1, b.x2));
      const y1 = Math.round(Math.min(b.y1, b.y2));
      const x2 = Math.round(Math.max(b.x1, b.x2));
      const y2 = Math.round(Math.max(b.y1, b.y2));
      if (x2 <= x1 || y2 <= y1) return null;
      return { x1, y1, x2, y2 };
    }
    function imagePoint(evt) {
      const r = canvas.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(canvas.width, (evt.clientX - r.left) * canvas.width / r.width)),
        y: Math.max(0, Math.min(canvas.height, (evt.clientY - r.top) * canvas.height / r.height)),
      };
    }
    function buildRegions() {
      const root = document.getElementById("regions");
      root.innerHTML = "";
      Object.entries(REGIONS).forEach(([key, cfg]) => {
        const el = document.createElement("div");
        el.className = "region";
        el.dataset.region = key;
        el.innerHTML = `<span class="swatch" style="background:${cfg.color}"></span><span>${cfg.label}</span>`;
        el.onclick = () => { activeRegion = key; updateRegions(); };
        root.appendChild(el);
      });
      updateRegions();
    }
    function updateRegions() {
      document.querySelectorAll(".region").forEach(el => el.classList.toggle("active", el.dataset.region === activeRegion));
    }
    function drawOne(key, b, transient=false) {
      if (!b) return;
      const cfg = REGIONS[key];
      ctx.strokeStyle = cfg.color;
      ctx.lineWidth = transient ? 2 : 4;
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      ctx.font = "22px Segoe UI";
      ctx.fillStyle = cfg.color;
      ctx.fillText(cfg.label, b.x1 + 4, Math.max(24, b.y1 - 8));
    }
    function redraw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (img.complete) ctx.drawImage(img, 0, 0);
      Object.entries(boxes).forEach(([key, b]) => drawOne(key, b));
      if (current) drawOne(activeRegion, normBox(current), true);
    }
    async function loadManifest() {
      const res = await fetch("/api/manifest");
      manifest = await res.json();
      buildRegions();
      if (!manifest.length) return status("No records need review. Use --show-all to inspect approved/rejected rows.");
      await loadImage(0);
    }
    async function loadImage(next) {
      index = Math.max(0, Math.min(manifest.length - 1, next));
      item = await (await fetch(`/api/item?index=${index}`)).json();
      boxes = item.boxes || {};
      lcdValueEl.value = item.lcd_screen_value || "";
      reviewStatusEl.value = item.review_status || "needs_recheck";
      counterEl.textContent = `Image ${index + 1} / ${manifest.length}`;
      pathEl.textContent = item.relative_image_path || item.image_id;
      const mode = showPreview ? "preview" : "raw";
      img = new Image();
      img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; redraw(); };
      img.onerror = () => {
        if (showPreview) {
          showPreview = false;
          status("DINO preview is missing, showing raw image instead.");
          loadImage(index);
        } else {
          status("Image file could not be loaded.");
        }
      };
      img.src = `/file?index=${index}&mode=${mode}&t=${Date.now()}`;
      status(`Loaded ${mode}.`);
    }
    canvas.addEventListener("mousedown", evt => {
      drawing = true;
      start = imagePoint(evt);
      current = { x1: start.x, y1: start.y, x2: start.x, y2: start.y };
      redraw();
    });
    canvas.addEventListener("mousemove", evt => {
      if (!drawing) return;
      const p = imagePoint(evt);
      current = { x1: start.x, y1: start.y, x2: p.x, y2: p.y };
      redraw();
    });
    window.addEventListener("mouseup", () => {
      if (!drawing) return;
      drawing = false;
      const b = normBox(current);
      if (b) boxes[activeRegion] = b;
      current = null;
      redraw();
    });
    async function save(goNext=false) {
      const payload = {
        row_index: item.row_index,
        review_status: reviewStatusEl.value,
        lcd_screen_value: lcdValueEl.value.trim(),
        boxes,
      };
      const res = await fetch("/api/save", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
      const result = await res.json();
      if (!result.ok) {
        status(`Save failed: ${result.error}`);
        return;
      }
      status(`Saved ${item.image_id}`);
      const nextIndex = goNext ? index + 1 : index;
      const res2 = await fetch("/api/manifest");
      manifest = await res2.json();
      if (!manifest.length) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        counterEl.textContent = "";
        pathEl.textContent = "";
        status("No records need review.");
        return;
      }
      await loadImage(Math.min(nextIndex, manifest.length - 1));
    }
    document.getElementById("prevBtn").onclick = () => loadImage(index - 1);
    document.getElementById("nextBtn").onclick = () => loadImage(index + 1);
    document.getElementById("clearActiveBtn").onclick = () => { delete boxes[activeRegion]; redraw(); };
    document.getElementById("clearAllBtn").onclick = () => { boxes = {}; redraw(); };
    document.getElementById("rawBtn").onclick = () => { showPreview = false; loadImage(index); };
    document.getElementById("previewBtn").onclick = () => { showPreview = true; loadImage(index); };
    document.getElementById("saveBtn").onclick = () => save(false);
    document.getElementById("saveNextBtn").onclick = () => save(true);
    document.addEventListener("keydown", evt => {
      if (["INPUT", "SELECT"].includes(evt.target.tagName)) return;
      if (evt.key === "1") activeRegion = "kazam_box";
      if (evt.key === "2") activeRegion = "multimeter";
      if (evt.key === "3") activeRegion = "lcd_screen";
      if (evt.key === "ArrowLeft") loadImage(index - 1);
      if (evt.key === "ArrowRight") loadImage(index + 1);
      updateRegions();
    });
    loadManifest();
  </script>
</body>
</html>
"""


SERVER_STATE = {"manifest_json": MANIFEST_JSON}
SERVER_STATE["show_all"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, payload, status_code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML.replace("__REGIONS__", json.dumps(REGIONS)))
            return
        if parsed.path == "/api/manifest":
            self.send_json([row_to_payload(row, idx) for idx, row in review_queue(load_rows())])
            return
        if parsed.path == "/api/item":
            rows = load_rows()
            queue = review_queue(rows)
            idx = int(parse_qs(parsed.query).get("index", ["0"])[0])
            if idx < 0 or idx >= len(queue):
                self.send_json({"error": "index out of range"}, 404)
                return
            row_index, row = queue[idx]
            self.send_json(row_to_payload(row, row_index))
            return
        if parsed.path == "/file":
            rows = load_rows()
            queue = review_queue(rows)
            qs = parse_qs(parsed.query)
            idx = int(qs.get("index", ["0"])[0])
            mode = unquote(qs.get("mode", ["raw"])[0])
            if idx < 0 or idx >= len(queue):
                self.send_error(404)
                return
            _, row = queue[idx]
            key = "candidate_preview_path" if mode == "preview" else "copied_image_path"
            path = resolve_allowed_file(row.get(key, ""))
            if path is None:
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
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
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
            rows = load_rows()
            idx = int(payload.get("row_index", payload.get("index")))
            row = rows[idx]
            review_status = payload.get("review_status", "needs_recheck")
            row["review_status"] = review_status
            row["lcd_screen_value"] = payload.get("lcd_screen_value", "")
            row["updated_at"] = datetime.now().isoformat(timespec="seconds")
            boxes = payload.get("boxes", {})
            for key in REGIONS:
                box = normalize_box(boxes.get(key))
                row[f"{key}_bbox_xyxy"] = json.dumps(box) if box else ""
            if review_status == "approved":
                if not row["lcd_screen_value"]:
                    self.send_json({"ok": False, "error": "approved rows require lcd_screen_value"}, 400)
                    return
                missing = [key for key in REGIONS if not parse_box(row.get(f"{key}_bbox_xyxy"))]
                if missing:
                    self.send_json({"ok": False, "error": f"approved rows missing boxes: {', '.join(missing)}"}, 400)
                    return
            if review_status == "rejected":
                delete_candidate_preview(row)
            rows[idx] = row
            write_rows(rows)
            self.send_json({"ok": True, "manifest": str(SERVER_STATE["manifest_json"])})
        except Exception as exc:
            self.send_json({"ok": False, "error": repr(exc)}, 400)


def main() -> None:
    args = parse_args()
    SERVER_STATE["manifest_json"] = args.manifest
    SERVER_STATE["show_all"] = args.show_all
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        print("Run yolo_ROI/run_dino_candidates.py first.")
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), Handler)
    url = f"http://{host}:{args.port}"
    print(f"Manifest: {args.manifest}")
    print(f"Open: {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
