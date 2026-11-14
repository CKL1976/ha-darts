"""Local, bounded capture collection for manually labelled dart tips."""
import io
import json
import math
from pathlib import Path
from threading import Lock
from time import time
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
from flask import Blueprint, Response, abort, render_template, request

diagnostics = Blueprint("diagnostics", __name__)
root = Path("/data/detection-samples")
guard = Lock()
recording = False
last_saved = 0.0
limit = 60


def capture_sample(before, after, prediction, calibration, cameras):
    global last_saved, recording
    with guard:
        if not recording or time() - last_saved < 4:
            return
        root.mkdir(parents=True, exist_ok=True)
        if len(list(root.glob("*/sample.json"))) >= limit:
            recording = False
            return
        sample_id = uuid4().hex
        folder = root / sample_id
        folder.mkdir()
        try:
            for stage, frames in (("before", before), ("after", after)):
                for side, frame in zip(("left", "right"), frames):
                    if not cv2.imwrite(str(folder / f"{side}-{stage}.png"), frame):
                        raise OSError("Kunne ikke gemme kamerabilledet.")
            metadata = {
                "id": sample_id, "created": time(), "prediction": prediction,
                "calibration": calibration, "cameras": cameras,
                "dimensions": {side: [frame.shape[1], frame.shape[0]]
                               for side, frame in zip(("left", "right"), after)},
                "tips": {},
            }
            (folder / "sample.json").write_text(json.dumps(metadata))
            last_saved = time()
        except Exception:
            for file in folder.iterdir():
                file.unlink()
            folder.rmdir()
            raise


def sample_path(sample_id):
    if len(sample_id) != 32 or any(c not in "0123456789abcdef" for c in sample_id):
        abort(404)
    folder = root / sample_id
    if not (folder / "sample.json").exists():
        abort(404)
    return folder


@diagnostics.get("/diagnostics")
def page():
    return render_template("diagnostics.html")


@diagnostics.route("/samples", methods=["GET", "POST"])
def samples():
    global recording, last_saved
    with guard:
        if request.method == "POST":
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or type(data.get("recording")) is not bool:
                abort(400)
            recording = data["recording"]
            last_saved = 0
        entries = [json.loads(file.read_text()) for file in root.glob("*/sample.json")]
        return {"recording": recording, "limit": limit,
                "samples": sorted(entries, key=lambda entry: entry["created"], reverse=True)}, 200, {"Cache-Control": "no-store"}


@diagnostics.get("/samples/<sample_id>/<side>/<stage>")
def picture(sample_id, side, stage):
    if side not in ("left", "right") or stage not in ("before", "after"):
        abort(404)
    return Response((sample_path(sample_id) / f"{side}-{stage}.png").read_bytes(),
                    mimetype="image/png", headers={"Cache-Control": "no-store"})


@diagnostics.post("/samples/<sample_id>/tips")
def label(sample_id):
    tips = request.get_json(silent=True)
    with guard:
        folder = sample_path(sample_id)
        metadata = json.loads((folder / "sample.json").read_text())
        if not isinstance(tips, dict) or set(tips) != {"left", "right"}:
            abort(400)
        for side, point in tips.items():
            width, height = metadata["dimensions"][side]
            if (not isinstance(point, list) or len(point) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
                or not (0 <= point[0] < width and 0 <= point[1] < height)):
                abort(400)
        metadata["tips"] = tips
        temporary = folder / "sample.tmp"
        temporary.write_text(json.dumps(metadata))
        temporary.replace(folder / "sample.json")
    return {"saved": True}


@diagnostics.get("/samples/export")
def export():
    buffer = io.BytesIO()
    with guard, ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for metadata in root.glob("*/sample.json"):
            for file in metadata.parent.iterdir():
                if file.suffix in (".png", ".json"):
                    archive.write(file, str(file.relative_to(root)))
    return Response(buffer.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": "attachment; filename=dart-samples.zip"})
