import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.request import urlopen

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, request
from waitress import serve

app = Flask(__name__)
lock = Lock()
options_path = Path("/data/options.json")
calibration_path = Path("/data/calibration.json")
sides = ("left", "right")
sectors = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5)
reference = None
darts = []


@app.before_request
def ingress_only():
    if request.remote_addr != "172.30.32.2":
        abort(403)


def capture(side):
    options = json.loads(options_path.read_text())
    with urlopen(options[f"{side}_url"], timeout=5) as response:
        data = response.read(8_000_001)
    if len(data) > 8_000_000:
        raise ValueError(f"{side}: Camera response exceeds 8 MB")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"{side}: Camera did not return an image")
    return frame


def capture_pair():
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(capture, sides))


def transform(points):
    source = np.asarray(points, dtype=np.float32)
    if source.shape != (4, 2) or not np.isfinite(source).all():
        raise ValueError("Vælg fire punkter for hvert kamera.")
    if not cv2.isContourConvex(source) or cv2.contourArea(source) < 100:
        raise ValueError("Kalibreringspunkterne ligger forkert.")
    # Outer double-ring intersections, clockwise from the 5/20 wire.
    radians = np.deg2rad((-9, 81, 171, 261))
    target = 170 * np.column_stack((np.sin(radians), -np.cos(radians)))
    return cv2.getPerspectiveTransform(source, target.astype(np.float32))


def dart_line(before, after, mapping):
    if before.shape != after.shape:
        raise ValueError("Kameraets billedstørrelse har ændret sig.")
    delta = cv2.cvtColor(cv2.absdiff(before, after), cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(delta, 35, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)
    if not candidates or cv2.contourArea(candidates[0]) < 150:
        raise ValueError("Kunne ikke finde en ny pil i begge kamerabilleder.")
    contour = candidates[0]
    # L1 fit reduces the influence of the flights on the projected shaft axis.
    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L1, 0, .01, .01).ravel()
    endpoints = np.float32([[[x, y], [x + 50 * vx, y + 50 * vy]]])
    a, b = cv2.perspectiveTransform(endpoints, mapping)[0]
    return a, b - a, mask


def field_at(x, y):
    radius = math.hypot(x, y)
    if radius <= 6.35:
        return "DB", 50
    if radius <= 15.9:
        return "SB", 25
    if radius > 170:
        return "MISS", 0
    angle = math.degrees(math.atan2(x, -y))
    sector = sectors[int((angle + 9) % 360 // 18)]
    factor = 3 if 99 <= radius <= 107 else 2 if radius >= 162 else 1
    return f"{' SDT'[factor]}{sector}", factor * sector


def detect(before, after, mappings):
    a, u, left = dart_line(before[0], after[0], mappings[0])
    b, v, right = dart_line(before[1], after[1], mappings[1])
    system = np.column_stack((u, -v))
    if min(np.linalg.norm(u), np.linalg.norm(v)) == 0 or np.linalg.cond(system) > 12:
        raise ValueError("Kameravinklerne giver et usikkert træfpunkt.")
    hit = a + np.linalg.solve(system, b - a)[0] * u
    if not np.isfinite(hit).all() or np.linalg.norm(hit) > 180:
        raise ValueError("Træfpunktet kunne ikke bekræftes på skiven.")
    for mask, mapping in zip((left, right), mappings):
        point = cv2.perspectiveTransform(hit.reshape(1, 1, 2), np.linalg.inv(mapping))
        x, y = np.rint(point[0, 0]).astype(int)
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            raise ValueError("Træfpunktet ligger uden for et kamerabillede.")
        if not mask[max(0, y-14):y+15, max(0, x-14):x+15].any():
            raise ValueError("Træfpunktet stemmer ikke med ændringen i begge billeder.")
    field, points = field_at(*hit)
    if any(field_at(*(hit + offset))[0] != field
           for offset in ((2, 0), (-2, 0), (0, 2), (0, -2))):
        raise ValueError(f"Usikkert træf tæt på en wire. Bedste bud: {field}.")
    return {"field": field, "points": points}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/image/<side>")
def image(side):
    if side not in sides:
        abort(404)
    try:
        with lock:
            frame = capture(side)
        _, jpeg = cv2.imencode(".jpg", frame)
        return Response(jpeg.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, cv2.error) as exc:
        return str(exc), 502


@app.post("/action/<action>")
def act(action):
    global reference
    with lock:
        try:
            if action == "calibrate":
                data = request.get_json()
                points = {side: data[side] for side in sides}
                for coordinates in points.values():
                    transform(coordinates)
                temp = calibration_path.with_suffix(".tmp")
                temp.write_text(json.dumps(points))
                temp.replace(calibration_path)
                reference = None
                darts.clear()
            elif action == "reset":
                reference = capture_pair()
                darts.clear()
            elif action == "dart":
                if reference is None or not calibration_path.exists():
                    raise ValueError("Kalibrér kameraerne og start med tom skive.")
                if len(darts) == 3:
                    raise ValueError("Turen er slut. Fjern pilene og start en ny tur.")
                points = json.loads(calibration_path.read_text())
                frames = capture_pair()
                darts.append(detect(reference, frames, [transform(points[s]) for s in sides]))
                reference = frames
            else:
                abort(404)
        except (OSError, ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
            return {"error": str(exc)}, 400
        return {"darts": list(darts), "total": sum(d["points"] for d in darts)}


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8099, threads=4)
