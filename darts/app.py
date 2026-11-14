import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, request
from waitress import serve

from mqtt_entities import DartEntities

app = Flask(__name__)
lock = Lock()
options_path = Path("/data/options.json")
calibration_path = Path("/data/calibration.json")
sides = ("left", "right")
sectors = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5)
reference = None
darts = []
entities = None


@app.before_request
def ingress_only():
    if request.remote_addr != "172.30.32.2":
        abort(403)


def capture(side):
    options = json.loads(options_path.read_text())
    for attempt in range(2):
        try:
            with urlopen(options[f"{side}_url"], timeout=5) as response:
                data = response.read(8_000_001)
            break
        except HTTPError as exc:
            exc.close()
            if exc.code != 404 or attempt == 1:
                raise ValueError(f"{side}: Kameraet svarede HTTP {exc.code}.") from exc
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


def axes(mask, transform):
    # Changed straight edges generate candidates; matching views constrain the impact.
    edges = cv2.Canny(mask, 40, 100)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, 32,
                               minLineLength=55, maxLineGap=15)
    if segments is None:
        return []
    segments = segments.reshape(-1, 2, 2)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    result = []
    for index in np.argsort(lengths)[-14:][::-1]:
        a, b = cv2.perspectiveTransform(segments[index:index+1].astype(np.float32),
                                        transform)[0]
        direction = b - a
        norm = np.linalg.norm(direction)
        if norm > 0.01 and np.isfinite(norm):
            result.append((a, direction / norm, float(lengths[index])))
    return result


def detect(masks, transforms, diagnostics=None):
    left_axes, right_axes = [axes(mask, transform)
                            for mask, transform in zip(masks, transforms)]
    if diagnostics is not None:
        diagnostics["changed_pixels"] = [cv2.countNonZero(mask) for mask in masks]
        diagnostics["line_counts"] = [len(left_axes), len(right_axes)]

    inverses = [np.linalg.inv(transform) for transform in transforms]
    candidates = {}
    for a, u, length_a in left_axes:
        for b, v, length_b in right_axes:
            system = np.column_stack((u, -v))
            if abs(np.linalg.det(system)) < 0.22:
                continue
            distance, _ = np.linalg.solve(system, b - a)
            hit = a + distance * u
            if np.linalg.norm(hit) > 185:
                continue
            # The impact should be near changed pixels in both original images.
            for mask, inverse in zip(masks, inverses):
                point = cv2.perspectiveTransform(hit.reshape(1, 1, 2),
                                                 inverse)[0, 0]
                x, y = np.rint(point).astype(int)
                if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                    break
                if not np.any(mask[max(0, y-10):y+11, max(0, x-10):x+11]):
                    break
            else:
                field, points = field_at(*hit)
                if field != "MISS":
                    key = tuple(np.rint(hit / 3).astype(int))
                    weight = length_a + length_b
                    if key not in candidates or weight > candidates[key][0]:
                        candidates[key] = (weight, hit, field, points)
    if not candidates:
        if diagnostics is not None:
            diagnostics["reason"] = ("no_lines_in_one_or_both_views"
                                     if not left_axes or not right_axes
                                     else "no_supported_intersection")
        return None
    ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
    weight, hit, field, points = ranked[0]
    best = {"field": field, "points": points,
            "x_mm": round(float(hit[0]), 1), "y_mm": round(float(hit[1]), 1)}
    alternatives = list(dict.fromkeys(
        item[2] for item in ranked[1:] if item[2] != field and item[0] >= weight * .78
    ))
    near_wire = any(field_at(*(hit + offset))[0] != field
                    for offset in ((2, 0), (-2, 0), (0, 2), (0, -2)))
    reasons = []
    if alternatives:
        reasons.append("competing_fields")
    if near_wire:
        reasons.append("near_wire")
    if diagnostics is not None:
        diagnostics.update(best=best, alternatives=alternatives,
                           reason=", ".join(reasons) if reasons else "accepted")
    return None if reasons else best


def detect_throw(before, after, mappings):
    masks = []
    for previous, current in zip(before, after):
        if previous.shape != current.shape:
            raise ValueError("Kameraets billedstørrelse har ændret sig.")
        delta = cv2.cvtColor(cv2.absdiff(previous, current), cv2.COLOR_BGR2GRAY)
        masks.append(cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1])

    diagnostics = {}
    hit = detect(masks, mappings, diagnostics)
    if hit is not None:
        return hit

    app.logger.warning("Kast afvist: %s", json.dumps(diagnostics))
    reason = diagnostics["reason"]
    if "best" in diagnostics:
        message = f"Usikkert kast. Bedste bud: {diagnostics['best']['field']}."
        if diagnostics["alternatives"]:
            message += " Andre mulige felter: " + ", ".join(diagnostics["alternatives"]) + "."
        if "near_wire" in reason:
            message += " Træfpunktet ligger tæt på en wire."
        raise ValueError(message)

    left, right = diagnostics["line_counts"]
    raise ValueError(
        f"Fandt intet fælles træfpunkt. Linjer i venstre/højre billede: {left}/{right}."
    )


@app.get("/")
def index():
    return render_template("index.html")


def encode_image(frame):
    success, jpeg = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Kunne ikke kode kamerabilledet.")
    return jpeg.tobytes()


@app.get("/image/<side>")
def image(side):
    if side not in sides:
        abort(404)
    try:
        with lock:
            frame = capture(side)
            jpeg = encode_image(frame)
            if entities:
                entities.update_image(side, jpeg)
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, cv2.error) as exc:
        return str(exc), 502


@app.post("/action/<action>")
def act(action):
    global reference
    with lock:
        frames = None
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
                frames = capture_pair()
                reference = frames
                darts.clear()
            elif action == "dart":
                if reference is None or not calibration_path.exists():
                    raise ValueError("Kalibrér kameraerne og start med tom skive.")
                if len(darts) == 3:
                    raise ValueError("Turen er slut. Fjern pilene og start en ny tur.")
                points = json.loads(calibration_path.read_text())
                frames = capture_pair()
                darts.append(detect_throw(reference, frames, [transform(points[s]) for s in sides]))
                reference = frames
            else:
                abort(404)
        except (OSError, ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
            return {"error": str(exc)}, 400
        if entities:
            entities.update_turn(darts)
            if frames is not None:
                for side, frame in zip(sides, frames):
                    try:
                        entities.update_image(side, encode_image(frame))
                    except (ValueError, cv2.error):
                        app.logger.warning("Kunne ikke publicere kamerabillede fra %s", side)
        return {"darts": list(darts), "total": sum(d["points"] for d in darts)}


if __name__ == "__main__":
    options = json.loads(options_path.read_text())
    if options.get("mqtt_host"):
        entities = DartEntities(options)
    serve(app, host="0.0.0.0", port=8099, threads=4)
