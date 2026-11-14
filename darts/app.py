import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, Thread
from time import sleep

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, request
from waitress import serve

from camera import CameraFeed
from mqtt_entities import DartEntities

app = Flask(__name__)
lock = Lock()
options_path = Path("/data/options.json")
calibration_path = Path("/data/calibration.json")
sides = ("left", "right")
sectors = (20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5)
sector_angle = 360 / len(sectors)
bull_inner, bull_outer = 6.35, 15.9
treble_inner, treble_outer = 99, 107
double_inner, double_outer = 162, 170
anchor_sectors = (20, 6, 3, 11)
reference = None
empty_board = None
mappings = None
pending = None
clear_frames = 0
status = "Start med tom skive."
darts = []
entities = None
cameras = {}
last_capture = {}


@app.before_request
def ingress_only():
    if request.remote_addr != "172.30.32.2":
        abort(403)


def capture(side):
    try:
        data, timestamp = cameras[side].latest(last_capture.get(side, 0))
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Kameraet returnerede et ulæseligt billede.")
    except ValueError as exc:
        raise ValueError(f"{side}: {exc}") from exc
    last_capture[side] = timestamp
    return frame


def capture_pair():
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(capture, sides))


def board_points(indices, radius, offset=0):
    angles = np.deg2rad(np.asarray(indices) * sector_angle + offset)
    return (radius * np.column_stack((np.sin(angles), -np.cos(angles)))).astype(np.float32)


def calibration_geometry(points, radius):
    source = np.asarray(points, dtype=np.float32)
    if source.shape not in ((4, 2), (9, 2)) or not np.isfinite(source).all():
        raise ValueError("Ugyldige kalibreringspunkter.")
    if not cv2.isContourConvex(source[:4]) or cv2.contourArea(source[:4]) < 100:
        raise ValueError("De fire punkter på dobbeltringen ligger forkert.")
    indices = [sectors.index(sector) for sector in anchor_sectors]
    target = board_points(indices, radius, -sector_angle / 2)
    if len(source) == 9:
        triple = board_points(indices, treble_outer, -sector_angle / 2)
        target = np.vstack((target, triple, np.zeros((1, 2), dtype=np.float32)))
    return source, target


def transform(points, radius=double_outer):
    source, target = calibration_geometry(points, radius)
    if len(source) == 4:
        return cv2.getPerspectiveTransform(source, target)
    # Fit in pixel space, where the manual clicks introduce measurement error.
    inverse, _ = cv2.findHomography(target, source, method=0)
    if inverse is None or not np.isfinite(inverse).all():
        raise ValueError("Kunne ikke tilpasse kalibreringen.")
    return np.linalg.inv(inverse)


def field_at(x, y):
    radius = math.hypot(x, y)
    if radius <= bull_inner:
        return "DB", 50
    if radius <= bull_outer:
        return "SB", 25
    if radius > double_outer:
        return "MISS", 0
    angle = math.degrees(math.atan2(x, -y))
    sector = sectors[int((angle + sector_angle / 2) % 360 // sector_angle)]
    factor = 3 if treble_inner <= radius <= treble_outer else 2 if radius >= double_inner else 1
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


def expected_color(field):
    if field == "DB":
        return "red"
    if field == "SB":
        return "green"
    if field.startswith("S"):
        return "neutral"
    if field.startswith(("D", "T")):
        return "red" if sectors.index(int(field[1:])) % 2 == 0 else "green"
    return None


def surface_color(frame, x, y):
    if not (4 <= x < frame.shape[1] - 4 and 4 <= y < frame.shape[0] - 4):
        return None
    patch = frame[y-4:y+5, x-4:x+5]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    saturated = (saturation >= 90) & (value >= 45)
    votes = {
        "red": saturated & ((hue <= 12) | (hue >= 168)),
        "green": saturated & (hue >= 36) & (hue <= 95),
        "neutral": (saturation <= 50) | (value < 35),
    }
    return next((color for color, pixels in votes.items() if pixels.mean() >= .80), None)


def colors_at(hit, before, inverses):
    colors = []
    for frame, inverse in zip(before, inverses):
        point = cv2.perspectiveTransform(hit.reshape(1, 1, 2), inverse)[0, 0]
        x, y = np.rint(point).astype(int)
        colors.append(surface_color(frame, x, y))
    return colors


def resolve_by_color(fields, hit, before, inverses):
    colors = colors_at(hit, before, inverses)
    if colors[0] is None or colors[0] != colors[1] or "MISS" in fields:
        return None
    matches = [field for field in fields if expected_color(field) == colors[0]]
    return matches[0] if len(matches) == 1 else None


def detect(masks, transforms, diagnostics=None, before=None):
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
    nearby = dict(field_at(*(hit + offset))
                  for offset in ((2, 0), (-2, 0), (0, 2), (0, -2)))
    near_wire = any(candidate != field for candidate in nearby)
    reasons = []
    if alternatives:
        reasons.append("competing_fields")
    if near_wire:
        reasons.append("near_wire")
    # Use color only for local wire ambiguity. A distant competing impact
    # still needs geometric evidence from both views.
    if not reasons and before is not None:
        colors = colors_at(hit, before, inverses)
        if colors[0] == colors[1] and colors[0] in ("red", "green") and (
            colors[0] != expected_color(field)
        ):
            reasons.append("color_mismatch")
    if near_wire and before is not None and all(other in nearby for other in alternatives):
        nearby[field] = points
        resolved = resolve_by_color(nearby, hit, before, inverses)
        if resolved is not None:
            return {**best, "field": resolved, "points": nearby[resolved]}
    if diagnostics is not None:
        diagnostics.update(best=best, alternatives=alternatives,
                           reason=", ".join(reasons) if reasons else "accepted")
    return None if reasons else best


def detect_throw(before, after, mappings, board):
    masks = []
    for previous, current in zip(before, after):
        if previous.shape != current.shape:
            raise ValueError("Kameraets billedstørrelse har ændret sig.")
        delta = cv2.cvtColor(cv2.absdiff(previous, current), cv2.COLOR_BGR2GRAY)
        masks.append(cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1])

    diagnostics = {}
    hit = detect(masks, mappings, diagnostics, before=board)
    if hit is not None:
        return hit

    reason = diagnostics["reason"]
    if "best" in diagnostics:
        message = f"Usikkert kast. Bedste bud: {diagnostics['best']['field']}."
        if diagnostics["alternatives"]:
            message += " Andre mulige felter: " + ", ".join(diagnostics["alternatives"]) + "."
        if "color_mismatch" in reason:
            message += " Begge kameraer viser en farve, der afviger fra skivekortet."
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


@app.get("/image/<side>")
def image(side):
    if side not in sides:
        abort(404)
    try:
        jpeg, _ = cameras[side].latest()
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, cv2.error) as exc:
        return str(exc), 502


def turn_state():
    return {"darts": list(darts), "total": sum(d["points"] for d in darts),
            "status": status,
            "mqtt": entities.connection_status if entities else "Angiv mqtt_host i Konfiguration"}


def board_is_clear(frames):
    for baseline, frame in zip(empty_board, frames):
        if baseline.shape != frame.shape:
            raise ValueError("Kameraets billedstørrelse har ændret sig.")
        delta = cv2.cvtColor(cv2.absdiff(baseline, frame), cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if np.any(stats[1:, cv2.CC_STAT_AREA] >= 100):
            return False
    return True


def analyze_frames(frames):
    global reference, pending, clear_frames, status
    if board_is_clear(frames):
        pending = None
        clear_frames += 1
        if clear_frames >= 2:
            reference = frames
            if darts:
                darts.clear()
            status = "Klar. Kast en pil."
        return

    clear_frames = 0
    if len(darts) >= 3:
        status = "Tur slut. Fjern pilene."
        return
    hit = detect_throw(reference, frames, mappings, empty_board)
    previous = pending
    pending = hit
    if previous is None or previous["field"] != hit["field"] or math.hypot(
        previous["x_mm"] - hit["x_mm"], previous["y_mm"] - hit["y_mm"]
    ) > 6:
        status = f"Bekræfter {hit['field']}."
        return

    darts.append(hit)
    reference = frames
    pending = None
    status = "Tur slut. Fjern pilene." if len(darts) == 3 else "Klar til næste pil."
    if entities:
        entities.update_throw(hit["points"])


def watch_board():
    global pending, clear_frames, status
    while True:
        sleep(1)
        with lock:
            if reference is None:
                continue
            try:
                frames = capture_pair()
                if all(a.shape == b.shape and np.count_nonzero(
                    cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_BGR2GRAY) > 30
                ) < 180 for a, b in zip(reference, frames)) and darts and not clear_frames:
                    pending = None
                    status = "Tur slut. Fjern pilene." if len(darts) == 3 else "Klar til næste pil."
                    continue
                analyze_frames(frames)
            except (OSError, ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                pending = None
                clear_frames = 0
                status = str(exc)


@app.get("/state")
def state():
    with lock:
        return turn_state(), 200, {"Cache-Control": "no-store"}


@app.post("/calibration-preview")
def calibration_preview():
    try:
        data = request.get_json()
        rings = [board_points(np.linspace(0, len(sectors), 181), radius)
                 for radius in (bull_inner, bull_outer, treble_inner, treble_outer,
                                double_inner, double_outer)]
        inside = board_points(range(len(sectors)), bull_outer, -sector_angle / 2)
        outside = board_points(range(len(sectors)), double_outer, -sector_angle / 2)
        spokes = [np.stack((a, b)) for a, b in zip(inside, outside)]
        number_positions = board_points(range(len(sectors)), double_outer + 13)
        labels = {}
        overlay = {}
        errors = {}
        for side in sides:
            inverse = np.linalg.inv(transform(data[side], radius=double_inner))
            source, target = calibration_geometry(data[side], double_inner)
            projected = cv2.perspectiveTransform(target[None], inverse)[0]
            errors[side] = np.round(np.linalg.norm(projected - source, axis=1), 1).tolist()
            overlay[side] = [
                cv2.perspectiveTransform(line[None], inverse)[0].tolist()
                for line in rings + spokes
            ]
            labels[side] = cv2.perspectiveTransform(number_positions[None], inverse)[0].tolist()
        return {**overlay, "errors": errors, "numbers": sectors,
                "labels": labels}
    except (KeyError, TypeError, ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}, 400


@app.post("/action/<action>")
def act(action):
    global reference, empty_board, mappings, pending, clear_frames, status
    with lock:
        frames = None
        try:
            if action == "calibrate":
                data = request.get_json()
                points = {side: data[side] for side in sides}
                for coordinates in points.values():
                    transform(coordinates, radius=double_inner)
                temp = calibration_path.with_suffix(".tmp")
                temp.write_text(json.dumps({**points, "ring": "inner_double"}))
                temp.replace(calibration_path)
                reference = None
                status = "Kalibrering gemt. Start med tom skive."
            elif action == "reset":
                if not calibration_path.exists():
                    raise ValueError("Kalibrér kameraerne først.")
                points = json.loads(calibration_path.read_text())
                radius = double_inner if points.get("ring") == "inner_double" else double_outer
                new_mappings = [transform(points[side], radius) for side in sides]
                frames = capture_pair()
                mappings = new_mappings
                empty_board = reference = frames
                status = "Klar. Kast en pil."
            else:
                abort(404)
        except (OSError, ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
            return {"error": str(exc)}, 400
        pending = None
        clear_frames = 0
        darts.clear()
        return turn_state()


if __name__ == "__main__":
    options = json.loads(options_path.read_text())
    cameras = {side: CameraFeed(options.get(f"{side}_live_url") or options[f"{side}_url"])
               for side in sides}
    if options.get("mqtt_host"):
        entities = DartEntities(options)
    Thread(target=watch_board, daemon=True).start()
    serve(app, host="0.0.0.0", port=8099, threads=4)
