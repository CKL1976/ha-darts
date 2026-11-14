import base64
import json
import math
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, sleep

import cv2
import numpy as np
from flask import Flask, Response, abort, render_template, request
from waitress import serve
from scipy.interpolate import LinearNDInterpolator

from camera import CameraFeed, RtspFeed
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
manual_mappings = None
next_refinement = 0
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
    with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
        return list(pool.map(capture, cameras))


def board_points(indices, radius, offset=0):
    angles = np.deg2rad(np.asarray(indices) * sector_angle + offset)
    return (radius * np.column_stack((np.sin(angles), -np.cos(angles)))).astype(np.float32)


mesh_radii = (bull_inner, bull_outer, treble_inner, treble_outer, double_inner, double_outer)
mesh_count = 1 + len(sectors) * len(mesh_radii)


def calibration_geometry(points, radius):
    source = np.asarray(points, dtype=np.float32)
    if source.shape not in ((4, 2), (9, 2), (mesh_count, 2)) or not np.isfinite(source).all():
        raise ValueError("Ugyldige kalibreringspunkter.")
    if len(source) == mesh_count:
        if len(np.unique(source, axis=0)) != mesh_count:
            raise ValueError("To kalibreringspunkter ligger samme sted.")
        return source, np.vstack(([0, 0], *[
            board_points(range(len(sectors)), ring, -sector_angle / 2)
            for ring in mesh_radii
        ])).astype(np.float32)
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
    forward = np.linalg.inv(inverse)
    if len(source) == mesh_count:
        return CalibrationMesh(source, target, forward, inverse)
    return forward


class CalibrationMesh:
    def __init__(self, source, target, forward, inverse):
        try:
            self.forward = LinearNDInterpolator(source, target)
            self.inverse = LinearNDInterpolator(target, source)
        except ValueError as exc:
            raise ValueError("Kalibreringspunkterne danner ikke en gyldig skive.") from exc
        self.fallback = (forward, inverse)
        # Reject swapped wire points or a folded local mapping.
        triangles = self.inverse.tri.simplices
        def orientation(points):
            a, b, c = points[triangles].transpose(1, 0, 2)
            u, v = b - a, c - a
            return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        signs = orientation(source) * orientation(target)
        if not (np.all(signs > 0) or np.all(signs < 0)):
            raise ValueError("Nogle wirepunkter er byttet om. Kontrollér rækkefølgen.")

    def project(self, points, backwards):
        interpolate = self.inverse if backwards else self.forward
        result = interpolate(points)
        outside = ~np.isfinite(result).all(axis=-1)
        if outside.any():
            result[outside] = cv2.perspectiveTransform(
                points[outside].reshape(1, -1, 2), self.fallback[backwards]
            )[0]
        return result.astype(np.float32)


def project(points, mapping, backwards=False):
    if isinstance(mapping, CalibrationMesh):
        return mapping.project(points, backwards)
    matrix = np.linalg.inv(mapping) if backwards else mapping
    return cv2.perspectiveTransform(points, matrix)


def field_at(x, y):
    radius = math.hypot(x, y)
    if radius <= bull_inner:
        return "DB", 50
    if radius <= bull_outer:
        return "SB", 25
    if radius > 225:
        return "MISS", 0
    if radius > double_outer:
        return "OUT", 0
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
        a, b = project(segments[index:index+1].astype(np.float32), transform)[0]
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
        point = project(hit.reshape(1, 1, 2), inverse, backwards=True)[0, 0]
        x, y = np.rint(point).astype(int)
        colors.append(surface_color(frame, x, y))
    return colors


def detect(masks, transforms, diagnostics=None, before=None, near=None):
    lines = [axes(mask, transform) for mask, transform in zip(masks, transforms)]
    if diagnostics is not None:
        diagnostics["line_counts"] = [len(view) for view in lines]

    inverses = transforms
    candidates = {}
    for left, right in combinations(range(len(lines)), 2):
        for a, u, length_a in lines[left]:
            for b, v, length_b in lines[right]:
                system = np.column_stack((u, -v))
                if abs(np.linalg.det(system)) < 0.22:
                    continue
                hit = a + np.linalg.solve(system, b - a)[0] * u
                if near is not None and np.linalg.norm(hit - near) > 18:
                    continue
                field, points = field_at(*hit)
                if field == "MISS":
                    continue
                supporting = set()
                for index, (mask, inverse) in enumerate(zip(masks, inverses)):
                    point = project(hit.reshape(1, 1, 2), inverse, backwards=True)[0, 0]
                    x, y = np.rint(point).astype(int)
                    if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
                        and mask[max(0, y-10):y+11, max(0, x-10):x+11].any()):
                        supporting.add(index)
                if left not in supporting or right not in supporting:
                    continue
                key = tuple(np.rint(hit / 3).astype(int))
                weight = (length_a + length_b) * (1 + .3 * (len(supporting) - 2))
                if key not in candidates or weight > candidates[key][0]:
                    candidates[key] = (weight, hit, field, points)
    if not candidates:
        return None

    def rank(candidate):
        weight, hit, field, _ = candidate
        colors = colors_at(hit, before, inverses) if before is not None else []
        matches = sum(color == expected_color(field) for color in colors)
        return weight * (1 + .10 * matches)

    _, hit, field, points = max(candidates.values(), key=rank)
    return {"field": field, "points": points,
            "x_mm": round(float(hit[0]), 1), "y_mm": round(float(hit[1]), 1)}


def center_hit(mask, mapping):
    # Center Cam is mounted so the tip appears below the barrel.
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if cv2.contourArea(contour) < 80 or height < 20 or max(width, height) > .45 * max(mask.shape):
        return None
    pixels = contour.reshape(-1, 2)
    bottom = pixels[pixels[:, 1] >= pixels[:, 1].max() - 2]
    # Opening can erase the thin steel tip. Follow the raw mask just
    # below the barrel, keeping the search close to its shaft direction.
    vx, vy, cx, cy = cv2.fitLine(contour, cv2.DIST_L2, 0, .01, .01).ravel()
    edge_y = int(bottom[:, 1].max())
    if abs(vy) > .35:
        for row in range(edge_y + 1, min(mask.shape[0], edge_y + min(32, height // 4))):
            expected_x = int(round(cx + (row - cy) * vx / vy))
            left, right = max(0, expected_x - 4), min(mask.shape[1], expected_x + 5)
            if left >= right:
                break
            columns = np.flatnonzero(mask[row, left:right])
            if not len(columns):
                break
            bottom = np.float32([[left + np.median(columns), row]])
    tip = np.float32([[np.median(bottom, axis=0)]])
    hit = project(tip, mapping)[0, 0]
    field, points = field_at(*hit)
    if field == "MISS":
        return None
    return {"field": field, "points": points,
            "x_mm": round(float(hit[0]), 1), "y_mm": round(float(hit[1]), 1),
            "center_pixel": tip[0, 0].tolist()}


def validate_center(hit, masks, transforms, board):
    position = np.float32([hit["x_mm"], hit["y_mm"]])
    # Find real edges before applying the spatial constraint. Cutting the
    # binary masks first introduces straight edges at the crop boundaries.
    verified = detect(masks[:2], transforms[:2], before=board[:2], near=position)
    if verified is None:
        return {**hit, "center_field": hit["field"], "validation": "unconfirmed"}
    return {**verified, "center_pixel": hit["center_pixel"],
            "center_field": hit["field"], "validation": "side_cameras"}


def detect_throw(before, after, mappings, board, primary=None):
    masks = []
    for previous, current in zip(before, after):
        if previous.shape != current.shape:
            raise ValueError("Kameraets billedstørrelse har ændret sig.")
        delta = cv2.cvtColor(cv2.absdiff(previous, current), cv2.COLOR_BGR2GRAY)
        masks.append(cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1])

    if len(masks) == 3:
        if primary is not None:
            return validate_center(primary, masks, mappings, board)
        hit = center_hit(masks[2], mappings[2])
        if hit is None:
            raise ValueError("Venter på et træf fra Center Cam.")
        return hit

    diagnostics = {}
    hit = detect(masks, mappings, diagnostics, before=board)
    if hit is not None:
        return hit

    counts = "/".join(map(str, diagnostics["line_counts"]))
    raise ValueError(
        f"Fandt intet fælles træfpunkt. Linjer pr. kamera: {counts}."
    )


@app.get("/")
def index():
    return render_template(
        "index.html", third_camera="third" in cameras,
        board={"sectors": sectors, "bull_inner": bull_inner, "bull_outer": bull_outer,
               "treble_inner": treble_inner, "treble_outer": treble_outer,
               "double_inner": double_inner, "double_outer": double_outer,
               "mesh_radii": mesh_radii},
    )


@app.get("/image/<side>")
def image(side):
    if side not in cameras:
        abort(404)
    try:
        jpeg, _ = cameras[side].latest()
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, cv2.error) as exc:
        return str(exc), 502


def turn_state():
    views = {}
    if mappings is not None:
        for side, mapping in zip(cameras, mappings):
            markers = []
            for dart in darts:
                position = np.float32([[[dart["x_mm"], dart["y_mm"]]]])
                pixel = project(position, mapping, backwards=True)[0, 0]
                markers.append({"final": pixel.tolist(),
                                "initial": dart.get("center_pixel") if side == "third" else None})
            views[side] = markers
    return {"camera_hits": views, "darts": list(darts), "total": sum(d["points"] for d in darts),
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
            refine_empty_board(frames)
            if darts:
                darts.clear()
                if entities:
                    entities.reset_turn()
            status = "Klar. Kast en pil."
        return

    clear_frames = 0
    if len(frames) == 3 and pending is not None:
        primary = darts[-1]
        corrected = detect_throw(pending, frames, mappings, empty_board, primary=primary)
        darts[-1] = corrected
        pending = None
        if entities and corrected["points"] != primary["points"]:
            entities.update_throw(corrected["points"], len(darts))
        status = "Tur slut. Fjern pilene." if len(darts) == 3 else "Klar til næste pil."
        return
    if len(darts) >= 3:
        status = "Tur slut. Fjern pilene."
        return
    hit = detect_throw(reference, frames, mappings, empty_board)
    if len(frames) == 2:
        previous = pending
        pending = hit
        if previous is None or previous["field"] != hit["field"] or math.hypot(
            previous["x_mm"] - hit["x_mm"], previous["y_mm"] - hit["y_mm"]
        ) > 6:
            status = f"Bekræfter {hit['field']}."
            return

    darts.append(hit)
    pending = reference if len(frames) == 3 else None
    reference = frames
    status = (f"Center Cam: {hit['field']}. Validerer." if pending is not None else
              "Tur slut. Fjern pilene." if len(darts) == 3 else "Klar til næste pil.")
    if entities:
        entities.update_throw(hit["points"], len(darts))


def refine_empty_board(frames):
    global next_refinement
    now = monotonic()
    if now < next_refinement or manual_mappings is None:
        return
    next_refinement = now + 30
    probes = board_points(range(len(sectors)), double_outer)[None]
    for index, frame in enumerate(frames):
        try:
            baseline = manual_mappings[index]
            if isinstance(baseline, CalibrationMesh):
                continue
            proposal = refine_calibration(frame, baseline)
            raw = cv2.perspectiveTransform(probes, np.linalg.inv(baseline))
            updated = cv2.perspectiveTransform(raw, proposal)
            if np.max(np.linalg.norm(updated - probes, axis=2)) <= 1:
                mappings[index] = proposal
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue


def watch_board():
    global pending, clear_frames, status
    while True:
        sleep(1)
        with lock:
            if reference is None:
                continue
            try:
                frames = capture_pair()
                if pending is None and all(a.shape == b.shape and np.count_nonzero(
                    cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_BGR2GRAY) > 30
                ) < 180 for a, b in zip(reference, frames)) and darts and not clear_frames:
                    pending = None
                    status = "Tur slut. Fjern pilene." if len(darts) == 3 else "Klar til næste pil."
                    continue
                analyze_frames(frames)
            except (OSError, ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                if len(cameras) == 2:
                    pending = None
                clear_frames = 0
                status = str(exc)


@app.get("/state")
def state():
    with lock:
        return turn_state(), 200, {"Cache-Control": "no-store"}


def refine_calibration(frame, initial):
    # Sample in the original pixels at half-millimeter intervals.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    colored = (((hue <= 12) | (hue >= 168) | ((hue >= 36) & (hue <= 95)))
               & (value >= 45))
    chroma = np.where(colored, saturation, 0).astype(np.float32)
    indices = np.arange(0, len(sectors), .25)
    directions = board_points(indices, 1)
    offsets = np.arange(-6, 6.5, .5)
    inverse = np.linalg.inv(initial)
    observed, targets = [], []
    for radius, sign in ((double_inner, 1), (double_outer, -1),
                         (treble_inner, 1), (treble_outer, -1)):
        polar = directions[:, None, :] * (radius + offsets[None, :, None])
        xy = cv2.perspectiveTransform(polar.astype(np.float32).reshape(1, -1, 2),
                                      inverse).reshape(polar.shape)
        signal = cv2.remap(chroma, xy[:, :, 0].copy(), xy[:, :, 1].copy(),
                           cv2.INTER_LINEAR)
        # Compare averaged samples on each side of the expected transition.
        gradient = (signal[:, 4:] + signal[:, 3:-1]
                    - signal[:, 1:-3] - signal[:, :-4]) * (.5 * sign)
        peaks = (gradient - 2 * np.abs(offsets[2:-2])).argmax(axis=1)
        good = gradient[np.arange(len(peaks)), peaks] >= 28
        if any(group.sum() < 3 for group in np.array_split(good, 4)):
            raise ValueError("Ringkanten er for utydelig i en del af billedet.")
        found = directions[good] * (radius + offsets[peaks[good] + 2])[:, None]
        raw = cv2.perspectiveTransform(found.astype(np.float32)[None], inverse)[0]
        observed.extend(raw)
        targets.extend(directions[good] * radius)

    observed = np.float32(observed)
    targets = np.float32(targets)
    fitted, inliers = cv2.findHomography(observed, targets, cv2.RANSAC, 1.5)
    if fitted is None or inliers.mean() < .55:
        raise ValueError("Ringkanterne stemmer ikke nok overens. Prøv manuel kalibrering.")

    def error(mapping):
        predicted = cv2.perspectiveTransform(observed[None], mapping)[0]
        # Evaluate both fits against the same samples from every ring and quadrant.
        return np.mean(np.minimum(np.linalg.norm(predicted - targets, axis=1), 4))

    return fitted if error(fitted) + .1 < error(initial) else initial


@app.post("/auto-calibrate")
def auto_calibrate():
    try:
        with lock:
            frames = capture_pair()
            saved = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
        if saved is None:
            raise ValueError("Gem først en manuel kalibrering, så appen kender skivens retning.")
        if any(side not in saved for side in cameras):
            raise ValueError("Kalibrér alle tilsluttede kameraer manuelt én gang først.")
        radius = double_inner if saved and saved.get("ring") == "inner_double" else double_outer
        proposal, images = {}, {}
        for side, frame in zip(cameras, frames):
            initial = transform(saved[side], radius)
            if isinstance(initial, CalibrationMesh):
                raise ValueError("Detaljeret kalibrering rettes punkt for punkt i den manuelle guide.")
            try:
                fitted = refine_calibration(frame, initial)
            except ValueError as exc:
                name = {"left": "Venstre kamera", "right": "Højre kamera", "third": "Center Cam"}[side]
                raise ValueError(f"{name}: {exc}") from exc
            anchors = np.vstack((
                board_points([sectors.index(n) for n in anchor_sectors], double_inner, -sector_angle / 2),
                board_points([sectors.index(n) for n in anchor_sectors], treble_outer, -sector_angle / 2),
                [[0, 0]],
            )).astype(np.float32)
            proposal[side] = cv2.perspectiveTransform(anchors[None], np.linalg.inv(fitted))[0].tolist()
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("Kunne ikke vise kalibreringsbilledet.")
            images[side] = base64.b64encode(jpeg.tobytes()).decode("ascii")
        return {"points": proposal, "images": images}
    except (OSError, ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}, 400


@app.post("/calibration-preview")
def calibration_preview():
    try:
        data = request.get_json()
        rings = [board_points(np.linspace(0, len(sectors), 181), radius)
                 for radius in (bull_inner, bull_outer, treble_inner, treble_outer,
                                double_inner, double_outer)]
        inside = board_points(range(len(sectors)), bull_outer, -sector_angle / 2)
        outside = board_points(range(len(sectors)), double_outer, -sector_angle / 2)
        spokes = [np.linspace(a, b, 51, dtype=np.float32) for a, b in zip(inside, outside)]
        number_positions = board_points(range(len(sectors)), double_outer + 13)
        labels = {}
        overlay = {}
        errors = {}
        for side in cameras:
            mapping = transform(data[side], radius=double_inner)
            source, target = calibration_geometry(data[side], double_inner)
            projected = project(target[None], mapping, backwards=True)[0]
            errors[side] = np.round(np.linalg.norm(projected - source, axis=1), 1).tolist()
            overlay[side] = [
                project(line[None], mapping, backwards=True)[0].tolist()
                for line in rings + spokes
            ]
            labels[side] = project(number_positions[None], mapping, backwards=True)[0].tolist()
        return {**overlay, "errors": errors, "numbers": sectors,
                "labels": labels}
    except (KeyError, TypeError, ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}, 400


@app.post("/action/<action>")
def act(action):
    global reference, empty_board, mappings, manual_mappings, next_refinement
    global pending, clear_frames, status
    with lock:
        frames = None
        try:
            if action == "calibrate":
                data = request.get_json()
                points = {side: data[side] for side in cameras}
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
                if any(side not in points for side in cameras):
                    raise ValueError("Kalibrér alle tilsluttede kameraer først.")
                new_mappings = [transform(points[side], radius) for side in cameras]
                frames = capture_pair()
                manual_mappings = new_mappings
                mappings = new_mappings.copy()
                next_refinement = 0
                empty_board = reference = frames
                status = "Klar. Kast en pil."
            else:
                abort(404)
        except (OSError, ValueError, KeyError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
            return {"error": str(exc)}, 400
        pending = None
        clear_frames = 0
        darts.clear()
        if entities:
            entities.reset_turn()
        return turn_state()


if __name__ == "__main__":
    options = json.loads(options_path.read_text())
    cameras = {side: CameraFeed(options.get(f"{side}_live_url") or options[f"{side}_url"])
               for side in sides}
    center_cam_url = options.get("center_cam") or options.get("third_camera_url")
    if center_cam_url:
        cameras["third"] = RtspFeed(center_cam_url)
    if options.get("mqtt_host"):
        entities = DartEntities(options)
    Thread(target=watch_board, daemon=True).start()
    serve(app, host="0.0.0.0", port=8099, threads=4)
