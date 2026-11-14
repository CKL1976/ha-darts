import base64
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, sleep

import numpy as np
from flask import Flask, Response, abort, render_template, request
from waitress import serve
from scipy import ndimage
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import least_squares

from diagnostics import diagnostics, capture_sample

from camera import SnapshotCamera
from imaging import (decode_image, encode_image, fit_homography, grayscale,
                     hsv_image, project_points, sample_image, valid_quad)
from mqtt_entities import DartEntities

app = Flask(__name__)
app.register_blueprint(diagnostics)
lock = Lock()
options_path = Path("/data/options.json")
calibration_path = Path("/data/calibration.json")
camera_path = Path("/data/cameras.json")
empty_path = Path("/data/empty-board.npz")
camera_roles = ("left", "right")
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
startup_frames = None
status = "Indtast snapshotadresserne under Kameraer."
darts = []
entities = None
cameras = {}
diagnostic_before = None



@app.before_request
def ingress_only():
    if request.remote_addr != "172.30.32.2":
        abort(403)


def capture(side):
    try:
        data = cameras[side].snapshot()
        frame = decode_image(data)
    except ValueError as exc:
        raise ValueError(f"{side}: {exc}") from exc
    return frame


def capture_pair():
    require_cameras()
    with ThreadPoolExecutor(max_workers=len(camera_roles)) as pool:
        return list(pool.map(capture, camera_roles))


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
    if not valid_quad(source[:4]):
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
        return fit_homography(source, target)
    # Fit in pixel space, where the manual clicks introduce measurement error.
    inverse = fit_homography(target, source)
    if inverse is None or not np.isfinite(inverse).all():
        raise ValueError("Kunne ikke tilpasse kalibreringen.")
    forward = np.linalg.inv(inverse)
    if len(source) == mesh_count:
        return CalibrationMesh(source, target, forward, inverse)
    return forward


class CalibrationMesh:
    def __init__(self, source, target, forward, inverse):
        # Add intermediate points along each curved ring. Interpolate the
        # correction to the perspective projection around the ring, so the
        # outer double stays curved and inside the calibrated region.
        base = project_points(target[None], inverse)[0]
        corrections = (source - base)[1:].reshape(len(mesh_radii), len(sectors), 2)
        extra_targets, extra_sources = [], []
        indices = np.arange(len(sectors))
        for ring, radius in enumerate(mesh_radii):
            for fraction in (.25, .5, .75):
                points = board_points(indices + fraction, radius, -sector_angle / 2)
                projected = project_points(points[None], inverse)[0]
                correction = corrections[ring]
                extra_targets.extend(points)
                extra_sources.extend(projected + (1 - fraction) * correction
                                     + fraction * np.roll(correction, -1, axis=0))
        # Extend the mesh beyond the scoring wire so points in the black
        # surround use the same local coordinate system when scored as OUT.
        outer = board_points(np.arange(0, len(sectors), .25), 190, -sector_angle / 2)
        projected = project_points(outer[None], inverse)[0]
        correction = corrections[-1]
        phase = np.arange(len(outer)) / 4
        index = phase.astype(int)
        fraction = (phase - index)[:, None]
        extra_targets.extend(outer)
        extra_sources.extend(projected + (1 - fraction) * correction[index]
                             + fraction * correction[(index + 1) % len(sectors)])
        source = np.vstack((source, extra_sources)).astype(np.float32)
        target = np.vstack((target, extra_targets)).astype(np.float32)
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
            result[outside] = project_points(
                points[outside].reshape(1, -1, 2), self.fallback[backwards]
            )[0]
        return result.astype(np.float32)


def project(points, mapping, backwards=False):
    if isinstance(mapping, CalibrationMesh):
        return mapping.project(points, backwards)
    matrix = np.linalg.inv(mapping) if backwards else mapping
    return project_points(points, matrix)


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


def axes(mask, mapping):
    # In both mounted camera views the flight is above the tip. Use the
    # silhouette's width along each near-vertical edge to find the taper.
    edges = canny(mask, low_threshold=40, high_threshold=100)
    scale = mask.shape[0] / 1536
    segments = probabilistic_hough_line(edges, threshold=24,
                                       line_length=max(18, round(40 * scale)),
                                       line_gap=max(5, round(12 * scale)), rng=0)
    if not segments:
        return []
    segments = np.asarray(segments)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    result = []
    along = np.arange(-75, 125, 2, dtype=np.float32) * scale
    across = np.arange(-16, 17, dtype=np.float32) * scale
    for index in np.argsort(lengths)[-70:][::-1]:
        top, bottom = sorted(segments[index], key=lambda point: point[1])
        direction = (bottom - top) / lengths[index]
        if direction[1] < .78:
            continue
        normal = np.array([direction[1], -direction[0]])
        grid = bottom + along[:, None, None] * direction + across[None, :, None] * normal
        profile = sample_image(mask, grid, order=0) > 0
        # A barrel has support on several adjacent cross-sections. The
        # support narrows at the point and disappears beyond the board entry.
        width = profile.sum(axis=1)
        central = profile[:, 7:26].any(axis=1)
        for tip_index in range(20, len(along) - 8):
            body = width[tip_index-20:tip_index-6]
            tip_width = width[tip_index-4:tip_index+1]
            if (np.count_nonzero((body >= 5) & (body <= 26)) < 9
                    or np.count_nonzero(central[tip_index-4:tip_index+1]) < 3
                    or np.median(tip_width) > .75 * np.median(body)
                    or np.count_nonzero(central[tip_index+1:tip_index+9]) > 1):
                continue
            tip_columns = np.where(profile[tip_index-4:tip_index+1, 7:26])[1] + 7
            tip = grid[tip_index, 16] + np.median(across[tip_columns]) * normal
            body_center = grid[tip_index-17, 16]
            body_columns = np.where(profile[tip_index-20:tip_index-6])[1]
            start = body_center + np.median(across[body_columns]) * normal
            board_tip = project(np.float32([[tip]]), mapping)[0, 0]
            if np.isfinite(board_tip).all() and field_at(*board_tip)[0] != "MISS":
                result.append((board_tip, start))
    return result


def detect(masks, transforms, diagnostics=None):
    views = [axes(mask, mapping) for mask, mapping in zip(masks, transforms)]
    if diagnostics is not None:
        diagnostics["tip_counts"] = [len(view) for view in views]
    if not all(views):
        return None
    left, right = views
    left_points = np.array([candidate[0] for candidate in left], dtype=np.float32)
    right_points = np.array([candidate[0] for candidate in right], dtype=np.float32)
    distances = np.linalg.norm(left_points[:, None] - right_points, axis=2)
    left_index, right_index = np.unravel_index(distances.argmin(), distances.shape)
    if distances[left_index, right_index] > 9:
        return None
    hit = (left_points[left_index] + right_points[right_index]) / 2
    field, points = field_at(*hit)
    lines_by_view = []
    for mapping, (_, start) in zip(transforms, (left[left_index], right[right_index])):
        pixel = project(hit.reshape(1, 1, 2), mapping, backwards=True)[0, 0]
        lines_by_view.append({"start": start.tolist(), "tip": pixel.tolist()})
    return {"field": field, "points": points,
            "x_mm": round(float(hit[0]), 1), "y_mm": round(float(hit[1]), 1),
            "camera_lines": lines_by_view}


def change_mask(previous, current):
    if previous.shape != current.shape:
        raise ValueError("Kameraets billedstørrelse har ændret sig.")
    delta = np.abs(grayscale(previous).astype(np.int16) - grayscale(current))
    return (delta > 30).astype(np.uint8) * 255


def detect_throw(before, after, mappings):
    masks = [change_mask(previous, current)
             for previous, current in zip(before, after)]
    diagnostics = {}
    hit = detect(masks, mappings, diagnostics)
    try:
        capture_sample(before, after, hit, json.loads(calibration_path.read_text()), camera_selection())
    except (OSError, ValueError):
        app.logger.exception("Kunne ikke gemme detektionsbilleder")
    if hit is not None:
        return hit
    counts = "/".join(map(str, diagnostics["tip_counts"]))
    raise ValueError(f"Fandt ingen pilespidser, der stemmer overens. Kandidater pr. kamera: {counts}.")


@app.post("/samples/capture")
def capture_diagnostic_pair():
    global diagnostic_before
    data = request.get_json(silent=True)
    stage = data.get("stage") if isinstance(data, dict) else None
    if stage not in ("before", "after"):
        return {"error": "Vælg før eller efter kast."}, 400
    with lock:
        try:
            calibration = saved_calibration()
            selection = camera_selection()
            if stage == "after" and (
                diagnostic_before is None
                or diagnostic_before[1:] != (selection, calibration)
            ):
                raise ValueError("Tag et nyt førbillede først.")
            frames = capture_pair()
            if stage == "before":
                diagnostic_before = (frames, selection, calibration)
                return {"saved": True}
            before = diagnostic_before[0]
            if any(first.shape != last.shape for first, last in zip(before, frames)):
                raise ValueError("Billedstørrelsen har ændret sig. Tag et nyt førbillede.")
            sample_id = capture_sample(before, frames, None, calibration, selection, manual=True)
            diagnostic_before = None
            return {"id": sample_id}
        except (OSError, ValueError) as exc:
            return {"error": str(exc)}, 400


def camera_selection():
    return {role: hashlib.sha256(camera.url.encode()).hexdigest()
            for role, camera in cameras.items()}


def require_cameras():
    if not cameras:
        raise ValueError("Vælg og gem de to kameraer først.")


def saved_calibration():
    require_cameras()
    if not calibration_path.exists():
        raise ValueError("Kalibrér kameraerne først.")
    saved = json.loads(calibration_path.read_text())
    previous = saved.get("camera_entities", {})
    if {role: previous.get(role) for role in camera_roles} != camera_selection():
        raise ValueError("Vælg Kalibrér 9 punkter for de valgte kameraer først.")
    return saved


@app.post("/cameras")
def select_cameras():
    global cameras, reference, empty_board, mappings, manual_mappings, pending, clear_frames, startup_frames, status
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return {"error": "Angiv en snapshotadresse til hvert kamera."}, 400
    selected = {role: data.get(role) for role in camera_roles}
    if any(not isinstance(value, str) or not value.strip() for value in selected.values()):
        return {"error": "Angiv en snapshotadresse til hvert kamera."}, 400
    selected = {role: value.strip() for role, value in selected.items()}
    if len(set(selected.values())) != len(camera_roles):
        return {"error": "Vælg to forskellige kameraer."}, 400
    try:
        proposed = {role: SnapshotCamera(url) for role, url in selected.items()}
        def check_camera(role):
            try:
                raw = proposed[role].snapshot()
                decode_image(raw)
            except ValueError as exc:
                name = "Venstre kamera" if role == "left" else "Højre kamera"
                raise ValueError(f"{name}: {exc}") from exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(check_camera, camera_roles))
        with lock:
            if selected != {role: camera.url for role, camera in cameras.items()}:
                temp = camera_path.with_suffix(".tmp")
                temp.write_text(json.dumps(selected))
                temp.replace(camera_path)
                cameras = proposed
                reference = empty_board = mappings = manual_mappings = pending = startup_frames = None
                clear_frames = 0
                darts.clear()
                status = "Kameraer gemt. Kalibrér med tom skive."
                if entities:
                    entities.reset_turn()
        return {"saved": True}
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}, 502


@app.get("/")
def index():
    return render_template(
        "index.html", configured=bool(cameras), selection=camera_selection(),
        camera_urls={role: camera.url for role, camera in cameras.items()},
        board={"sectors": sectors, "bull_inner": bull_inner, "bull_outer": bull_outer,
               "treble_inner": treble_inner, "treble_outer": treble_outer,
               "double_inner": double_inner, "double_outer": double_outer,
               "mesh_radii": mesh_radii},
    )


@app.get("/image/<side>")
def camera_image(side):
    camera = cameras.get(side)
    if camera is None:
        abort(404)
    try:
        image = camera.snapshot(preview=True)
        if not image.startswith(b"\xff\xd8"):
            image = encode_image(decode_image(image), "JPEG")
        return Response(image, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except (OSError, ValueError) as exc:
        return str(exc), 502


def turn_state():
    views = {}
    if mappings is not None:
        for index, side in enumerate(cameras):
            views[side] = [
                {**dart["camera_lines"][index], "field": dart["field"]}
                for dart in darts
            ]
    return {"camera_hits": views, "darts": list(darts), "total": sum(d["points"] for d in darts),
            "status": status,
            "mqtt": entities.connection_status if entities else "Angiv mqtt_host i Konfiguration"}


def board_is_clear(frames, baseline_frames):
    for baseline, frame in zip(baseline_frames, frames):
        if baseline.shape != frame.shape:
            raise ValueError("Kameraets billedstørrelse har ændret sig.")
        delta = grayscale(np.abs(baseline.astype(np.int16) - frame))
        mask = ndimage.binary_opening(delta > 30, structure=np.ones((3, 3)))
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
        if np.any(np.bincount(labels.ravel())[1:] >= 100):
            return False
    return True


def analyze_frames(frames):
    global reference, pending, clear_frames, status
    if board_is_clear(frames, empty_board):
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
    if pending is None and darts and board_is_clear(frames, reference):
        return
    if len(darts) >= 3:
        status = "Tur slut. Fjern pilene."
        return
    hit = detect_throw(reference, frames, mappings)
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
            if frame is None:
                continue
            baseline = manual_mappings[index]
            if isinstance(baseline, CalibrationMesh):
                continue
            proposal = refine_calibration(frame, baseline)
            raw = project_points(probes, np.linalg.inv(baseline))
            updated = project_points(raw, proposal)
            if np.max(np.linalg.norm(updated - probes, axis=2)) <= 1:
                mappings[index] = proposal
        except (ValueError, np.linalg.LinAlgError):
            continue


def calibration_signature():
    return hashlib.sha256(calibration_path.read_bytes()).hexdigest()


def store_empty_board(frames):
    temp = empty_path.with_name("empty-board.tmp.npz")
    np.savez_compressed(temp, signature=calibration_signature(), **dict(zip(camera_roles, frames)))
    temp.replace(empty_path)


def prepare_board(frames, points):
    global reference, empty_board, mappings, manual_mappings, startup_frames, status
    radius = double_inner if points.get("ring") == "inner_double" else double_outer
    if empty_board is None and empty_path.exists():
        try:
            with np.load(empty_path, allow_pickle=False) as saved:
                if saved["signature"].item() == calibration_signature():
                    baseline = [saved[role] for role in camera_roles]
                    if all(a.shape == b.shape for a, b in zip(baseline, frames)):
                        empty_board = baseline
        except (OSError, ValueError, KeyError):
            pass
    if empty_board is None:
        status = "Hold skiven tom, mens appen gør klar til første tur."
        if startup_frames is None or not board_is_clear(frames, startup_frames):
            startup_frames = frames
            return
    elif not board_is_clear(frames, empty_board):
        status = "Fjern pilene for at starte en ny tur."
        return
    manual_mappings = [transform(points[side], radius) for side in camera_roles]
    mappings = manual_mappings.copy()
    if empty_board is None:
        store_empty_board(frames)
        empty_board = frames
    reference = frames
    startup_frames = None
    if entities:
        entities.reset_turn()
    status = "Klar. Kast en pil."


def watch_board():
    global pending, clear_frames, startup_frames, status
    while True:
        sleep(1)
        with lock:
            if not cameras or not calibration_path.exists():
                continue
            try:
                if reference is None:
                    points = saved_calibration()
                    prepare_board(capture_pair(), points)
                else:
                    analyze_frames(capture_pair())
            except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
                pending = None
                startup_frames = None
                clear_frames = 0
                status = str(exc)


@app.get("/state")
def state():
    with lock:
        return turn_state(), 200, {"Cache-Control": "no-store"}


def refine_calibration(frame, initial):
    # Sample in the original pixels at half-millimeter intervals.
    hsv = hsv_image(frame)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    colored = (((hue <= 12) | (hue >= 168) | ((hue >= 36) & (hue <= 95)))
               & (value >= 45))
    chroma = np.where(colored, saturation, 0).astype(np.float32)
    indices = np.arange(0, len(sectors), .25)
    directions = board_points(indices, 1)
    offsets = np.arange(-6, 6.5, .5)
    detailed = isinstance(initial, CalibrationMesh)
    observed, targets = [], []
    for radius, sign in ((double_inner, 1), (double_outer, -1),
                         (treble_inner, 1), (treble_outer, -1)):
        polar = directions[:, None, :] * (radius + offsets[None, :, None])
        xy = project(polar.astype(np.float32).reshape(1, -1, 2), initial, backwards=True).reshape(polar.shape)
        signal = sample_image(chroma, xy, order=1)
        # Compare averaged samples on each side of the expected transition.
        gradient = (signal[:, 4:] + signal[:, 3:-1]
                    - signal[:, 1:-3] - signal[:, :-4]) * (.5 * sign)
        peaks = (gradient - 2 * np.abs(offsets[2:-2])).argmax(axis=1)
        good = gradient[np.arange(len(peaks)), peaks] >= 28
        if any(group.sum() < 3 for group in np.array_split(good, 4)):
            raise ValueError("Ringkanten er for utydelig i en del af billedet.")
        found = directions[good] * (radius + offsets[peaks[good] + 2])[:, None]
        samples = found if detailed else project(found.astype(np.float32)[None], initial, backwards=True)[0]
        observed.extend(samples)
        targets.extend(directions[good] * radius)

    observed = np.float32(observed)
    targets = np.float32(targets)
    fitted, inliers = fit_homography(observed, targets, robust=True)
    if fitted is None or inliers.mean() < .55:
        raise ValueError("Ringkanterne stemmer ikke nok overens. Prøv manuel kalibrering.")

    def error(mapping):
        predicted = project_points(observed[None], mapping)[0]
        # Evaluate both fits against the same samples from every ring and quadrant.
        return np.mean(np.minimum(np.linalg.norm(predicted - targets, axis=1), 4))

    baseline = np.eye(3) if detailed else initial
    return fitted if error(fitted) + .1 < error(baseline) else baseline


def wire_profiles(gray, mapping, points, normals, spoke_start):
    # A steel wire is a narrow ridge or groove with two shoulders. A single
    # brightness step at the edge of a colored segment has only one shoulder.
    offsets = np.arange(-4, 4.25, .25, dtype=np.float32)
    probes = points[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    pixels = project(probes.reshape(1, -1, 2), mapping, backwards=True).reshape(probes.shape)
    signal = sample_image(gray, pixels, order=1)
    center = signal[:, 4:-4]
    left = center - signal[:, :-8]
    right = center - signal[:, 8:]
    contrast = np.maximum(np.minimum(left, right), .75 * np.minimum(-left, -right))
    # The blade between black and white sisal can have an intermediate
    # brightness. In that case its center is between two same-sign edges.
    step = np.minimum(np.abs(left), np.abs(right))
    contrast[spoke_start:] = np.maximum(
        contrast[spoke_start:],
        np.where((left * right)[spoke_start:] < 0, .65 * step[spoke_start:], 0),
    )
    peak = (contrast - 1.8 * np.abs(offsets[4:-4])).argmax(axis=1)
    rows = np.arange(len(points))
    strength = contrast[rows, peak]
    displacement = offsets[peak + 4]
    good = (strength >= 9) & (peak > 0) & (peak < contrast.shape[1] - 1)
    good &= ((pixels[:, :, 0] >= 1) & (pixels[:, :, 0] < gray.shape[1] - 1)
             & (pixels[:, :, 1] >= 1) & (pixels[:, :, 1] < gray.shape[0] - 1)).all(axis=1)
    return displacement, good


def fit_wire_offsets(points, normals, shifts, spread):
    points = points.astype(np.float64)

    def shifted(samples, parameters):
        matrix = np.eye(3) + np.r_[parameters, 0].reshape(3, 3)
        return project_points(
            (samples / double_outer)[None], matrix
        )[0] * double_outer

    def residual(parameters):
        return np.sum((shifted(points, parameters) - points) * normals, axis=1) - shifts

    fit = least_squares(residual, np.zeros(8), loss="soft_l1", f_scale=spread,
                        bounds=(-.16, .16))
    return fit.fun, lambda samples: shifted(samples, fit.x)


def align_ring_bands(frame, initial, directions, target):
    # Locate both edges of each scoring band together before looking for
    # the thin wire. The paired edges distinguish the ring from board texture.
    hsv = hsv_image(frame)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    colored = ((hue <= 12) | (hue >= 168) | ((hue >= 36) & (hue <= 95))) & (value >= 35)
    chroma = np.where(colored, saturation, 0).astype(np.float32)
    offsets = np.arange(-13, 13.5, .5, dtype=np.float32)
    points, normals, shifts = [], [], []
    for inner, outer in ((treble_inner, treble_outer), (double_inner, double_outer)):
        profiles = []
        for radius in (inner, outer):
            probes = directions[:, None, :] * (radius + offsets[None, :, None])
            pixels = project(probes.reshape(1, -1, 2), initial, backwards=True).reshape(probes.shape)
            signal = sample_image(chroma, pixels, order=1)
            profiles.append(signal[:, 4:] - signal[:, :-4])
        strength = np.minimum(profiles[0], -profiles[1])
        peaks = (strength - .7 * np.abs(offsets[2:-2])).argmax(axis=1)
        good = strength[np.arange(len(peaks)), peaks] >= 20
        if any(group.sum() < 7 for group in np.array_split(good, 4)):
            return initial
        points.extend(directions[good] * ((inner + outer) / 2))
        normals.extend(directions[good])
        shifts.extend(offsets[peaks[good] + 2])
    residual, adjust = fit_wire_offsets(np.asarray(points), np.asarray(normals), shifts, 1.3)
    if np.median(np.abs(residual)) > 2 or np.median(np.abs(residual)) + .25 >= np.median(np.abs(shifts)):
        return initial
    anchors = project(adjust(target).astype(np.float32)[None], initial, backwards=True)[0]
    try:
        return transform(anchors)
    except (ValueError, np.linalg.LinAlgError):
        return initial


def fit_steel_wires(frame, initial):
    gray = grayscale(frame).astype(np.float32)
    ring_directions = board_points(
        np.arange(len(sectors)).repeat(5) + np.tile([.2, .35, .5, .65, .8], len(sectors)),
        1, -sector_angle / 2,
    )
    spokes = board_points(range(len(sectors)), 1, -sector_angle / 2)
    tangents = np.column_stack((-spokes[:, 1], spokes[:, 0]))
    # Spokes stop at the outer bull. Skip ring crossings in both profiles.
    spoke_radii = np.array([23, 31, 43, 56, 69, 82, 94, 112, 124, 136, 148, 157])
    ring_points = (np.asarray(mesh_radii, dtype=np.float32)[:, None, None]
                   * ring_directions[None]).reshape(-1, 2)
    spoke_points = (spokes[:, None, :] * spoke_radii[None, :, None]).reshape(-1, 2)
    samples = np.vstack((ring_points, spoke_points)).astype(np.float32)
    normals = np.vstack((np.tile(ring_directions, (len(mesh_radii), 1)),
                         np.repeat(tangents, len(spoke_radii), axis=0)))
    target = np.vstack(([[0, 0]], *[spokes * radius for radius in mesh_radii]))
    initial = align_ring_bands(frame, initial, ring_directions, target)
    start = len(ring_points)
    matching, good = wire_profiles(gray, initial, samples, normals, start)
    # Require nearby samples on the same physical spoke to agree before
    # trusting a ridge. Isolated ridges in sisal often look like wire.
    spoke_shifts = matching[start:].reshape(len(sectors), -1)
    spoke_mask = good[start:].reshape(spoke_shifts.shape)
    neighbors = (np.abs(spoke_shifts[:, :, None] - spoke_shifts[:, None, :]) <= 1.5)
    spoke_mask &= (neighbors & spoke_mask[:, None, :]).sum(axis=2) >= 3
    ring_good = good[:len(ring_points)].reshape(len(mesh_radii), len(sectors), -1)
    spoke_good = good[len(ring_points):].reshape(len(sectors), -1)
    for quadrant in range(4):
        group = slice(5 * quadrant, 5 * (quadrant + 1))
        if ((ring_good[2:, group].sum(axis=(1, 2)) < 5).any()
                or (spoke_good[group].sum(axis=1) >= 3).sum() < 2):
            raise ValueError("Stålwirerne kunne ikke genkendes rundt om hele skiven.")

    residual, adjust = fit_wire_offsets(samples[good], normals[good], matching[good], .7)
    if np.median(np.abs(residual)) > 1.15:
        raise ValueError("Fundne stålwirer stemmer ikke overens.")

    adjusted = adjust(target)
    # Correct residual lens distortion around each intersection using its
    # neighboring ring arcs and the nearby length of the same spoke.
    remaining = np.full(len(samples), np.nan)
    remaining[good] = -residual
    ring_residual = remaining[:len(ring_points)].reshape(len(mesh_radii), len(sectors), -1)
    spoke_residual = remaining[len(ring_points):].reshape(len(sectors), -1)
    for ring, radius in enumerate(mesh_radii):
        for sector in range(len(sectors)):
            radial = np.r_[ring_residual[ring, sector - 1, -3:],
                           ring_residual[ring, sector, :3]]
            radial = radial[np.isfinite(radial)]
            nearby = np.argsort(np.abs(spoke_radii - radius))[:5]
            angular = spoke_residual[sector, nearby]
            angular = angular[np.isfinite(angular)]
            anchor = adjusted[1 + ring * len(sectors) + sector]
            if len(radial) >= 4:
                anchor += np.clip(np.median(radial), -2, 2) * spokes[sector]
            if radius >= treble_inner and len(angular) >= 3:
                anchor += np.clip(np.median(angular), -2, 2) * tangents[sector]
    source = project(adjusted.astype(np.float32)[None], initial, backwards=True)[0]
    transform(source)
    return source.tolist()


@app.post("/auto-calibrate")
def auto_calibrate():
    try:
        with lock:
            saved = saved_calibration()
            frames = capture_pair()
            captured_roles = tuple(cameras)
        radius = double_inner if saved and saved.get("ring") == "inner_double" else double_outer
        proposal, images, warnings = {}, {}, {}
        for side, frame in zip(captured_roles, frames):
            initial = transform(saved[side], radius)
            try:
                proposal[side] = fit_steel_wires(frame, initial)
            except (ValueError, np.linalg.LinAlgError) as exc:
                name = {"left": "Venstre kamera", "right": "Højre kamera"}[side]
                warnings[side] = f"{name}: {exc}"
                continue
            jpeg = encode_image(frame, "JPEG")
            images[side] = base64.b64encode(jpeg).decode("ascii")
        if warnings:
            return {"error": " ".join(warnings.values())}, 422
        return {"points": proposal, "images": images}
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
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
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}, 400


@app.post("/action/<action>")
def act(action):
    global reference, empty_board, mappings, manual_mappings, next_refinement
    global pending, clear_frames, startup_frames, status
    with lock:
        frames = None
        try:
            if action == "calibrate":
                require_cameras()
                data = request.get_json()
                if not isinstance(data, dict) or data.get("camera_entities") != camera_selection():
                    raise ValueError("Kameraerne er ændret. Genindlæs appen og kalibrér igen.")
                points = {side: data[side] for side in cameras}
                for coordinates in points.values():
                    transform(coordinates, radius=double_inner)
                temp = calibration_path.with_suffix(".tmp")
                temp.write_text(json.dumps({**points, "ring": "inner_double",
                                            "camera_entities": camera_selection()}))
                temp.replace(calibration_path)
                reference = empty_board = mappings = manual_mappings = startup_frames = None
                next_refinement = 0
                status = "Kalibrering gemt. Hold skiven tom."
            elif action == "reset":
                points = saved_calibration()
                radius = double_inner if points.get("ring") == "inner_double" else double_outer
                new_mappings = [transform(points[side], radius) for side in cameras]
                frames = capture_pair()
                store_empty_board(frames)
                manual_mappings = new_mappings
                mappings = new_mappings.copy()
                next_refinement = 0
                empty_board = reference = frames
                status = "Klar. Kast en pil."
            else:
                abort(404)
        except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            return {"error": str(exc)}, 400
        pending = None
        clear_frames = 0
        darts.clear()
        if entities:
            entities.reset_turn()
        return turn_state()


if __name__ == "__main__":
    options = json.loads(options_path.read_text())
    if camera_path.exists():
        selected = json.loads(camera_path.read_text())
        try:
            cameras = {role: SnapshotCamera(selected[role]) for role in camera_roles}
            status = "Start med tom skive."
        except (KeyError, ValueError):
            status = "Indtast snapshotadresserne under Kameraer."
    if options.get("mqtt_host"):
        entities = DartEntities(options)
    Thread(target=watch_board, daemon=True).start()
    serve(app, host="0.0.0.0", port=8099, threads=4)
