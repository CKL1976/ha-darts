"""Image I/O and geometry for camera snapshots."""
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform


def decode_image(data):
    try:
        with Image.open(BytesIO(data)) as image:
            # Keep the channel order used by saved empty-board references.
            return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Kameraet returnerede et ulæseligt billede.") from exc


def encode_image(frame, format):
    buffer = BytesIO()
    Image.fromarray(frame[:, :, ::-1]).save(buffer, format=format)
    return buffer.getvalue()


def grayscale(frame):
    return np.rint(frame @ np.array([.114, .587, .299], dtype=np.float32)).astype(np.uint8)


def hsv_image(frame):
    hsv = np.asarray(Image.fromarray(frame[:, :, ::-1]).convert("HSV")).copy()
    hsv[:, :, 0] = np.rint(hsv[:, :, 0] * (180 / 255)).astype(np.uint8)
    return hsv


def sample_image(image, coordinates, order):
    return map_coordinates(image, [coordinates[..., 1], coordinates[..., 0]],
                           order=order, mode="grid-constant", prefilter=False)


def project_points(points, matrix):
    projected = points @ matrix[:, :2].T + matrix[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (projected[..., :2] / projected[..., 2:]).astype(points.dtype)


def fit_homography(source, target, *, robust=False):
    if robust:
        model, inliers = ransac((source, target), ProjectiveTransform,
                                min_samples=4, residual_threshold=1.5,
                                max_trials=150, rng=0)
        return (model.params if model is not None else None), inliers
    model = ProjectiveTransform()
    if not model.estimate(source, target) or not np.isfinite(model.params).all():
        raise ValueError("Kunne ikke tilpasse kalibreringen.")
    return model.params


def valid_quad(points):
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    x, y = points.T
    area = abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2
    return area >= 100 and (np.all(turns > 0) or np.all(turns < 0))
