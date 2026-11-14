"""DeepDarts dart-tip inference using the Dart-Vision float32 LiteRT export."""
from hashlib import sha256
from urllib.request import urlopen

import numpy as np
from PIL import Image
from scipy.special import expit

MAX_MODEL_BYTES = 64 * 1024 * 1024

D1_URL = ("https://raw.githubusercontent.com/iambhabha/Dart-Vision/master/"
          "deepdarts_models/deepdarts_d1.tflite")
D1_SHA256 = "8bcaa21b54b053649a4b93929a5e97426e8d4f2eeeee3d749ef77a856df91a3b"


def download_d1():
    with urlopen(D1_URL, timeout=20) as response:
        weights = response.read(MAX_MODEL_BYTES + 1)
    if sha256(weights).hexdigest() != D1_SHA256:
        raise ValueError("Den hentede fil matcher ikke den verificerede model.")
    return weights


class DeepDarts:
    def __init__(self, weights):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_content=weights, num_threads=2)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()
        if (len(inputs) != 1 or tuple(inputs[0]["shape"]) != (1, 800, 800, 3)
                or inputs[0]["dtype"] != np.float32
                or sorted(tuple(item["shape"]) for item in self.outputs)
                != [(1, 25, 25, 30), (1, 50, 50, 30)]
                or any(item["dtype"] != np.float32 for item in self.outputs)):
            raise ValueError("Vælg DeepDarts' 800 × 800 float32 TFLite-model med to YOLO-udgange.")
        self.input_index = inputs[0]["index"]

    def tips(self, frame):
        height, width = frame.shape[:2]
        image = Image.fromarray(frame[:, :, ::-1]).resize((800, 800), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)[None] / 255
        self.interpreter.set_tensor(self.input_index, pixels)
        self.interpreter.invoke()
        heads = [self.interpreter.get_tensor(item["index"]) for item in self.outputs]
        return decode_tips(heads, width, height)


def decode_tips(heads, width, height):
    predictions = []
    for head in heads:
        size = head.shape[1]
        cells = head.reshape(size, size, 3, 10)
        scores = expit(cells[..., 4]) * expit(cells[..., 5])
        dart = cells[..., 5] >= np.max(cells[..., 6:], axis=-1)
        rows, columns, anchors = np.where(dart & (scores >= .25))
        offsets = expit(cells[rows, columns, anchors, :2])
        for row, column, anchor, (dx, dy) in zip(rows, columns, anchors, offsets):
            point = np.array([(column + dx) / size, (row + dy) / size], dtype=np.float32)
            predictions.append((float(scores[row, column, anchor]), point))
    accepted = []
    for _, point in sorted(predictions, key=lambda prediction: prediction[0], reverse=True):
        if all(np.linalg.norm(point - earlier) > .0125 for earlier in accepted):
            accepted.append(point)
        if len(accepted) == 12:
            break
    return [point * np.float32([width, height]) for point in accepted]
