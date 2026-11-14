"""DeepDarts dart-tip inference using the Dart-Vision float32 LiteRT export."""
from hashlib import sha256
from urllib.request import urlopen

import numpy as np
from PIL import Image

MAX_MODEL_BYTES = 64 * 1024 * 1024

MODEL_HASHES = {
    "d1": "8bcaa21b54b053649a4b93929a5e97426e8d4f2eeeee3d749ef77a856df91a3b",
    "d2": "395cde242447371b5fa8fb37af1117de8425513461db6c3f119bcb5efb279975",
}


def download_model(variant):
    expected = MODEL_HASHES[variant]
    url = ("https://raw.githubusercontent.com/iambhabha/Dart-Vision/master/"
           f"deepdarts_models/deepdarts_{variant}.tflite")
    with urlopen(url, timeout=20) as response:
        weights = response.read(MAX_MODEL_BYTES + 1)
    if sha256(weights).hexdigest() != expected:
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
        cells = head.reshape(-1, 10)
        # The verified export already returns normalized box centers
        # and probabilities. Applying sigmoid or grid offsets again corrupts both.
        scores = cells[:, 4] * cells[:, 5]
        dart = np.argmax(cells[:, 5:], axis=1) == 0
        valid = np.all((cells[:, :2] >= 0) & (cells[:, :2] < 1), axis=1)
        for score, point in zip(scores[dart & valid & (scores >= .25)],
                                cells[dart & valid & (scores >= .25), :2]):
            predictions.append((float(score), point))
    accepted = []
    for _, point in sorted(predictions, key=lambda prediction: prediction[0], reverse=True):
        if all(np.linalg.norm(point - earlier) > .0125 for earlier in accepted):
            accepted.append(point)
        if len(accepted) == 12:
            break
    return [point * np.float32([width, height]) for point in accepted]
