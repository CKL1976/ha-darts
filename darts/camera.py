"""Read HTTP JPEG snapshots or a continuous MJPEG response."""

from http.client import HTTPException
from threading import Condition, Thread
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import urlopen

MAX_FRAME = 8_000_000


def jpeg_frames(response):
    content_type = response.headers.get_content_type()
    if content_type != "multipart/x-mixed-replace":
        data = response.read(MAX_FRAME + 1)
        if len(data) > MAX_FRAME or not data.startswith(b"\xff\xd8"):
            raise ValueError("Kameraet returnerede ikke et gyldigt JPEG-billede.")
        yield data
        return

    buffer = bytearray()
    while chunk := response.read1(65536):
        buffer.extend(chunk)
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                del buffer[:-1]
                break
            del buffer[:start]
            end = buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(buffer) > MAX_FRAME:
                    raise ValueError("Kamerabilledet er for stort.")
                break
            if end + 2 > MAX_FRAME:
                raise ValueError("Kamerabilledet er for stort.")
            yield bytes(buffer[:end + 2])
            del buffer[:end + 2]


class CameraFeed:
    def __init__(self, url):
        self.url = url
        self.condition = Condition()
        self.frame = None
        self.updated = 0.0
        self.error = "Venter på kamerabillede."
        Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while True:
            try:
                with urlopen(self.url, timeout=4) as response:
                    for frame in jpeg_frames(response):
                        with self.condition:
                            self.frame = frame
                            self.updated = monotonic()
                            self.condition.notify_all()
            except HTTPError as exc:
                exc.close()
                self.error = f"Kameraet svarede HTTP {exc.code}."
            except (OSError, ValueError, HTTPException) as exc:
                self.error = str(exc)
            sleep(0.8)

    def latest(self, newer_than=0.0):
        with self.condition:
            if not self.condition.wait_for(
                lambda: self.updated > newer_than and monotonic() - self.updated < 3,
                timeout=5,
            ):
                raise ValueError(self.error if self.frame is None else "Kamerabilledet opdateres ikke.")
            return self.frame, self.updated
