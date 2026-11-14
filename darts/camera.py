"""Fetch still images directly from configured HTTP cameras."""

from http.client import HTTPException
from threading import Lock
from time import monotonic
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_FRAME = 8_000_000


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = build_opener(NoRedirect())


class SnapshotCamera:
    def __init__(self, url):
        try:
            parts = urlsplit(url)
            valid = (parts.scheme in ("http", "https") and parts.hostname
                     and parts.username is None and parts.password is None
                     and not parts.fragment and not any(c.isspace() for c in url))
            parts.port  # Validate the port before attempting a connection.
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Angiv en gyldig HTTP- eller HTTPS-adresse til et stillbillede.")
        self.url = url
        self.lock = Lock()
        self.frame = None
        self.timestamp = 0

    def snapshot(self, preview=False):
        with self.lock:
            if preview and self.frame is not None and monotonic() - self.timestamp < 2:
                return self.frame
            try:
                request = Request(self.url, headers={"Accept": "image/*", "Cache-Control": "no-cache"})
                with opener.open(request, timeout=8) as response:
                    frame = response.read(MAX_FRAME + 1)
            except HTTPError as exc:
                exc.close()
                raise ValueError(f"Kameraet returnerede HTTP {exc.code}.") from exc
            except (OSError, HTTPException) as exc:
                raise ValueError("Kunne ikke hente stillbilledet fra kameraet.") from exc
            if not frame or len(frame) > MAX_FRAME:
                raise ValueError("Kameraets stillbillede er tomt eller for stort.")
            self.frame = frame
            self.timestamp = monotonic()
            return frame
