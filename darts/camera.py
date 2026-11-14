"""Fetch still images directly from configured HTTP cameras."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
        self.retry_at = 0

    def snapshot(self, preview=False):
        with self.lock:
            now = monotonic()
            if preview and self.frame is not None and (
                now - self.timestamp < 5 or now < self.retry_at
            ):
                return self.frame
            if now < self.retry_at:
                raise ValueError("Kameraet begrænser billedhentning. Prøver igen om lidt.")
            try:
                request = Request(self.url, headers={"Accept": "image/*", "Cache-Control": "no-cache"})
                with opener.open(request, timeout=8) as response:
                    frame = response.read(MAX_FRAME + 1)
            except HTTPError as exc:
                retry_after = exc.headers.get("Retry-After", "")
                exc.close()
                if exc.code == 429:
                    if retry_after.isdecimal():
                        delay = int(retry_after)
                    else:
                        try:
                            retry_time = parsedate_to_datetime(retry_after)
                            delay = (retry_time - datetime.now(timezone.utc)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            delay = 5
                    self.retry_at = monotonic() + max(5, delay)
                    if preview and self.frame is not None:
                        return self.frame
                    raise ValueError("Kameraet begrænser billedhentning. Prøver igen om lidt.") from exc
                raise ValueError(f"Kameraet returnerede HTTP {exc.code}.") from exc
            except (OSError, HTTPException) as exc:
                raise ValueError("Kunne ikke hente stillbilledet fra kameraet.") from exc
            if not frame or len(frame) > MAX_FRAME:
                raise ValueError("Kameraets stillbillede er tomt eller for stort.")
            self.frame = frame
            self.timestamp = monotonic()
            return frame
