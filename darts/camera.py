"""Discover camera entities and fetch snapshots through Home Assistant Core."""

import json
import os
import re
from http.client import HTTPException
from threading import Lock
from time import monotonic
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

CORE_API = "http://supervisor/core/api"
MAX_FRAME = 8_000_000
CAMERA_ID = re.compile(r"camera\.[a-z0-9_]+")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = build_opener(NoRedirect())


def core_get(path, timeout):
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        raise ValueError("Supervisor-token mangler. Start HA Darts som en Home Assistant-app.")
    request = Request(CORE_API + path, headers={"Authorization": f"Bearer {token}"})
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read(MAX_FRAME + 1)
    except HTTPError as exc:
        exc.close()
        raise ValueError(f"Home Assistant svarede HTTP {exc.code}.") from exc
    except (OSError, HTTPException) as exc:
        raise ValueError("Home Assistant returnerede ikke et svar i tide.") from exc
    if len(data) > MAX_FRAME:
        raise ValueError("Svaret fra Home Assistant er for stort.")
    return data


def camera_entities():
    states = json.loads(core_get("/states", timeout=5))
    entities = [
        {"id": state["entity_id"], "name": state["attributes"].get("friendly_name") or state["entity_id"],
         "available": state["state"] != "unavailable"}
        for state in states if CAMERA_ID.fullmatch(state["entity_id"])
    ]
    return sorted(entities, key=lambda camera: (camera["name"].casefold(), camera["id"]))


class EntityCamera:
    def __init__(self, entity_id):
        if not CAMERA_ID.fullmatch(entity_id):
            raise ValueError("Ugyldigt kamera-id.")
        self.entity_id = entity_id
        self.lock = Lock()
        self.frame = None
        self.timestamp = 0

    def snapshot(self, preview=False):
        with self.lock:
            if preview and self.frame is not None and monotonic() - self.timestamp < 2:
                return self.frame
            self.frame = core_get("/camera_proxy/" + self.entity_id, timeout=8)
            self.timestamp = monotonic()
            return self.frame
