"""Publish the last scored dart via MQTT discovery."""

import json
import logging
from threading import Lock

import paho.mqtt.client as mqtt

LOGGER = logging.getLogger(__name__)
BASE = "ha_darts"
SENSOR = "seneste_kast"
STATE_TOPIC = f"{BASE}/sensor/{SENSOR}"
OBSOLETE = (
    ("sensor", "antal_pile", f"{BASE}/sensor/antal_pile"),
    ("sensor", "tur_score", f"{BASE}/sensor/tur_score"),
    ("image", "venstre", f"{BASE}/image/left"),
    ("image", "hoejre", f"{BASE}/image/right"),
)


class DartEntities:
    def __init__(self, options):
        self.lock = Lock()
        self.last_score = "unknown"
        self.discovery_prefix = options.get("mqtt_discovery_prefix", "homeassistant")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if options.get("mqtt_username"):
            self.client.username_pw_set(options["mqtt_username"], options.get("mqtt_password"))
        self.connection_status = "Forbinder"
        self.client.on_connect = self.on_connect
        self.client.on_connect_fail = self.on_connect_fail
        self.client.on_disconnect = self.on_disconnect
        self.client.enable_logger(LOGGER)
        self.client.connect_async(options["mqtt_host"], options.get("mqtt_port", 1883))
        self.client.loop_start()

    def on_connect_fail(self, client, userdata):
        self.connection_status = "Forbindelse mislykkedes – prøver igen"
        LOGGER.warning("MQTT: broker connection failed")

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connection_status = "Afbrudt"

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self.connection_status = f"Afvist: {reason_code}"
            LOGGER.warning("MQTT connection rejected: %s", reason_code)
            return
        with self.lock:
            self.connection_status = "Forbundet"
            for platform, key, topic in OBSOLETE:
                self.send(self.discovery_topic(platform, key), b"")
                self.send(topic, b"")
            self.send(f"{BASE}/turn", b"")
            self.send(self.discovery_topic("sensor", SENSOR), json.dumps({
                "name": "Seneste kast",
                "unique_id": f"{BASE}_{SENSOR}",
                "default_entity_id": f"sensor.{BASE}_{SENSOR}",
                "state_topic": STATE_TOPIC,
                "device": {"identifiers": [BASE], "name": "HA Darts"},
            }))
            self.send(STATE_TOPIC, self.last_score)

    def discovery_topic(self, platform, key):
        return f"{self.discovery_prefix}/{platform}/{BASE}_{key}/config"

    def send(self, topic, payload):
        result = self.client.publish(topic, payload, qos=1, retain=True)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            LOGGER.warning("MQTT: failed to queue %s (code %s)", topic, result.rc)

    def update_throw(self, points):
        with self.lock:
            self.last_score = str(points)
            if self.client.is_connected():
                self.send(STATE_TOPIC, self.last_score)
