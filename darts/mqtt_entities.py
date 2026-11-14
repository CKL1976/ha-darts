"""Publish the last scored dart via MQTT discovery."""

import json
import logging
from threading import Lock

import paho.mqtt.client as mqtt

LOGGER = logging.getLogger(__name__)
BASE = "ha_darts"
PIL_KEYS = ("pil_1", "pil_2", "pil_3")
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
        self.last_score = None
        self.turn_scores = {}
        self.pending_score = None
        self.discovery_prefix = options.get("mqtt_discovery_prefix", "homeassistant")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if options.get("mqtt_username"):
            self.client.username_pw_set(options["mqtt_username"], options.get("mqtt_password"))
        self.connection_status = "Forbinder"
        self.client.on_connect = self.on_connect
        self.client.on_connect_fail = self.on_connect_fail
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish
        self.client.on_message = self.on_message
        self.client.enable_logger(LOGGER)
        self.client.connect_async(options["mqtt_host"], options.get("mqtt_port", 1883))
        self.client.loop_start()

    def on_connect_fail(self, client, userdata):
        self.connection_status = "Forbindelse mislykkedes – prøver igen"
        LOGGER.warning("MQTT: broker connection failed")

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        with self.lock:
            self.pending_score = None
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
            client.subscribe(f"{self.discovery_prefix}/status", qos=1)
            self.announce_sensor()

    def announce_sensor(self):
        self.send(self.discovery_topic("sensor", SENSOR), json.dumps({
            "name": "Seneste kast",
            "unique_id": f"{BASE}_{SENSOR}",
            "default_entity_id": f"sensor.{BASE}_{SENSOR}",
            "state_topic": STATE_TOPIC,
            "value_template": "{{ value | int }}",
            "device": {"identifiers": [BASE], "name": "HA Darts"},
        }))
        for number, key in enumerate(PIL_KEYS, 1):
            self.send(self.discovery_topic("sensor", key), json.dumps({
                "name": f"Pil {number}",
                "unique_id": f"{BASE}_{key}",
                "default_entity_id": f"sensor.{BASE}_{key}",
                "state_topic": f"{BASE}/sensor/{key}",
                "unit_of_measurement": "point",
                "value_template": "{{ value | int if value != 'None' else none }}",
                "device": {"identifiers": [BASE], "name": "HA Darts"},
            }))
            self.send_slot(key)
        if self.last_score is not None:
            self.publish_score()

    def on_message(self, client, userdata, message):
        if message.payload == b"online":
            with self.lock:
                self.announce_sensor()

    def on_publish(self, client, userdata, mid, reason_code, properties):
        with self.lock:
            if self.pending_score is None or self.pending_score[0] != mid:
                return
            _, score = self.pending_score
            self.pending_score = None
            if reason_code.is_failure:
                self.connection_status = f"Point afvist: {reason_code}"
            else:
                self.connection_status = f"Broker kvitterede for {score} point"

    def publish_score(self):
        result = self.send(STATE_TOPIC, self.last_score)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            self.pending_score = (result.mid, self.last_score)
            self.connection_status = f"Afventer kvittering for {self.last_score} point"
        else:
            self.connection_status = f"Kunne ikke sende point (kode {result.rc})"

    def discovery_topic(self, platform, key):
        return f"{self.discovery_prefix}/{platform}/{BASE}_{key}/config"

    def send(self, topic, payload):
        result = self.client.publish(topic, payload, qos=1, retain=True)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            LOGGER.warning("MQTT: failed to queue %s (code %s)", topic, result.rc)
        return result

    def send_slot(self, key):
        self.send(f"{BASE}/sensor/{key}", self.turn_scores.get(key, "None"))

    def reset_turn(self):
        with self.lock:
            self.turn_scores.clear()
            if self.client.is_connected():
                for key in PIL_KEYS:
                    self.send_slot(key)

    def update_throw(self, points, number):
        with self.lock:
            self.last_score = str(points)
            key = PIL_KEYS[number - 1]
            self.turn_scores[key] = self.last_score
            if self.client.is_connected():
                self.send_slot(key)
                self.publish_score()
