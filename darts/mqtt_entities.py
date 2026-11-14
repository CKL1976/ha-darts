"""Publish the current turn and camera snapshots via MQTT discovery."""

import json
import logging
from threading import Lock

import paho.mqtt.client as mqtt

LOGGER = logging.getLogger(__name__)
BASE = "ha_darts"
SENSORS = {
    "seneste_kast": ("Seneste kast", "last_throw"),
    "antal_pile": ("Antal pile", "count"),
    "tur_score": ("Tur score", "total"),
}
IMAGES = {"left": ("venstre", "Venstre"), "right": ("hoejre", "Højre")}


class DartEntities:
    def __init__(self, options):
        self.lock = Lock()
        self.state = {"last_throw": "—", "count": 0, "total": 0}
        self.images = {}
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
            for key, (name, _) in SENSORS.items():
                self.discover("sensor", key, name, {
                    "state_topic": f"{BASE}/sensor/{key}",
                })
            for side, (key, name) in IMAGES.items():
                self.discover("image", key, name, {"image_topic": f"{BASE}/image/{side}"})
            self.publish_turn()
            for side, jpeg in self.images.items():
                self.publish_image(side, jpeg)

    def discover(self, platform, key, name, config):
        identifier = f"ha_darts_{key}"
        config.update(name=name, unique_id=identifier,
                      default_entity_id=f"{platform}.{identifier}",
                      device={"identifiers": [BASE], "name": "HA Darts"})
        topic = f"{self.discovery_prefix}/{platform}/{identifier}/config"
        self.client.publish(topic, json.dumps(config), qos=1, retain=True)

    def publish_turn(self):
        self.client.publish(f"{BASE}/turn", json.dumps(self.state), qos=1, retain=True)
        for key, (_, field) in SENSORS.items():
            result = self.client.publish(f"{BASE}/sensor/{key}", str(self.state[field]),
                                         qos=1, retain=True)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                LOGGER.warning("MQTT: failed to queue %s (code %s)", key, result.rc)

    def publish_image(self, side, jpeg):
        self.client.publish(f"{BASE}/image/{side}", jpeg, qos=1, retain=True)

    def update_turn(self, darts):
        with self.lock:
            self.state = {
                "last_throw": darts[-1]["field"] if darts else "—",
                "count": len(darts),
                "total": sum(dart["points"] for dart in darts),
            }
            if self.client.is_connected():
                self.publish_turn()

    def update_image(self, side, jpeg):
        with self.lock:
            self.images[side] = jpeg
            if self.client.is_connected():
                self.publish_image(side, jpeg)
