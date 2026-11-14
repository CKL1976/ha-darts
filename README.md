# HA Darts

Home Assistant OS-app til to GRAN EYE-kameraer. Appen henter stillbilleder via
HTTP og finder den nye pil ud fra forskellen mellem billeder før og efter kastet.

## Installation

Tilføj dette GitHub-repo under Indstillinger → Apps → App-butik → Repositories.
Installér HA Darts. Under Konfiguration skal du erstatte eksempeladresserne med dine to kameraers URL'er, inklusive /picture.
Start appen og åbn webgrænsefladen.

## Første tur

1. Tøm skiven, og vælg Kalibrér. Vælg krydset på dobbeltringens
   **inderste** wire ved 5/20, 13/6, 17/3 og 8/11. Sæt X i zoomvinduet
   dér, hvor rød/grøn møder sort/hvid. Bekræft krydset for begge kameraer.
2. Vælg Start tur med tom skive. Ældre kalibreringer bruger stadig den
   yderste wire, indtil du gennemfører en ny kalibrering.
3. Kast én pil, og vent på scoren, før du kaster igen. Appen analyserer
   automatisk og tæller op til tre pile, også når webvisningen er lukket.
4. Fjern alle pile. Appen starter næste tur, når den genkender den tomme
   skive i begge kameraer. Efter genstart skal du starte med tom skive igen.

## MQTT-entiteter

Angiv brokerens adresse i `mqtt_host` samt port og eventuelt login under
appens Konfiguration. MQTT-integrationen i Home Assistant skal være på den
samme broker og bruge samme discovery-prefix. Lad `mqtt_host` stå tom for
at køre appen uden MQTT.

Appen registrerer tre sensorer og to billedentiteter under enheden HA Darts:

- `sensor.ha_darts_seneste_kast`
- `sensor.ha_darts_antal_pile`
- `sensor.ha_darts_tur_score`
- `image.ha_darts_venstre`
- `image.ha_darts_hoejre`

Scoren opdateres ved godkendte kast og nulstilling. Billeder sendes ved
billedhentning, turstart og godkendte kast. Brokeren gemmer de seneste
billeder på de to MQTT-emner, så de kan vises efter genstart af Home Assistant.
