# HA Darts

Home Assistant OS-app til to GRAN EYE-kameraer. Appen henter stillbilleder via
HTTP og finder den nye pil ud fra forskellen mellem billeder før og efter kastet.

## Installation

Tilføj dette GitHub-repo under Indstillinger → Apps → App-butik → Repositories.
Installér HA Darts. Under Konfiguration skal du erstatte eksempeladresserne med dine to kameraers URL'er, inklusive /picture.
Start appen og åbn webgrænsefladen.

## Første tur

1. Vælg Kalibrér. Brug én pil som markør ved de fire angivne kryds på
   dobbeltringens yderste wire. Hent billeder ved hvert punkt og klik nær spidsen
   i hvert billede. Klik præcist på kontaktpunktet i zoomvinduet og vælg
   Bekræft punkt for hvert kamera. Du kan flytte markøren før bekræftelse. Hold kameraerne fast under brug.
2. Fjern pilen og vælg Start tur med tom skive.
3. Kast én pil, og vælg Analysér ny pil. Vent på resultatet, før du kaster igen.
   Gentag op til tre gange. Ved usikkert resultat vises en fejl, og den
   sidst godkendte score bevares.
4. Fjern alle pile inden næste tur.

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
Detektion er knapstyret.
