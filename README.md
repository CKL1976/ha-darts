# HA Darts

Home Assistant OS-app til to GRAN EYE-kameraer. Appen henter stillbilleder via
HTTP og finder den nye pil ud fra forskellen mellem billeder før og efter kastet.

## Installation

Tilføj dette GitHub-repo under Indstillinger → Apps → App-butik → Repositories.
Installér HA Darts. Under Konfiguration kan du ændre de to kameraers URL'er.
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

Detektion er foreløbig knapstyret. Appen viser scoren i sin egen
Home Assistant-visning. MQTT-sensorer og automatisk kastudløsning er endnu
ikke implementeret.
