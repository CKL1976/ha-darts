# HA Darts

Home Assistant OS-app til to GRAN EYE-kameraer. Appen henter stillbilleder via
HTTP og finder den nye pil ud fra forskellen mellem billeder før og efter kastet.

## Installation

Tilføj dette GitHub-repo under Indstillinger → Apps → App-butik → Repositories.
Installér HA Darts. Under Konfiguration skal du erstatte eksempeladresserne med dine to kameraers URL'er, inklusive /picture.
Start appen og åbn webgrænsefladen.

## Første tur

1. Tøm skiven, og vælg Kalibrér. Markér de fire kryds på dobbeltringens
   inderste wire, de samme fire sektorgrænser på tripelringens yderste wire
   og centrum af bull. Ret valgfri punkter i zoomvisningen før du gemmer.
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
at køre appen uden MQTT. Forbindelsesstatus vises i appens webvisning.
Enheden HA Darts har fire MQTT-sensorer:

- `sensor.ha_darts_seneste_kast` viser point for det seneste godkendte kast.
- `sensor.ha_darts_pil_1`, `sensor.ha_darts_pil_2` og `sensor.ha_darts_pil_3`
  viser point for hvert godkendt kast i turen. De nulstilles til ukendt ved
  næste tur og viser først et tal, når den pågældende pil er registreret.

De tidligere billed-, antal- og totalsensorer fjernes via MQTT Discovery
ved forbindelse til samme broker og discovery-prefix.

## Livekameraer

Appen holder en HTTP MJPEG-forbindelse åben pr. kamera og genbruger det
seneste billede til visning og detektion. Et almindeligt JPEG-endpoint
hentes løbende. Hvis kameraets MJPEG-feed bruger en anden sti end /picture,
angiv den i `left_live_url` og `right_live_url`. Kalibrering fryser billederne,
så krydsene kan markeres på samme billedpar.

## Farve ved wirekryds

Ved et træf tæt på en wire kan appen bruge farven i billedet af den tomme
skive til at vælge mellem nærliggende felter. Begge kameraer skal vise
samme tydelige farve, og feltet skal være entydigt blandt kandidaterne.

## Skivekort

Kalibreringen og kastdetektionen bruger samme turneringslayout som Unicorn
Eclipse Pro 2. Før du gemmer, vises de 20 tal og wirerne på begge billeder.
Kontrollér især 20 øverst, 6 til højre, 3 nederst og 11 til venstre på den
fysiske skive. Ved tydelig farveuoverensstemmelse i begge kameraer markeres
kastet som usikkert.

## Automatisk finjustering

Tøm skiven og vælg Auto-kalibrér tom skive. En gemt manuel kalibrering
giver startposition og retning. Kontrollér forslaget i begge billeder,
og gem, hvis overlejringen passer. Kastdetektionen scorer den bedst
rangerede kandidat, når den bekræftes i to aflæsninger. MQTT-status viser
brokerens kvittering for pointværdien.

Tilføj eventuelt en RTSP-adresse i `third_camera_url` i appens konfiguration.
Kamera 3 vises i appen som løbende opdaterede billeder uden lyd.
