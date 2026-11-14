# HA Darts

Home Assistant OS-app, der finder den nye pil ud fra stillbilleder fra
to HTTP-kameraer før og efter kastet.

## Installation

Tilføj dette GitHub-repo under Indstillinger → Apps → App-butik → Repositories.
Installér og start HA Darts, og åbn appens webgrænseflade.

1. Åbn **Kameraer** i HA Darts, og indtast snapshotadressen for venstre
   og højre kamera, fx `http://kamera.local/picture`.
2. Tryk **Gem kameraer**. Appen tester begge adresser.
3. Vælg **Kalibrér 9 punkter** med tom skive efter skift af kameraadresse.

## Første tur

1. Tøm skiven, og vælg **Kalibrér 9 punkter**. Markér de fire kryds på dobbeltringens
   inderste wire, de samme fire sektorgrænser på tripelringens yderste wire
   og centrum af bull. Ret valgfri punkter i zoomvisningen før du gemmer.
2. Hold skiven tom, indtil appen viser **Klar. Kast en pil.**
3. Kast én pil, og vent på scoren, før du kaster igen. Appen analyserer
   automatisk og tæller op til tre pile, også når webvisningen er lukket.
4. Fjern alle pile. Appen starter næste tur, når den genkender den tomme
   skive i begge kameraer. Markeringer og point ryddes automatisk. Ved første
   start efter opdatering eller ny kalibrering skal skiven være tom, så appen
   kan gemme referencebilleder til automatisk start efter genstart.

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

## Kastdetektion

Venstre og højre kamera leverer stillbilleder til automatisk detektion og
scoring. Home Assistant håndterer forbindelserne til kameraerne. Kalibrering fryser billederne,
så krydsene kan markeres i samme billede.

## Skivekort

Kalibreringen og kastdetektionen bruger samme turneringslayout som Unicorn
Eclipse Pro 2. Før du gemmer, vises de 20 tal og wirerne på begge billeder.
Kontrollér især 20 øverst, 6 til højre, 3 nederst og 11 til venstre på den
fysiske skive.

## Automatisk finjustering

Tøm skiven og vælg **1-touch kalibrering**. En gemt manuel kalibrering
giver startposition og retning. Kontrollér forslaget i begge billeder,
og gem, hvis overlejringen passer. Kastdetektionen scorer den bedst
rangerede kandidat, når den bekræftes i to aflæsninger. MQTT-status viser
brokerens kvittering for pointværdien.

## Detaljeret kalibrering

Vælg **Detaljeret kalibrering**, og markér bullens centrum og de 20
sektorgrænser på hver af de seks ringkanter i hvert kamerabillede.
Ved bull forlænges sektorlinjen visuelt til ringkanten. Appen bruger
punkterne til lokale korrektioner. Kontrollér overlejringen og ret de
ønskede punkter, før du gemmer. Automatisk finjustering anvendes kun
på den ældre kalibrering med fire eller ni punkter.

## DeepDarts

Åbn **DeepDarts** i appen og upload en float32 TFLite-model med
800 × 800 RGB-input og to YOLO-udgange. Appen understøtter det
[TFLite-format, Dart-Vision beskriver](https://github.com/iambhabha/Dart-Vision#input--output-specification).
Modelfilen skal skaffes separat. Se [DeepDarts-artiklen](https://arxiv.org/abs/2105.09880)
og [forskernes kode og modelhenvisninger](https://github.com/wmcnally/deep-darts).

Appen bruger modellen til at finde spidser i begge kamerabilleder og
beregner scoren ud fra den gemte kamerakalibrering. Den almindelige
detektor tager over, når DeepDarts ikke finder et match mellem kameraerne.
