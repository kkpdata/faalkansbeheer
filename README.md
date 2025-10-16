# faalkansbeheer
Het beheren en combineren van faalpaden en faalkansen voor waterkeringen.

## Doel en procesbeschrijving

Doel: het opzetten en koppelen van datastructuren zodanig dat:

* faalpaden gedefinieerd en beheerd kunnen worden;
* er vanuit een verzameling fragility-punten geassembleerd kan worden tot de faalkans van een dijktraject;
* het mogelijk is te rekenen met scenario's, ook voor indirecte mechanismen;
* tussenliggende stappen uitgevoerd kunnen worden;
* resultaten grafisch weergegeven kunnen worden.

Om de faalkans van een dijktraject te bepalen zijn de volgende stappen voorzien:

1. per doorsnede beschrijven van de mogelijke faalpaden
2. per (ondergrond)scenario voor elke knoop in het faalpad toekennen van fragility-punten
3. doorsnede koppelen aan vakken
4. combineren en integreren van fragilitycurves
5. combineren van scenariokansen
6. opschalen van doorsnedekansen naar vakkans
7. assembleren naar trajectkans


## Installatie

* maak een kopie van deze repository door `git clone `
* maak een virtuele Python (versie 3.12 of hoger) environment aan. 
* installeer alle afhankelijkheden door `pip install -r requirements.txt`

## Quickstart

Nog aanvullen

## Documentatie

De documentatie kan worden gegeneerd middels het commando `sphinx-build -M html docs\ docs\_build`. Of in de map docs door `make html`.  
Je vindt de documentatie daarna terug in de map `\docs\_build\html\index.html`. Dit bestand opent in de browser. Tip: voeg de documentatie toe aan je favorieten van de browser.
