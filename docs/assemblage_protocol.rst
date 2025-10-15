Assemblageprotocol voor waterkeringen documentation
===================================================

Bronnen
-------

De implementatie van het assemblageprotocol is gebaseerd op de volgende bronnen:

- Rode draad #10 - Assembleren van Adviesteamd Dijkontwerp https://adviesteamdijkontwerp.nl/rode-draden/rode-draad-nr-10-assembleren/


Globale Werkwijze bepalen faalkans dijktraject
----------------------------------------------

De faalkans van een dijktraject staat gelijk aan de faalkans van een seriesysteem: als één component faalt, dan faalt het systeem. De assemblage maakt gebruik van de elementaire boven- en ondergrens van een seriesysteem. 

Bij het bepalen van de faalkans van een dijktraject worden de volgende stappen doorlopen:

1. bepalen van de kans per doorsnede :math:`P_{f,dsn}`. Dit kan de uitkomst zijn van een faalpadanalyse of een foutenboomanalyse. Het effect van indirecte mechanismen zit in de doorsnedekans verwerkt. De eenheid van de doorsnedekans is kans per jaar.
2. verschalen van de doorsnedekans naar kans per vak via :math:`N_{vak}`. 
3. combineren van de vakkansen tot de kans dat een specifiek faalmechanisme ergens in het traject optreedt en tot falen leidt. Voor mechanismen die sterk afhankelijk zijn van de belasting, wordt de elementaire bovengrens gebruikt (MAX). Voor de overige mechanismen die niet onafhankelijk van elkaar optreden wordt gebruik gemaakt van de elementaire ondergrens (SOM).
4. combineren van de kansen van de verschillende faalmechanismen tot de kans dat het traject faalt. Afhankelijk van de mate van afhankelijkheid tussen de faalmechanismen wordt gebruik gemaakt van de elementaire boven- of ondergrens.

Werkwijze bepalen faalkans per dijkvak
--------------------------------------

Bij het bepalen van de faalkans van een vak in een dijktraject worden de volgende stappen doorlopen:

1. bepalen van de kans per vak door het combineren van de vakkansen van de verschillende faalmechanismen tot de kans dat het vak faalt. Hierbij wordt wederom gebruik gemaakt van de eigenschappen van het faalmechanisme. Voor faalmechanismen die sterk afhankelijk zijn van de belasting, wordt de elementaire bovengrens gebruikt (MAX). Voor de overige mechanismen de (SOM).
2. Bepalen van de boven- en ondergrenzen door het combineren van de vakkansen.