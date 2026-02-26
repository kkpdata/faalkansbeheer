# Stappenplan combineren van FragilityCurves en Assembleren naar trajectresultaat

Deze toelichting is geschreven om het script te maken om FC te combineren, te integreren en te assembleren tot trajectkans.

Nodig:

- map met excels per vak
- map met Hydra-berekeningen (hfreq.txt)

Per vak is een excel opgezet met een vaste structuur. De naamgeving van deze excels staat vrij omdat de meta-info van het vak in de excel is opgenomen.

- De meta-informatie staat op regel 6 en geeft een koppeling met HR en scenariokans
- Per ondergrondscenario is er een tabblad (naamgeving staat vrij, zolang het maar niet "FP_overslag" en "FP_graverij" heet)

Opmerking: deze aanpak gaat er van uit dat indirecte mechanismen en overslag voor alle ondergrondscenario's gelijk zijn.

N.B. Dit is anders dan dat we besproken hebben, maar past beter in de huidige structuur.

Het script leest alle excels in een map in. Per excel:

1. Meta-info wordt weggeschreven naar een aparte tabel. Dit is de basistabel voor het resultaat
2. Combineren van FC's per faalpad in de excel
3. Wegschrijven figuren FC per knooppunt in een faalpad
4. Combineren van faalpaden per scenarioberekening
5. Wegschrijven van figuren FC per scenario en FC na combinatie van scenario's
6. Integreren van FC per scenario en FC na scenario's
7. Wegschrijven van resultaten (zie map example output voor de structuur)
