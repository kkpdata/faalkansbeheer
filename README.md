# faalkansbeheer
Het beheren en combineren van faalpaden en faalkansen voor waterkeringen.

Hierarchie is als volgt:

- Een `dijktraject` bestaat uit meerdere `dijkvakken`
- voor elk `dijkvak` zijn er één of meerdere `faalpaden` mogelijk
- Een `faalpad` bestaat uit meerdere `knopen`
- Voor elke `knoop` wordt per `scenario` een `faalkans` bepaald
- Een `faalkans` wordt gevormd door een `FragilityCurve`.
- Een `FragilityCurve` bestaat uit één of meerdere `FragilityPunten`.
- Een `FragilityPunt` kan bestaan uit een (conditionele) probabilistische berekening.
