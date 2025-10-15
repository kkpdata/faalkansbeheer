.. faalkansbeheer documentation master file, created by
   sphinx-quickstart on Wed Oct 15 22:15:57 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Faalkansbeheer documentation
============================

 
Het beheren en combineren van faalpaden en faalkansen voor waterkeringen.

Hierarchie is als volgt:

- Een `dijktraject` bestaat uit meerdere `dijkvakken`
- voor elk `dijkvak` zijn er één of meerdere `faalpaden` mogelijk
- Een `faalpad` bestaat uit meerdere `knopen`
- Voor elke `knoop` wordt per `scenario` een `faalkans` bepaald
- Een `faalkans` wordt gevormd door een `FragilityCurve`.
- Een `FragilityCurve` bestaat uit één of meerdere `FragilityPunten`.
- Een `FragilityPunt` kan bestaan uit een (conditionele) probabilistische berekening.


.. toctree::
   :maxdepth: 2
   :caption: Inhoud:

   assemblage_protocol
   assemblage_functions

