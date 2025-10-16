Logisch model voor opdeling faalkans dijktraject
================================================

De faalkans van een `dijktraject` bestaat uit een seriesysteem van meerdere `componenten`. 
Omdat het een seriesysteem is, faalt het `dijktraject` als één van de `componenten` faalt.
`Componenten` zijn gekoppeld aan `doorsneden`. `Doorsneden` zijn gekoppeld aan `vakken`.

Een `vak` bestaat dus uit meerdere `componenten` die bijdragen aan de faalkans van het seriesysteem.

Elk `component` bestaat uit:

* een faalpad met één of meerdere `knopen`. Een `knoop` is ook te zien als een `component`. Een faalpad is een uitgeklede gebeurtenissenboom met alleen die gebeurtenissen die tot falen leiden. De kans op een gebeurtenis is conditioneel aan voorgaande gebeurtenissen;
* een foutenboom met één of meerdere `elementen` die door .

Elk `element` of `knoop` heeft als resultaat een faalkans. Deze faalkansen zijn verschillend in het domein waarvoor ze gelden.
Bij een `knoop` is deze faalkans conditioneel aan de voorgaande gebeurtenis.

Gekoppeld aan een `element` of `knoop` zit een `faalmechanismemodel`. 

Een `faalmechanismemodel` kan je zien als:

* een `faalkans`;
* een `fragilitycurve` bestaande uit één of meerdere `fragility-punten`. Een `fragility-punt` is een conditionele kans en kan bestaan uit een (conditionele) probabilistische berekening;
* een probabilistische berekening heeft een `grenstoestandfunctie`.


Hierarchie is dus als volgt:

- Een `dijktraject` bestaat uit meerdere `dijkvakken`
- voor elk `dijkvak` zijn er één of meerdere `faalpaden` mogelijk
- Een `faalpad` bestaat uit meerdere `knopen`
- Voor elke `knoop` wordt per `scenario` een `faalkans` bepaald
- Een `faalkans` wordt gevormd door een `FragilityCurve`.
- Een `FragilityCurve` bestaat uit één of meerdere `FragilityPunten`.
- Een `FragilityPunt` kan bestaan uit een (conditionele) probabilistische berekening.
