# A thermal-based convection scheme

This repository documents a simple parcel-based investigation of
atmospheric convection, which is modelled as a series of discrete
entraining thermals. Although time constraints prevented sufficient
development of the model, it is possible that, with further
improvements, it could be used in a convection parametrisation in
a climate model.

## Methods

The model is an implementation of the heterogeneous parcel model
described by Sherwood et al. (2013): the entraining thermal is
considered as a collection of non-entraining, non-mixing components
which are progressively added during ascent/descent. For the purposes
of determining buoyancy and velocity, the properties of the components
are averaged, weighted by their respective mass fractions.
[`thermal_generator.py`](https://github.com/tschanzer/convection-scheme/blob/main/convection_scheme/thermal_generator.py)
defines the `ThermalGenerator` class, which is instantiated with
the environmental sounding data and supplies `updraft` and `downdraft`
methods to simulate ascending and descending thermals.

One of the main focuses of the project was the coupling of downdrafts
to updrafts via the precipitation generation mechanism. Under the
(admittedly somewhat unrealistic) assumption that downdrafts are
initiated by the evaporation of all precipitation that falls out of a
preceding updraft within a layer of fixed thickness above the downdraft
initiation level, we explored the dependence of downdraft strength
on updraft and downdraft entrainment rates and downdraft initiation
level.
[`coupler.py`](https://github.com/tschanzer/convection-scheme/blob/main/convection_scheme/coupler.py)
provides utilities for the relevant calculations, including the
simulation of ensembles of updrafts and downdrafts with spectrums of
entrainment rates.

As part of the investigation of updraft/downdraft ensembles, we
implemented the stochastic parcel model described by Romps and Kuang
(2010). This model treats entrainment as a discrete Poisson process,
where the thermal propagates without entrainment, except at a finite 
set of random entrainment events. Both the distance between events
and the mass entrained at each event are exponentially distributed.
[`stochastic_generator.py`](https://github.com/tschanzer/convection-scheme/blob/main/convection_scheme/stochastic_generator.py)
and [`stochastic_coupler`](https://github.com/tschanzer/convection-scheme/blob/main/convection_scheme/stochastic_coupler.py)
serve purposes analagous to their deterministic counterparts.

In all cases, the class and method docstrings should provide sufficient
information for new users.

## Results
Here, we omit discussion of updraft/downdraft ensembles and the
stochastic parcel model; it is difficult to draw concrete conclusions
about them in this extremely simple and idealised setting.

Although the detrimental effect of entrainment is rather
obvious, the dependence of precipitation-driven downdraft strength on
initiation level
is more difficult to explain and varies from sounding to sounding.
Notably, in some soundings and for higher updraft entrainment rates
(i.e., less available precipitation), there are two initiation levels
in the lowest 4 km that result in maximal downdraft strength.
The location of a low-level maximum is usually present and well
approximated by the level at which maximum precipitation is available
from the updraft (close to the LCL), but the location of a maximum
higher in the atmosphere, which is only present in some soundings, is
more difficult to predict. Analysis of the environmental equivalent
potential temperature profile has been somewhat successful; maxima
in downdraft strength may coincide with local minima in $\theta_e$.
Further work is needed to determine the cause of this two-maximum
structure.

Brief consideration was given to the behaviour of downdrafts in
soundings that were artificially made warmer while preserving
relative humidity and CAPE, to simulate a warmer climate. It was found
that, since the same relative humidity at a higher temperature implies
a higher specific humidity, updrafts in the warmer soundings produce
more precipitation and therefore have the potential to trigger stronger
downdrafts. Warming the sounding in this way seems to have no effect
on the existing relationship between downdraft strength and initiation
level.

## References
Romps, D. M., & Kuang, Z. (2010). Nature versus nurture in shallow
convection. *Journal of the Atmospheric Sciences*, **67**(5), 1655–1666.
https://doi.org/10.1175/2009JAS3307.1


Sherwood, S. C., Hernández-Deckers, D., Colin, M., & Robinson, F. (2013).
Slippery thermals and the cumulus entrainment paradox. *Journal of the
Atmospheric Sciences*, **70**(8), 2426–2442.
https://doi.org/10.1175/JAS-D-12-0220.1

