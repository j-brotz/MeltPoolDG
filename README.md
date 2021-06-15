# MeltPoolDG

[![Doxygen](https://github.com/MeltPoolDG/MeltPoolDG/workflows/Doxygen/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions?query=workflow%3ADoxygen)
[![Indent](https://github.com/MeltPoolDG/MeltPoolDG/workflows/Indent/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions?query=workflow%3AIndent)
[![GitHub CI](https://github.com/MeltPoolDG/MeltPoolDG/workflows/GitHub%20CI/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions?query=workflow%3A%22GitHub+CI%22)
[![Nightly test](https://github.com/MeltPoolDG/MeltPoolDG/actions/workflows/nightly.yml/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions/workflows/nightly.yml)

## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of `MeltPoolDG` is to provide application-oriented, research-driven solvers for modeling the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are and will be based on continuous and discontinuous finite element methods in an Eulerian setting. For interface modeling of the multi-phase flow problem including phase-transition (evporation/melting), diffuse interface capturing schemes such as level set methods and phase-field methods are used. It strongly builds upon the general purpose finite element library [deal.II](https://github.com/dealii/dealii) and the highly efficient, matrix-free two-phase flow solver [adaflo](https://github.com/kronbichler/adaflo). Furthermore, via `deal.II` we access also other important third-party libraries such as `p4est`, `Trilinos` and `Metis`.

`MeltPoolDG`is freely available under the LGPLV2 license. Please find the details in the [LICENSE](https://github.com/MeltPoolDG/MeltPoolDG/blob/master/LICENSE) file.

![alt text](doc/MeltPoolDG.png?raw=true)

## Authors

The principal developers are (in chronological order of entering the project):
* [Magdalena Schreter](https://www.uibk.ac.at/bft/mitarbeiter/schreter.html) [@mschreter](https://github.com/mschreter), University of Innsbruck (AT)/Technical University of Munich (DE)
* [Peter Munch](https://www.mw.tum.de/lnm/staff/peter-munch) [@peterrum](https://github.com/peterrum), Technical University of Munich/Helmholtz Zentrum hereon (DE)
* [Nils Much](https://www.mw.tum.de/lnm/staff/nils-much/) [@nmuch](https://github.com/nmuch), Technical University of Munich (DE)

## Documentation

The documentation can be found under https://meltpooldg.github.io/MeltPoolDG/.
