<img align="left" width="324" src="doc/logo/logo_text_text.png">


<br clear="left"/>
<br clear="left"/>

[![Indent](https://github.com/MeltPoolDG/MeltPoolDG-dev/workflows/Indent/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG-dev/actions/workflows/indent.yml)
[![GitHub CI](https://github.com/MeltPoolDG/MeltPoolDG-dev/actions/workflows/nightly_tests.yml/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG-dev/actions/workflows/nightly_tests.yml)
[![Nightly Install](https://github.com/MeltPoolDG/MeltPoolDG-dev/actions/workflows/nightly_install.yml/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG-dev/actions/workflows/nightly_install.yml)
[![Documentation](https://github.com/MeltPoolDG/MeltPoolDG-dev/actions/workflows/deploy.yml/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG-dev/actions/workflows/deploy.yml)

## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of `MeltPoolDG` is to provide application-oriented, research-driven solvers for modeling the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are and will be based on continuous and discontinuous finite element methods in an Eulerian setting. For interface modeling of the multi-phase flow problem including phase-transition (evporation/melting), diffuse interface capturing schemes such as level set methods and phase-field methods are used. It strongly builds upon the general purpose finite element library [deal.II](https://github.com/dealii/dealii) and the highly efficient, matrix-free two-phase flow solver [adaflo](https://github.com/kronbichler/adaflo). Furthermore, via `deal.II` we access also other important third-party libraries such as `p4est`, `Trilinos` and `Metis`.

`MeltPoolDG` is freely available under the LGPLV2 license. Please find the details in the [LICENSE](https://github.com/MeltPoolDG/MeltPoolDG/blob/master/LICENSE) file.

![alt text](doc/MeltPoolDG.png?raw=true)

## Authors

The principal developers are (in chronological order of entering the project):
* [Magdalena Schreter-Fleischhacker](https://www.uibk.ac.at/bft/mitarbeiter/schreter.html) [@mschreter](https://github.com/mschreter), University of Innsbruck (AT)/Technical University of Munich (DE)
* [Peter Munch](https://peterrum.github.io/) [@peterrum](https://github.com/peterrum), Technical University of Berlin (DE)
* [Nils Much](https://www.mw.tum.de/lnm/staff/nils-much/) [@nmuch](https://github.com/nmuch), Technical University of Munich (DE)

We gratefully acknowledge the contributions and discussions with Christoph Meier (Technical University of Munich) and Martin Kronbichler (Ruhr University Bochum).

## Installation

A description on the installation procedure including the coniguration scripts can be found in [scripts/config](https://github.com/MeltPoolDG/MeltPoolDG-dev/tree/master/scripts/config) directory.

## Documentation

see https://meltpooldg.github.io/MeltPoolDG/ and https://meltpooldg.github.io/MeltPoolDG/doxygen
