# MeltPoolDG

[![Doxygen](https://github.com/MeltPoolDG/MeltPoolDG/workflows/Doxygen/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions?query=workflow%3ADoxygen)
[![Indent](https://github.com/MeltPoolDG/MeltPoolDG/workflows/Indent/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions?query=workflow%3AIndent)
[![GitHub CI](https://github.com/MeltPoolDG/MeltPoolDG/workflows/GitHub%20CI/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions?query=workflow%3A%22GitHub+CI%22)
[![Nightly test](https://github.com/MeltPoolDG/MeltPoolDG/actions/workflows/nightly.yml/badge.svg)](https://github.com/MeltPoolDG/MeltPoolDG/actions/workflows/nightly.yml)

## (DG)-FEM-based multi-phase flow solvers for high-fidelity metal additive manufacturing process simulations

The aim of this project is to provide solvers for simulating the thermo-hydrodynamics in the vicinity of the melt pool during selective laser melting, including melt pool formation and the interaction of the multi-phase flow system (liquid metal/ambient gas/metal vapor). They are based on continuous and discontinuous finite element methods in an Eulerian setting. For modelling interfaces in the multi-phase flow problem including evaporation, level set methods and phase-field methods will be provided.

This project depends on the following third-party libraries:

- deal.II
- p4est
- Trilinos
- METIS (optional)
- adaflo (optional)

![alt text](doc/MeltPoolDG.png?raw=true)

### Documentation

The documentation can be found under https://meltpooldg.github.io/MeltPoolDG/.
