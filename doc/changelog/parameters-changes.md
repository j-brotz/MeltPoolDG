# Parameters changelog
All notable changes of the input parameters will be documented in this file.

## 2022-05-20
- Add parameter enable/disable the advection of the level set in
the melt pool problem

```json
{
  "base": {
    "problem name": "melt_pool"
  },
  "problem specific": {
    "do advect level set": "true|false"
  }
}
```

## 2022-05-19
- Add parameter for determining if the heaviside representation of the level 
set function should be calculated localized or not

```json
{
  "levelset": {
    "ls do localized heaviside": "true|false"
  }
}
```

## 2022-05-13
- Add parameter for activation temperature of the recoil pressure

```json
{
  "recoil pressure": {
    "activation temperature": ""
  }
}
```

## 2022-05-13
- Add parameter for a user-defined material model in the Navier-Stokes solver

```json
{
  "Navier-Stokes": {
    "adaflo": {
      "Navier-Stokes": {
        "constitutive type": "user defined"
      }
    }
  }
}
```

## 2022-03-16
- Add parameter for absolute tolerance of linear solvers.

```json
{
  "heat|reinitialization|curvature|normal vector|advection diffusion": {
    "linear solver": {
      "abs tolerance": ""
    }
  }
}
```

## 2022-03-11
- Add parameter to enable interpolation of the level set field to the pressure 
  space for computing the level set gradient in evaporative mass flux source terms.
```json
{
  "evaporation": {
    "evapor do level set pressure gradient interpolation": "true|false"
  }
}
```

## 2022-03-10
- Add a parameter that controls whether higher order cells should be written to the vtu output.
```json
{
  "paraview": {
    "write higher order cells": "true|false"
  }
}
```

## 2022-03-09
- Add parameter to distinguish how to consider evaporative mass flux in the 
  level set equation
```json
{
  "evaporation": {
+   "evapor level set source term type": "interface_velocity|rhs" 
  }
}

```

## 2022-02-17
- Delete ambiguous parameter
```diff
{
  "melt pool": {
-   "mp liquid melting point": ""
  }
}

```

## 2022-02-17
- Add problem-specific parameter for AMR in the melt pool problem
```json
{
  "base": {
    "problem name": "melt_pool"
  },
  "problem specific": {
    "amr" : {
      "fraction of melting point refined in solid": "0.0<x<1.0"
    }
  }
}
```

## 2022-02-09
- Add problem-specific parameter for AMR in the melt pool problem
```json
{
  "base": {
    "problem name": "melt_pool"
  },
  "problem specific": {
    "amr" : {
      "strategy": "generic|adaflo|KellyErrorEstimator",
      "do auto detect frequency": "true|false"
      "do refine all interface cells": "true|false"
      "automatic grid refinement type": "fixed_number|fixed_fraction"
    }
  }
}

```

## 2022-01-07
- Add parameter for narrow band threshold for normal vector and curvature
```json
{
  "normal vector": {
    "narrow band threshold": ""
  },
  "curvature": {
    "narrow band threshold": ""
  }
}
```

## 2021-12-21
- Add new option for Dirac delta approximation type
```json
{
  "dirac delta function approximation": {
    "type": "reciprocal_times_heaviside_phase_weighted"
  }
}
```

## 2021-12-20
- Delete ambiguous parameter 
```json
{
  "levelset" : {
     "ls do matrix free": ""
  }
}
```
- Change names for the linear solver data from
```json
{
  "heat": {
    "heat do matrix free": ""
    "heat solver rel tolerance": "",
    "heat solver type": "",
    "heat solver preconditioner type": "",
    "heat solver max iterations": "",
  },
  "reinitialization": {
    "reinit do matrix free": ""
    "reinit solver rel tolerance": "",
    "reinit solver type": "",
    "reinit solver preconditioner type": "",
    "reinit solver max iterations": "",
  },
  "curvature": {
    "curv do matrix free": ""
  },
  "normal vector": {
    "normal vec do matrix free": ""
  },
  "advection diffusion": {
    "advec diff do matrix free": ""
  }
}
```
to
```json
{
  "heat": {
    "linear solver": {
      "do matrix free": "",
      "rel tolerance": "",
      "solver type": "",
      "preconditioner type": "",
      "max iterations": "",
    }
  },
  "reinitialization": {
    "linear solver": {
      "do matrix free": "",
      "rel tolerance": "",
      "solver type": "",
      "preconditioner type": "",
      "max iterations": "",
    }
  },
  "curvature": {
    "linear solver": {
      "do matrix free": "",
    }
  },
  "normal vector": {
    "linear solver": {
      "do matrix free": "",
    }
  },
  "advection diffusion": {
    "linear solver": {
      "do matrix free": "",
    }
  }
}
```

## 2021-12-16
- Change names for the Dirac delta approximation types and add a new option "reciprocal_phase_weighted"
```json
{
  "dirac delta function approximation": {
    "type": "heaviside_phase_weighted|quad_heaviside_phase_weighted|heaviside_times_heaviside_phase_weighted|reciprocal_phase_weighted",
  }
}
```

## 2021-12-10
- Add new parameter to distinguish how interfacial forces are calculated: 
```json
{
  "recoil pressure" : {
     "interface distributed flux type": "continuous|interface value"
  }
}
```
