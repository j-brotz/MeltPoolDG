# Parameters changelog
All notable changes of the input parameters will be documented in this file.


## 2022-11-02
- Add specification for FE interpolation of heat equation in melt pool problem
```diff
{
  "problem type": "melt_pool",
  "heat": {
+   "degree": "",
+   "n q points 1d": "",
+   "n subdivisions": ""
  }
}
```

## 2022-11-29
- Add parameter for iteration of nonlinear evaporation-induced coupling term
in the heat equation
```diff
{
  "problem type": "melt_pool",
  "problem specific": {
+    "coupling temp evapor": 
+    {
+      "n max iter": "",
+      "tol": ""
+    }
  }
}
```

## 2022-11-28
- Allow larger time step size during heat up phase in the melt pool problem
```diff
{
  "problem type": "melt_pool",
  "problem specific": {
+    "mp heat up": 
+    {
+      "max temperature": "",
+      "time step size": "",
+      "max change factor time step size": ""
+    }
  }
}
```

## 2022-11-09
- Add restart parameters
```diff
{
+  "restart":
+  {
+    "load": "",
+    "save": "",
+    "prefix": "",
+    "write frequency": "",
+    "write time step size": "",
+  }
}
```
## 2022-10-28
- Add parameter to enable curvature computation
```json
{
  "curvature":
  {
    "enable": "true|false"
  }
}
```

## 2022-10-24
- Add parameter for iteration of nonlinear evaporation-induced coupling term
in the level set equation
```json
{
  "problem type": "melt_pool",
  "problem specific": {
    "coupling ls evapor": 
    {
      "n max iter": "",
      "tol"
    }
  }
}
```

## 2022-10-10
- Add a parameter to enable user-defined postprocessing
```json
{
  "paraview": {
    "do user defined postprocessing": "true"
  },
```
- Add a parameter to enable profiling
```json
{
  "profiling": {
    "enable": "true"
  },
```

## 2022-10-04
- Add a parameter to enable extrapolation of solution vectors for coupling terms
```json
{
  "problem type": "melt_pool",
  "problem specific": {
    "do extrapolate coupling terms": "true|false"
  },
}
```
- Add parameter for monitoring
```json
{
  "linear solver": {
    "monitor type": "none|reduced|all"
  }
}
```

- Add parameter to use slip boundaries instead of no-slip boundaries in the recoil pressure simulation
```diff
{
  "base": {
    "application name": "recoil_pressure"
  },
  "simulation specific domain": {
+   "slip boundary": "false|true"
  }
}
```

- Add uniform laser heat source
```json
{
  "laser": {
    "laser heat source model": "uniform"
  }
}
```

## 2022-10-03
- Add a parameter for predictor type of linear solver
```json
{
    "linear solver": {
      "predictor": "none|linear_extrapolation"
    }
}
```

## 2022-10-03
- Add a parameter for prescribing an analytical function for the time step size
```json
{
  "time stepping": {
    "time step size function": "0.01*t"
  }
}
```

## 2022-09-14
- Add parameter to distinguish between sharp (surface integral) and diffuse
  (volume integral) of the evaporative heat loss term at the vapor surface
```json
{
  "evaporation": {
    "evapor formulation source term heat": "sharp|diffuse"
  },
}
```

## 2022-09-14
- Add parameter to distinguish between sharp (surface integral) and diffuse
  (volume integral) of the evaporative heat loss term at the vapor surface
```json
{
  "evaporation": {
    "evapor formulation source term heat": "sharp|diffuse"
  },
}
```

## 2022-09-12
- Add parameters to control write frequency of profiling output 
```json
{
  "profiling": {
    "write frequency": "",
    "write time step size": "",
}
```
- Add new option `interface_sharp` for Gaussian laser heat source
```json
{
  "laser": {
    "laser impact type": "interface_sharp"
  }
}
```

## 2022-09-07
- Add parameter to specify request variables for paraview output
```json
{
  "paraview": {
    "output variables": "var1,var2"
  }
}
```

## 2022-07-08
- Add parameter to make the heat operator interpolate the product of density and capacity instead of both separately.
```json
{
  "heat": {
    "interpolate rho times cp": "true|false"
  }
}
```

## 2022-06-30
- Add tolerance for activating reinitialization 
```json
{
  "levelset": {
    "tol reinit": ""
  }
}
```

## 2022-06-26
- Add option to determine weights for phase weighted Dirac delta function 
  automatically
```json
{
  "dirac delta function approximation": {
    "auto weights": "true|false"
  }
}
```
- Introduce new parameter to enable time-dependent boundary conditions in the
  HeatOperation.
```diff
{
  "heat": {
+   "enable time dependent bc": "true|false"
  }
}
```
- Add new parameter
```diff
{
 "recoil pressure": {
+  "model type": "phenomenological|hybrid",
  }
}
```

## 2022-06-01
- Remove parameter of MeltPoolProblem and rename parameter "do evaporative mass flux"
  to "do evaporative velocity jump"

```diff
{
  "base": {
    "problem name": "melt_pool"
  },
  "problem specific": {
-   "do evaporation": "",
-   "do evaporative mass flux": "",
+   "do evaporative velocity jump": ""
  }
}
```

## 2022-06-01
- Extend parameter and rename parameter values

```diff
{
  "evaporation": {
~   "evapor evaporative mass flux": "any time-dependent function e.g. 1.*t",
~   "evapor evaporation model": "constant|hardt_wondra|recoil_pressure"
  }
```

## 2022-06-07
- Add parameter to recoil pressure simulation for cell repetitions per dimension applied before global refinement.

```json
{
  "application name": "recoil_pressure",
  "simulation specific domain": {
    "cell repetitions": "x,y,z"
  }
}
```

## 2022-05-26
- Move parameter 

```diff
{
  "reinitialization": {
-   "reinit dtau": ""
  },
  "levelset": {
+   "ls reinit time step size": ""
  }
}
```

- Add parameter enable/disable the time step limit due to explicit treatment
of surface tension in the melt pool problem

```json
{
  "base": {
    "problem name": "melt_pool"
  },
  "surface tension": {
    "time step limit": {
      "enable": "true|false",
      "scale factor": "",
    }
  }
}
```

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
```diff
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
