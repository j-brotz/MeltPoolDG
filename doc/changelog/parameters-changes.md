# Parameters changelog
All notable changes of the input parameters will be documented in this file.

## 2024-03-20
- move sticking constant from material
```diff
{
  "material":
  {
-    "sticking constant": "1"
  }
  "evaporation":
  {
      "recoil pressure":
      {
+    "sticking constant": "1"
      }
  }
}
```

## 2024-03-08
-  Introduce fe data
```diff
{
  "base": {
-    "degree": "",
-    "n q points 1d": "",
-    "do simplex"; "",
+    "fe": {
+      "type": "FE_Q|FE_SimplexP|FE_Q_iso_Q1|FE_DGQ",
+      "degree": ""
+    }
  }
  "heat": {
-    "degree": "",
-    "n q points 1d": "",
-    "n subdivisions": "",
+    "fe": {
+      "type": "FE_Q|FE_SimplexP|FE_Q_iso_Q1|FE_DGQ",
+      "degree": ""
+    }
  }
  "level set": {
-    "n subdivisions": "",
+    "fe": {
+      "type": "FE_Q|FE_SimplexP|FE_Q_iso_Q1|FE_DGQ",
+      "degree": ""
+    }
  }
}
```

## 2024-03-06
- Rename flow section
```diff
{
-  "Navier-Stokes": {
-  }
  "base": {
-    "gravity": ""
  }
-  "surface tension": {
-  }
-  "darcy damping": {
-  }
+  "flow": {
+    "gravity": "",
+    "surface tension": {
+    },
+    "darcy damping": {
+    }
+  }
}
```

## 2024-03-07
- Refactor level set data; moved curvature, normal vector, reinitialization,
    advection diffusion inside level set struct
```diff
-    "advection diffusion":
-    {
-        "advec diff diffusivity": "0",
-        "advec diff implementation": "meltpooldg",
-        "advec diff time integration scheme": "crank_nicolson",
-    },
-    "curvature":
-    {
-        "curv damping scale factor": "0",
-        "curv do narrow band": "false",
-        "curv implementation": "meltpooldg",
-        "curv verbosity level": "0",
-        "enable": "true",
-        "narrow band threshold": "1",
-    },
-    "normal vector":
-    {
-        "narrow band threshold": "1",
-        "normal vec damping scale factor": "0.5",
-        "normal vec do narrow band": "false",
-        "normal vec implementation": "meltpooldg",
-        "normal vec verbosity level": "0",
-    },
-    "reinitialization":
-    {
-        "reinit constant epsilon": "-1",
-        "reinit implementation": "meltpooldg",
-        "reinit max n steps": "5",
-        "reinit modeltype": "olsson2007",
-        "reinit scale factor epsilon": "0.5",
-    }
-    "levelset":
+    "level set":
     {
-        "ls do curvature correction": "false",
-        "ls do localized heaviside": "true",
-        "ls do reinitialization": "true",
-        "ls implementation": "meltpooldg",
-        "ls n initial reinit steps": "-1",
-        "ls n subdivisions": "1",
-        "ls reinit time step size": "-1",
-        "ls time integration scheme": "crank_nicolson",
-        "tol reinit": "2.22507e-308",
+        "do localized heaviside": "true",
+        "n subdivisions": "1",
+        "advection diffusion":
+        {
+            "diffusivity": "0",
+            "implementation": "meltpooldg",
+            "time integration scheme": "crank_nicolson",
+        },
+        "curvature":
+        {
+            "do curvature correction": "false",
+            "enable": "true",
+            "filter parameter": "0",
+            "implementation": "meltpooldg",
+            "verbosity level": "0",
+            "narrow band":
+            {
+                "enable": "false",
+                "level set threshold": "1"
+            },
+        },
+        "normal vector":
+        {
+            "filter parameter": "0.5",
+            "implementation": "meltpooldg",
+            "verbosity level": "0",
+            "narrow band":
+            {
+                "enable": "false",
+                "level set threshold": "1"
+            },
+        },
+        "reinitialization":
+        {
+            "enable": "true",
+            "implementation": "meltpooldg",
+            "max n steps": "5",
+            "n initial steps": "-1",
+            "tolerance": "2.22507e-308",
+            "type": "olsson2007",
+            "interface thickness parameter":
+            {
+                "type": "proportional_to_cell_size",
+                "val": "0.5"
+            },
     },
```

## 2024-03-01
- Refactor evaporation data; moved recoil pressure to evaporation struct
```diff
    "evaporation": {
-        "evapor do level set pressure gradient interpolation": "false",
+        "do level set pressure gradient interpolation": "false",
-        "evapor level set source term type": "interface_velocity",
+        "formulation source term level set": "interface_velocity_local",
-        "evapor formulation evaporative mass flux over interface": "continuous",
+        "interface temperature evaluation type": "local_value",
-        "evapor evaporation model": "constant",
+        "evaporative mass flux model": "analytical",
-        "evapor evaporative mass flux": "0.0",
+        "analytical": {
+            "function": "0.0"
+        },
-        "evapor formulation source term heat": "diffuse",
+        "evaporative cooling": {
+            "consider enthalpy transport vapor mass flux": "false",
+            "enable": "false",
+            "model": "regularized",
+            "dirac delta function approximation": {
+                "auto weights": "false",
+                "gas phase weight": "1",
+                "gas phase weight 2": "1",
+                "heavy phase weight": "1",
+                "heavy phase weight 2": "1",
+                "type": "norm_of_indicator_gradient"
+            }
+        },
-        "evapor formulation source term continuity": "diffuse",
+        "evaporative dilation rate": {
+            "enable": "false",
+            "model": "regularized"
+        },
-        "evapor coefficient": "0",
+        "hardt wondra": {
+            "coefficient": "0"
+        },
~        "recoil pressure": {
~            "activation temperature": "0",
~            "ambient gas pressure": "101300",
+            "enable": "false",
~            "interface distributed flux type": "local_value",
~            "pressure coefficient": "0.55",
~            "temperature constant": "0",
~            "type": "phenomenological",
~            "dirac delta function approximation": {
~                "auto weights": "false",
~                "gas phase weight": "1",
~                "gas phase weight 2": "1",
~                "heavy phase weight": "1",
~                "heavy phase weight 2": "1",
~                "type": "norm_of_indicator_gradient"
~            }
~        },
+        "thickness integral": {
+            "subdivisions MCA": "1",
+            "subdivisions per side": "10"
+        }
-        "evapor line integral n subdivisions MCA": "1",
-        "evapor line integral n subdivisions per side": "10",
    },
    "heat": {
-        "dirac delta function approximation": {
-            "auto weights": "false",
-            "gas phase weight": "1",
-            "gas phase weight 2": "1",
-            "heavy phase weight": "1",
-            "heavy phase weight 2": "1",
-            "type": "norm_of_indicator_gradient"
-        },
    },
  "base": {
    "problem name": "melt_pool"
  },
  "problem specific": {
-   "do evaporative velocity jump": ""
-   "do evaporative heat flux": ""
-   "do recoil pressure": ""
  }
}
```
- new parameters (their action has not existed before)
```diff
{
    "evaporation": {
+        "evaporative cooling": {
+            "consider enthalpy transport vapor mass flux": "false",
+        },
    },
}
```
- deleted parameters
```diff
{
    "evaporation": {
-        "evapor evaporative mass flux scale factor": "1",
-        "evapor ls value gas": "-1",
-        "evapor ls value liquid": "1"
    },
}
```

## 2024-02-29
- Introduce substructures for heat radiation and convection BC
```diff
{
  "heat": {
-    "emissivity": "",
-    "convection coefficient": "",
-    "temperature infinity": "",
+    "radiative boundary condition": {
+      "emissivity": "",
+      "temperature infinity": "",
+    },
+    "convective boundary condition": {
+      "convection coefficient": "",
+      "temperature infinity": "",
+    }
  }
}
```

## 2024-02-28
- Refactor material data
```diff
{
  "material": {
-    "material first ": "conductivity",
-    "material first ": "capacity",
-    "material first ": "density",
-    "material first ": "viscosity",
+    "gas": {
+      "thermal conductivity": "",
+      "specific heat capacity": "",
+      "density": "",
+      "dynamic viscosity": "",
+    }
-    "material second ": "conductivity",
-    "material second ": "capacity",
-    "material second ": "density",
-    "material second ": "viscosity",
+    "liquid": {
+      "thermal conductivity": "",
+      "specific heat capacity": "",
+      "density": "",
+      "dynamic viscosity": "",
+    }
-    "material solid ": "conductivity",
-    "material solid ": "capacity",
-    "material solid ": "density",
-    "material solid ": "viscosity",
+    "solid": {
+      "thermal conductivity": "",
+      "specific heat capacity": "",
+      "density": "",
+      "dynamic viscosity": "",
+    }
-    "material solidus temperature": "",
-    "material melting point": "",
+    "solidus temperature": "",
-    "material liquidus temperature": "",
+    "liquidus temperature": "",
-    "material boiling temperature": "",
+    "boiling temperature": "",
-    "material latent heat of evaporation": "",
+    "latent heat of evaporation": "",
-    "material molar mass": "",
+    "molar mass": "",
-    "material sticking constant": "",
+    "sticking constant": "",
-    "material specific enthalpy reference temperature": "",
+    "specific enthalpy reference temperature": "",
-    "material two phase properties transition type": "",
+    "two phase fluid properties transition type": "",
-    "material solidification type": "",
+    "solid liquid properties transition type": "",
  }
}
```

## 2024-02-27
- Refactor recoil pressure data
```diff
{
    "recoil pressure":
    {
+       "ambient gas pressure": "",
-       "recoil pressure constant": "",
+       "pressure coefficient": "0...1",
-       "recoil temperature constant": "",
+       "temperature constant": "",
    }
}
```
- Add new evaporation model type
```diff
{
    "evaporation":
    {
~       "evapor evaporation model type": "saturated_vapor_pressure",
    }
}
```



## 2024-02-26
- Refactor output data

```diff
{
-  "paraview": {
+  "output": {
-    "paraview do output": "",
-    "paraview directory": "",
+    "directory": "",
-    "paraview write frequency": "",
+    "write frequency": "",
-    "paraview write time step size": "",
+    "write time step size": "",
-    "paraview filename": "",
-    "paraview n digits timestep": "",
-    "paraview print boundary id": "",
-    "paraview output subdomains": "",
-    "output material id": "",
-    "write higher order cells": "",
-    "paraview n groups": "",
-    "paraview n patches": "",
+    "paraview": {
+      "enable": "true|false",
+      "filename": "",
+      "n digits timestep": "",
+      "print boundary id": "",
+      "output subdomains": "",
+      "output material id": "",
+      "write higher order cells": "",
+      "n groups": "",
+      "n patches": ""
+    }
  }
}
```

## 2024-02-26
- Rafactor heat data
```diff
{
  "heat": {
-    "heat convection coefficient": "",  
+    "convection coefficient": "",
-    "heat emissivity": "",  
+    "emissivity": "",
-    "heat temperature infinity": "",  
+    "temperature infinity": "",
-    "heat nlsolve max nonlinear iterations": "",
-    "heat nlsolve field correction tolerance": "",
-    "heat nlsolve residual tolerance": "",
-    "heat nlsolve max nonlinear iterations alt": "",
-    "heat nlsolve field correction tolerance alt": "",
-    "heat nlsolve residual tolerance alt": ""
    "nlsolve":  {
+      "max nonlinear iterations": "",
+      "field correction tolerance": "",
+      "residual tolerance": "",
+      "max nonlinear iterations alt": "",
+      "field correction tolerance alt": "",
+      "residual tolerance alt": ""
    }
  }
} 
```

## 2024-02-22
- Refactor laser data
```diff
{
  "laser": {
-    "laser heat source model": "",
+    "model": "analytical_temperature|volumetric|interface_projection_regularized|interface_projection_sharp|interface_projection_sharp_conforming|RTE",
-    "laser impact type": "",
+    "intensity profile": "uniform|Gauss|Gusarov",
-    "laser power": "",
+    "power": "",
-    "laser power over time": "",
+    "power over time": "",
-    "laser power start time": "",
+    "power start time": "",
-    "laser power end time": "",
+    "power end time": "",
-    "laser gauss absorptivity gas": "",
+    "absorptivity gas": "",
-    "laser gauss absorptivity liquid": "",
+    "absorptivity liquid": "",
-    "laser do move": "",
+    "do move": "",
-    "laser scan speed": "",
+    "scan speed": "",
-    "laser gauss laser beam radius": "",
-    "laser gusarov laser beam radius": "",
+    "radius": "",
-    "laser gusarov reflectivity": "",
-    "laser gusarov extinction coefficient": "",
-    "laser gusarov layer thickness": "",
+    "gusarov": {
+      "reflectivity": "",
+      "extinction coefficient": "",
+      "layer thickness": "",
+    },
    "analytical": {
-      "absorptivity liquid": "",
-      "absorptivity gas": "",
    }
  }
}
```

## 2024-02-20
- Refactor laser data
```diff
{
  "laser": {
-    "laser center": ""
+    "starting position": ""
+    "direction": "0,0,-1"
  }
}
```
- Make RTE direction a problem specific parameter
```diff
{
  "rte": {
-    "laser direction": ""
  },
  "base": {
    "problem name": "radiative_transport"
  },
  "problem specific": {
+    "direction": "0,0,-1"
  }
}
```

## 2024-02-14
- Change parameter back to bool to choose to interpolate the volumetric thermal
capacity and remove specific interpolation for thermal conductivity
```diff
{
  "heat": {
-    "interpolate rho times cp": "none|sharp|smooth|reciprocal",
+    "use volume-specific thermal capacity for phase interpolation": "false|true",
-    "interpolate k": ""
   }
}
```
- Adjust name of temperature evaluation method
```json
{
  "recoil pressure" : {
     "interface distributed flux type": "local_value|interface_value"
  }
}
```

## 2024-02-13
- RTE: make pseudo time stepping a predictor and remove time_dependent_problem
```diff
{
  "rte": {
-    "problem type": "",
+    "predictor type": "none|pseudo_time_stepping"
  }
}
```
- RTE pseudo time stepping: use time_stepping_data
```diff
{
  "rte": {
    "predictor type": "pseudo_time_stepping",
    "pseudo time stepping": {
-      "max n steps": "",
-      "time step size": "",
+      "time stepping": {
+        "max n steps": "",
+        "time step size": "",
+      }
    }
  }
}
```
- rename RTE verbosity level
```diff
{
  "rte": {
-    "verbosity level": "",
+    "rte verbosity level": "",
  }
}
```

## 2023-11-20
- Revise parameters of profiling output
```diff
{
  "profiling": {
-   "write frequency": "",
    "write time step size": "",
+   "time type": "real|simulation"
}
```


## 2023-11-06
- Add new option `amr strategy` for heat
``` diff
{
  "heat": {
   "problem specific": {
+     "amr strategy": "generic|KellyErrorEstimator"
    }
  }
}
```

## 2023-10-31
- Add new option `sharp_conforming` for evaporative cooling
```json
{
  "evaporation": {
~    "evapor formulation source term heat": "sharp_conforming"
  },
}
```

## 2023-10-30
- Add new option `interface_sharp_conforming` for Gaussian laser heat source
```json
{
  "laser": {
    "laser impact type": "interface_sharp_conforming"
  }
}
```

## 2023-10-16
- Add problem specific parameter to level set problem
``` diff
{
  "base": {
    "problem name": "level_set"
    }, 
+  "problem specific": {
+   "amr":
+     {
+         "strategy": "generic|refine_all_interface_cells"
+     }
+  }
}
```

## 2023-08-30
- Add alternative type for closest point projection with collinearity
```diff
{
  "levelset": {
+    "nearest point":
    {
+       "type": "closest_point_normal_collinear_coquerelle",
    }
  }
}
```

## 2023-08-31
- Removed revised_gradient_based absorption model
``` diff
  {
  "rte": {
-    "absorptivity type": "constant|gradient_based|revised_gradient_based",
+    "absorptivity type": "constant|gradient_based",
  }
```

## 2023-08-17
- Add a parameter Radiative Transfer Equation that prevent a singular matrix when run in parallel
``` diff
  {
  "rte": {
+    "avoid singular matrix absorptivity": "",
    },
  }
```

## 2023-08-03
- Add some parameters related to usage of pseudo-time problem within radiative_transport. Add parameter to choose Absorption coefficient formulation.
``` diff
  {
  "rte": {
+    "problem type": "plain|time_dependent_predictor|time_dependent_problem",
+    "absorptivity type": "constant|gradient_based|revised_gradient_based",
+    "absorptivity": {
+      "absorptivity gas": "",
+      "absorptivity liquid": "",
+      "avoid div zero constant" : ""
+    },
+    "pseudo time stepping": {
+      "diffusion term scaling": "",
+      "advection term scaling": "",
+      "time step size": "",
+      "pseudo time scaling": "",
+      "max n steps": "",
+      "rel tolerance": ""
+      "linear solver": {
+       },
+    },
  }
```
- Add parameters for simulation: radiative_transport
``` diff
{
  "base": {
    "application name": "radiative_transport",
    "problem name": "radiative_transport"
    },
  "simulation specific parameters": {
+    "interface case": "straight|single_powder_particle|powderbed",
+    "center in": "",
+    "radius in": "",
+    "power in": "",
+    "powder particle radius": "",
+    "offset": ""
  }
}
```

## 2023-08-01
- Remove unused parameters, resolve redundant parameter do melt pool and 
restructure analytical laser section
```diff
{
  "base": {
    "problem name": "melt_pool"
    },
  "problem specific": {
-    "do melt pool": ""
  },
  "melt pool": {
     "mp liquid melt pool radius",
     "mp liquid melt pool depth",
  },
  "laser": {
-    "laser analytical max temperature": "3500.",
-    "laser analytical absorptivity gas": "0",
-    "laser analytical absorptivity liquid": "0.5",
-    "laser analytical ambient temperature": "500.",
-    "laser analytical temperature x to y ratio": "3."
+    "analytical": {
+      "absorptivity liquid": "0.5",
+      "absorptivity gas": "0",
+      "ambient temperature": "500.",
+      "max temperature": "3500.",
+      "temperature x to y ratio": "3."
    }
  }
}
```

## 2023-07-05
- Move some heat parameters to simulation/problem specific section
```diff
{
  "heat": {
-    "heat velocity": "",
-    "heat two phase": "",
-    "heat solidification": ""
  }
```
- problem: melt_pool
```diff
{
  "base": {
    "problem name": "melt_pool"
    }
  "problem specific": {
+    "do solidification": "true/false"
  }
}
```
- problem: heat_transfer
```diff
{
  "base": {
    "problem name": "heat_transfer"
    }
+  "problem specific": {
+    "do solidification": "true/false"
+  }
```
- simulation: melt_front_propagation
```diff
{
  "base": {
    "application name": "melt_front_propagation"
    }
+  "simulation specific parameters": {
+    "do two phase": "true/false"
+  }
```
- simulation: unidirectional_heat_transfer
```diff
{
  "base": {
    "application name": "unidirectional_heat_transfer"
    }
  "simulation specific": {
+    "do solidification": "true/false",
+    "do two phase": "true/false",
+    "velocity": ""
  }
```

## 2023-05-09
- Add parameters for Radiative Transfer Equation
```json
  {
  "rte": {
    "verbosity level": "",
    "laser direction": ",",
    "absorptivity gas": "",
    "absorptivity liquid": "",
    "avoid div zero constant": "",
    "linear solver": {
    "monitor type": "",
    "do matrix free": "",
    "abs tolerance": "",
    "preconditioner type": ""
    }
  }
```

## 2023-05-08
- Rename and parameters for calculating nearest points to isocontour
of a level set function
```diff
{
  "levelset": {
+    "nearest point":
-    "closest point projection":
    {
-       "max phi": "",
+       "narrow band threshold": "",
-       "enforce colinearity": "true|false",
+       "type": "nearest_point|closest_point_normal|closest_point_normal_collinear",
    }
  }
```

## 2023-03-22
- Add parameters for closest point projection
```diff
{
  "levelset": {
    "closest point projection":
    {
+       "enforce colinearity": "true|false",
+       "max phi": "",
    }
  }
```


## 2023-02-20
- Add parameters for closest point projection
```diff
{
  "levelset": {
+    "closest point projection":
+    {
+        "max iter": "",
+        "rel tol": "",
+    }
  }
  "evaporation": {
-    "evapor interface value n iterations": "",
  }
```

## 2023-02-15
- Add SUPG to advection diffusion
```diff
{
  "advection diffusion": {
     "convection stabilization":
     {
        "type": "SUPG|none",
        "coefficient": "",
     }
  }
```


## 2023-02-08
- Add option for calculating transport velocity of level set
```diff
{
  "evaporation": {
+   "evapor level set source term type": "interface_velocity_sharp_heavy" 
  }
}

```

## 2023-01-14
- Add option for calculating transport velocity of level set
```diff
{
  "evaporation": {
+   "evapor level set source term type": "interface_velocity_sharp" 
  }
}

```

## 2023-01-13
- Remove redundant parameter
```diff
  "levelset": {
-   "ls artificial diffusivity": ""
  }
}
```

## 2022-12-08
- Add parameter to the heat operator for individual parameter interpolation 
of the conductivity (k) and/or the volumetric heat capacity (rho times cp)
```diff
{
  "heat": {
~    "interpolate rho times cp": "none|sharp|smooth|reciprocal"
+    "interpolate k": "none|sharp|smooth|reciprocal"
  }
}
```

## 2022-12-08
- Add parameter to specify outlet pressure
```diff
{
  "base": {
    "application name": "recoil_pressure"
  },
  "simulation specific domain": {
    "evaporation boundary": "true"
+   "outlet pressure": ""
  }
}

```
## 2022-12-07
- Modify restart parameters
```diff
{
  "restart":
  {
-   "write frequency": "",
    "time type": "real|simulation",
  }
}

## 2022-12-05
- Add new Dirac delta function type
```diff
{
  "dirac delta function approximation": {
+    "type": "heavy_phase_only"
  }
}
```

## 2022-12-02
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
