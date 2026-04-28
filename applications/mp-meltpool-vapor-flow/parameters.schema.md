# mp-meltpool-vapor-flow: Parameter description

## Contents

- [`base`](#base)
- [`time stepping`](#time-stepping)
- [`output`](#output)
- [`profiling`](#profiling)

---

<a id="base"></a>
## `🔷 base`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `case name` | `string` | `not_initialized` | Sets the base name for the application that will be fed to the problem type. |
| `dimension` | `integer` | `2` | Defines the dimension of the problem |
| `number` | `string` | `double` | Floating point number format. Currently, only 'double' is explicitely instantiated.<br><br>Allowed values:<br>- `double` |
| `global refinements` | `integer` | `1` | Defines the number of initial global refinements |
| `do print parameters` | `boolean` | `True` | Set this parameter to true to list parameters in output |
| `verbosity level` | `integer` | `1` | Sets the verbosity level of the console output: 0: silent: for non-robust tests and benchmark runs; 1: minimal: for robust tests; 2: detailed; 3: full |
| [`fe`](#base-fe) | `object` |  | [See table](#base-fe) |

<a id="base-fe"></a>
### `base: fe`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `FE_Q` | Finite Element.FE_Q: hexahedral continuous finite element with polynomial degree p; FE_SimplexP: tetrahedral continuous finite element with polynomial degree p; FE_Q_iso_Q1: hexahedral continuous finite element with p subdivisions containing linear elements; FE_DGQ: hexahedral discontinuous finite element with polynomial degree p<br><br>Allowed values:<br>- `not_initialized`<br>- `FE_Q`<br>- `FE_SimplexP`<br>- `FE_Q_iso_Q1`<br>- `FE_DGQ` |
| `degree` | `integer` | `1` | Defines the degree p of the finite element type. If "type" is "FE_Q_iso_Q1" this parameter defines the number of subdivisions. |


---

<a id="time-stepping"></a>
## `🔷 time stepping`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start time` | `number` | `0.0` | Defines the start time for the solution of the levelset problem |
| `end time` | `number` | `1.0` | Sets the end time for the solution of the levelset problem |
| `time step size` | `number` | `0.01` | Sets the step size for time stepping. For non-uniform time stepping, this parameter determines the size of the first time step. |
| `max n steps` | `integer` | `10000000` | Sets the maximum number of melt_pool steps |
| `time step size function` | `string` | `0.0*t` | Set an analytical function to determine the time step size. For the prediction of the new time increment, the old time is used. |


---

<a id="output"></a>
## `🔷 output`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `directory` | `string` | `./` | Sets the base directory for all output. |
| `write frequency` | `integer` | `1` | Every n timestep that should be written |
| `write time step size` | `number` | `1.79769e+308` | Write output output every given time step. If this parameter is set, the output write frequency is deactivated. |
| `output variables` | `array` | `['all']` | Specify variables that you request to output. |
| `do user defined postprocessing` | `boolean` | `False` | Set this parameter to true to enable user defined postprocessing. |
| [`paraview`](#output-paraview) | `object` |  | [See table](#output-paraview) |
| [`particles`](#output-particles) | `object` |  | [See table](#output-particles) |

<a id="output-paraview"></a>
### `output: paraview`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to activate paraview output. |
| `filename` | `string` | `solution` | Sets the base name for paraview output files. |
| `n digits timestep` | `integer` | `4` | Number of digits for the frame number of the vtu-file. |
| `print boundary id` | `boolean` | `False` | Set this parameter to true to output a vtu-file with the boundary id. |
| `output subdomains` | `boolean` | `False` | Set this parameter to true to output the subdomain ranks. |
| `output material id` | `boolean` | `False` | Set to true to output the material id. |
| `write higher order cells` | `boolean` | `True` | Set this parameter to false to write bi- or trilinear data only. Set this parameter to true to write higher order cell data. Note: higher order cell data can only be written for hexahedron meshes and 2 or 3 dimensions. |
| `n groups` | `integer` | `1` | Number of parallel written vtu-files. |
| `n patches` | `integer` | `0` | Control number of patches to enable high-order. |

<a id="output-particles"></a>
### `output: particles`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true to activate particle paraview output. |
| `filename` | `string` | `particle` | Sets the base name for particle output files. |


---

<a id="profiling"></a>
## `🔷 profiling`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `boolean` | `False` | Set this parameter to true if profiling should be enabled. It will be automaticallyenabled for verbosity level >=1. |
| `write time step size` | `number` | `10.0` | Write profiling output every given time step size. If this parameter is set, the specified parameter for write frequency is overwritten. |
| `time type` | `string` | `real` | Choose the type of time measure to write profiling information.<br><br>Allowed values:<br>- `real`<br>- `simulation` |


---
