{
  "base": {
    "case name": "wall_wetting",
    "dimension": "2",
    "global refinements": "4",
    "verbosity level": "3",
    "fe": {
      "degree": "1"
    }
  },
  "time stepping": {
    "start time": "0.0",
    "end time": "END_TIME"
  },
  "output": {
    "directory": "../../meltpooldg_results/CASE_NAME",
    "do user defined postprocessing": "true",
    "paraview": {
      "filename": "wall_wetting",
      "enable": "true"
    }
  },
  "reinitialization": {
    "linear solver": {
      "do matrix free": "true",
      "preconditioner type": "Diagonal"
    },
    "type": "olsson2007",
    "interface thickness parameter": {
      "type": "proportional_to_cell_size",
      "val": "EPSILON_N_FACTOR"
    },
    "tangential diffusion factor": "EPSILON_T_FACTOR"
  },
  "normal vector": {
    "compute normalized vector": "true",
    "linear solver": {
      "do matrix free": "true",
      "preconditioner type": "AMG"
    }
  },
  "simulation specific": {
    "contact angle": "STATIC_CONTACT_ANGLE",
    "gamma factor": "GAMMA_FACTOR"
  }
}
