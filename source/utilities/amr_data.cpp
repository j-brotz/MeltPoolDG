#include <meltpooldg/utilities/amr_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  AdaptiveMeshingData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("adaptive meshing");
    {
      prm.add_parameter("do amr",
                        do_amr,
                        "Set this parameter to true to activate adaptive meshing");
      prm.add_parameter("do not modify boundary cells",
                        do_not_modify_boundary_cells,
                        "Set this parameter to true to not refine/coarsen along boundaries.");
      prm.add_parameter("upper perc to refine",
                        upper_perc_to_refine,
                        "Defines the (upper) percentage of elements that should be refined");
      prm.add_parameter("lower perc to coarsen",
                        lower_perc_to_coarsen,
                        "Defines the (lower) percentage of elements that should be coarsened");
      prm.add_parameter(
        "max grid refinement level",
        max_grid_refinement_level,
        "Defines the number of maximum refinement steps one grid cell will be undergone.");
      prm.add_parameter(
        "min grid refinement level",
        min_grid_refinement_level,
        "Defines the number of minimum refinement steps one grid cell will be undergone.");
      prm.add_parameter("n initial refinement cycles",
                        n_initial_refinement_cycles,
                        "Defines the number of initial refinements.");
      prm.add_parameter("every n step",
                        every_n_step,
                        "Defines at every nth step the amr should be performed.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  AdaptiveMeshingData<number>::post(const unsigned int global_refinements,
                                    const bool         load_restart_data)
  {
    // set the min grid refinement level if not user-specified
    if (min_grid_refinement_level == 1)
      min_grid_refinement_level = global_refinements;
    // do not allow initial refinement cycles in case of restart load
    if (load_restart_data)
      n_initial_refinement_cycles = 0;
  }

  template struct AdaptiveMeshingData<double>;
} // namespace MeltPoolDG
