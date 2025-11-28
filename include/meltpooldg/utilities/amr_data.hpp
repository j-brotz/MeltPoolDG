#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  template <typename number>
  struct AdaptiveMeshingData
  {
    bool         do_amr                                     = false;
    bool         do_not_modify_boundary_cells               = false;
    number       upper_perc_to_refine                       = 0.0;
    number       lower_perc_to_coarsen                      = 0.0;
    int          n_initial_refinement_cycles                = 0;
    int          every_n_step                               = 1;
    unsigned int max_grid_refinement_level                  = 12;
    int          min_grid_refinement_level                  = -1;
    int          min_cells_marked_for_refinement            = 1;
    number       indicator_min_value_to_mark_for_refinement = 0.0;
    bool         solution_transfer_average_values           = false;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const unsigned int global_refinements, const bool load_restart_data);
  };

} // namespace MeltPoolDG
