#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/amr_data.hpp>

#include <functional>


namespace MeltPoolDG::CutUtil
{
  /**
   * TODO document what needs to be done in each of the lambdas
   */
  template <int dim, typename VectorType, typename number = VectorType::value_type>
  void
  refine_grid(const AMR::MarkCellsForRefinementType<dim>                 &mark_cells_for_refinement,
              const AMR::AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
              const std::function<void()>                                &post,
              const std::function<void()>                                &distribute_level_set_dofs,
              const dealii::DoFHandler<dim>                              *level_set_dof_handler,
              const std::function<void()>       &setup_remaining_dof_system,
              const AdaptiveMeshingData<number> &amr,
              dealii::Triangulation<dim>        &tria,
              const int                          n_time_step);
} // namespace MeltPoolDG::CutUtil