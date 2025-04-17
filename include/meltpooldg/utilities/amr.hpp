#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/utilities/amr_data.hpp>

#include <functional>
#include <utility>
#include <vector>


namespace MeltPoolDG::AMR
{
  template <int dim>
  using MarkCellsForRefinementType = std::function<bool(dealii::Triangulation<dim> &)>;

  template <int dim, typename VectorType>
  using DoFHandlerAndVectorDataType = std::vector<
    std::pair<const dealii::DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>;

  template <int dim, typename VectorType>
  using AttachDoFHandlerAndVectorsType =
    std::function<void(DoFHandlerAndVectorDataType<dim, VectorType> &)>;

  template <typename number>
  inline bool
  now(const AdaptiveMeshingData<number> &amr, const int n_time_step)
  {
    return not(n_time_step % amr.every_n_step) or n_time_step == 0;
  }

  /**
   * TODO document what needs to be done in each of the lambdas
   */
  template <int dim, typename VectorType, typename number = VectorType::value_type>
  void
  refine_grid(const MarkCellsForRefinementType<dim>                 &mark_cells_for_refinement,
              const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
              const std::function<void()>                           &post,
              const std::function<void()>                           &setup_dof_system,
              const AdaptiveMeshingData<number>                     &amr,
              dealii::Triangulation<dim>                            &tria,
              const int                                              n_time_step);

  /**
   * Same as above, but for only one @param dof_handler
   */
  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const std::function<bool(dealii::Triangulation<dim> &)> &mark_cells_for_refinement,
              const std::function<void(std::vector<VectorType *> &)>  &attach_vectors,
              const std::function<void()>                             &post,
              const std::function<void()>                             &setup_dof_system,
              const AdaptiveMeshingData<number>                       &amr,
              dealii::DoFHandler<dim>                                 &dof_handler,
              const int                                                n_time_step);

  namespace internal
  {
    // Limit the maximum and minimum refinement levels of cells of the grid and do not
    // coarsen/refine cells along boundary
    template <int dim, typename number>
    void
    limit_amr(dealii::Triangulation<dim> &tria, const AdaptiveMeshingData<number> &amr_data);
  } // namespace internal
} // namespace MeltPoolDG::AMR
