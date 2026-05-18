#include <meltpooldg/utilities/amr.hpp>
//
#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools_common.h>

#include <memory>

namespace MeltPoolDG::AMR
{
  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const MarkCellsForRefinementType<dim>                 &mark_cells_for_refinement,
              const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
              const std::function<void()>                           &setup_dof_system,
              const AdaptiveMeshingData<number>                     &amr,
              dealii::Triangulation<dim>                            &tria,
              const int                                              n_time_step,
              const std::function<void()>                           &post,
              const std::function<void()>                           &pre)
  {
    if (not now(amr, n_time_step))
      return;

    if (mark_cells_for_refinement(tria) == false)
      return;

    internal::limit_amr(tria, amr);

    if (not attach_vectors)
      {
        // short circuit this function
        if (pre)
          pre();
        tria.prepare_coarsening_and_refinement();
        tria.execute_coarsening_and_refinement();
        setup_dof_system();
        if (post)
          post();
        return;
      }

    // the following is very similar to the code in
    // CutUtil::SolutionTransferOperator::transfer_solution_constant_dofs()

    DoFHandlerAndVectorDataType<dim, VectorType> data;

    attach_vectors(data);
    data.shrink_to_fit();

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    // Initialize the triangulation change from the old grid to the new grid
    if (pre)
      pre();
    tria.prepare_coarsening_and_refinement();

    // Initialize the solution transfer from the old grid to the new grid
    std::vector<std::unique_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers(
      n_dof_handlers);

    std::vector<std::vector<VectorType *>>       new_grid_solutions(n_dof_handlers);
    std::vector<std::vector<const VectorType *>> old_grid_solutions(n_dof_handlers);
    std::vector<std::vector<bool>>               update_ghost_values(n_dof_handlers);

    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        // collect pointers to the DoF vectors for the current DoFHandler
        data[j].second(new_grid_solutions[j]);

        old_grid_solutions[j].resize(new_grid_solutions[j].size());
        update_ghost_values[j].resize(new_grid_solutions[j].size(), true);
        for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
          {
            old_grid_solutions[j][i] = new_grid_solutions[j][i];
            if (not new_grid_solutions[j][i]->has_ghost_elements())
              {
                update_ghost_values[j][i] = false;
                new_grid_solutions[j][i]->update_ghost_values();
              }
          }

        solution_transfers[j] = std::make_unique<dealii::SolutionTransfer<dim, VectorType>>(
          *data[j].first, amr.solution_transfer_average_values);
        solution_transfers[j]->prepare_for_coarsening_and_refinement(old_grid_solutions[j]);
      }

    tria.execute_coarsening_and_refinement();

    // update dof-related scratch data to match the current triangulation
    setup_dof_system();

    // interpolate the given solution to the new discretization
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        for (const auto &v : new_grid_solutions[j])
          v->zero_out_ghost_values();

        solution_transfers[j]->interpolate(new_grid_solutions[j]);

        for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
          if (update_ghost_values[j][i])
            new_grid_solutions[j][i]->update_ghost_values();
      }

    if (post)
      post();
  }


  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const MarkCellsForRefinementType<dim>                  &mark_cells_for_refinement,
              const std::function<void(std::vector<VectorType *> &)> &attach_vectors,
              const std::function<void()>                            &setup_dof_system,
              const AdaptiveMeshingData<number>                      &amr,
              dealii::DoFHandler<dim>                                &dof_handler,
              const int                                               n_time_step,
              const std::function<void()>                            &post,
              const std::function<void()>                            &pre)
  {
    refine_grid<dim, VectorType>(
      mark_cells_for_refinement,
      [&](DoFHandlerAndVectorDataType<dim, VectorType> &data) {
        data.emplace_back(&dof_handler, attach_vectors);
      },
      setup_dof_system,
      amr,
      const_cast<dealii::Triangulation<dim> &>(dof_handler.get_triangulation()),
      n_time_step,
      post,
      pre);
  }

  template <int dim, typename number>
  bool
  mark_fixed_number(dealii::Triangulation<dim>            &tria,
                    const dealii::Vector<number>          &indicator,
                    const AdaptiveMeshingData<number>     &amr_data,
                    const dealii::types::global_cell_index max_n_cells)
  {
    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      tria, indicator, amr_data.upper_perc_to_refine, amr_data.lower_perc_to_coarsen, max_n_cells);

    if (amr_data.indicator_min_value_to_mark_for_refinement > 0.)
      internal::clear_refine_flags_below_threshold(
        tria, indicator, amr_data.indicator_min_value_to_mark_for_refinement);

    return internal::do_refine_based_on_n_marked_cells(tria,
                                                       amr_data.min_cells_marked_for_refinement);
  }

  template <int dim, typename number>
  bool
  mark_fixed_fraction(dealii::Triangulation<dim>        &tria,
                      const dealii::Vector<number>      &indicator,
                      const AdaptiveMeshingData<number> &amr_data)
  {
    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      tria,
      indicator,
      amr_data.upper_perc_to_refine,
      amr_data.lower_perc_to_coarsen,
      dealii::VectorTools::L1_norm);

    if (amr_data.indicator_min_value_to_mark_for_refinement > 0.)
      internal::clear_refine_flags_below_threshold(
        tria, indicator, amr_data.indicator_min_value_to_mark_for_refinement);

    return internal::do_refine_based_on_n_marked_cells(tria,
                                                       amr_data.min_cells_marked_for_refinement);
  }

  template void
  refine_grid(
    const MarkCellsForRefinementType<1> &,
    const AttachDoFHandlerAndVectorsType<1, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::Triangulation<1> &,
    const int,
    const std::function<void()> &,
    const std::function<void()> &);
  template void
  refine_grid(
    const MarkCellsForRefinementType<2> &,
    const AttachDoFHandlerAndVectorsType<2, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::Triangulation<2> &,
    const int,
    const std::function<void()> &,
    const std::function<void()> &);
  template void
  refine_grid(
    const MarkCellsForRefinementType<3> &,
    const AttachDoFHandlerAndVectorsType<3, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::Triangulation<3> &,
    const int,
    const std::function<void()> &,
    const std::function<void()> &);



  template void
  refine_grid(
    const MarkCellsForRefinementType<1> &,
    const std::function<void(std::vector<dealii::LinearAlgebra::distributed::Vector<double> *> &)>
      &,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::DoFHandler<1> &,
    const int,
    const std::function<void()> &,
    const std::function<void()> &);
  template void
  refine_grid(
    const MarkCellsForRefinementType<2> &,
    const std::function<void(std::vector<dealii::LinearAlgebra::distributed::Vector<double> *> &)>
      &,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::DoFHandler<2> &,
    const int,
    const std::function<void()> &,
    const std::function<void()> &);
  template void
  refine_grid(
    const MarkCellsForRefinementType<3> &,
    const std::function<void(std::vector<dealii::LinearAlgebra::distributed::Vector<double> *> &)>
      &,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::DoFHandler<3> &,
    const int,
    const std::function<void()> &,
    const std::function<void()> &);

  template bool
  mark_fixed_number(dealii::Triangulation<1>          &tria,
                    const dealii::Vector<double>      &indicator,
                    const AdaptiveMeshingData<double> &amr_data,
                    dealii::types::global_cell_index   max_n_cells);
  template bool
  mark_fixed_number(dealii::Triangulation<2>          &tria,
                    const dealii::Vector<double>      &indicator,
                    const AdaptiveMeshingData<double> &amr_data,
                    dealii::types::global_cell_index   max_n_cells);
  template bool
  mark_fixed_number(dealii::Triangulation<3>          &tria,
                    const dealii::Vector<double>      &indicator,
                    const AdaptiveMeshingData<double> &amr_data,
                    dealii::types::global_cell_index   max_n_cells);

  template bool
  mark_fixed_fraction(dealii::Triangulation<1>          &tria,
                      const dealii::Vector<double>      &indicator,
                      const AdaptiveMeshingData<double> &amr_data);
  template bool
  mark_fixed_fraction(dealii::Triangulation<2>          &tria,
                      const dealii::Vector<double>      &indicator,
                      const AdaptiveMeshingData<double> &amr_data);
  template bool
  mark_fixed_fraction(dealii::Triangulation<3>          &tria,
                      const dealii::Vector<double>      &indicator,
                      const AdaptiveMeshingData<double> &amr_data);

  namespace internal
  {
    template <int dim, typename number>
    void
    limit_amr(dealii::Triangulation<dim> &tria, const AdaptiveMeshingData<number> &amr_data)
    {
      // Limit the maximum and minimum refinement levels of cells of the grid.
      if (tria.n_levels() > amr_data.max_grid_refinement_level)
        for (auto &cell : tria.active_cell_iterators_on_level(amr_data.max_grid_refinement_level))
          if (cell->is_locally_owned())
            cell->clear_refine_flag();
      if (amr_data.min_grid_refinement_level > 0)
        for (auto &cell : tria.active_cell_iterators_on_level(amr_data.min_grid_refinement_level))
          if (cell->is_locally_owned())
            cell->clear_coarsen_flag();

      // do not coarsen/refine cells along boundary
      if (amr_data.do_not_modify_boundary_cells)
        for (auto &cell : tria.active_cell_iterators())
          {
            if (not cell->is_locally_owned())
              continue;

            for (auto &face : cell->face_iterators())
              {
                if (not face->at_boundary())
                  continue;
                if (cell->refine_flag_set())
                  cell->clear_refine_flag();
                else
                  cell->clear_coarsen_flag();
              }
          }
    }

    template <int dim>
    bool
    do_refine_based_on_n_marked_cells(const dealii::Triangulation<dim> &tria,
                                      const unsigned                    min_cells)
    {
      unsigned n_coarsen_or_refine = 0;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          if (cell->refine_flag_set() or cell->coarsen_flag_set())
            ++n_coarsen_or_refine;

      MPI_Allreduce(MPI_IN_PLACE, &n_coarsen_or_refine, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

      return n_coarsen_or_refine >= min_cells;
    }

    template <int dim, typename number>
    void
    clear_refine_flags_below_threshold(dealii::Triangulation<dim>   &tria,
                                       const dealii::Vector<number> &indicator,
                                       const number                  threshold)
    {
      for (auto &cell : tria.active_cell_iterators())
        if (cell->refine_flag_set())
          if (indicator[cell->active_cell_index()] < threshold)
            cell->clear_refine_flag();
    }

    template void
    limit_amr(dealii::Triangulation<1> &, const AdaptiveMeshingData<double> &);
    template void
    limit_amr(dealii::Triangulation<2> &, const AdaptiveMeshingData<double> &);
    template void
    limit_amr(dealii::Triangulation<3> &, const AdaptiveMeshingData<double> &);

    template bool
    do_refine_based_on_n_marked_cells(const dealii::Triangulation<1> &tria, unsigned min_cells);
    template bool
    do_refine_based_on_n_marked_cells(const dealii::Triangulation<2> &tria, unsigned min_cells);
    template bool
    do_refine_based_on_n_marked_cells(const dealii::Triangulation<3> &tria, unsigned min_cells);

    template void
    clear_refine_flags_below_threshold(dealii::Triangulation<1>     &tria,
                                       const dealii::Vector<double> &indicator,
                                       double                        threshold);
    template void
    clear_refine_flags_below_threshold(dealii::Triangulation<2>     &tria,
                                       const dealii::Vector<double> &indicator,
                                       double                        threshold);
    template void
    clear_refine_flags_below_threshold(dealii::Triangulation<3>     &tria,
                                       const dealii::Vector<double> &indicator,
                                       double                        threshold);
  } // namespace internal
} // namespace MeltPoolDG::AMR
