#include <meltpooldg/utilities/cut_solution_transfer.hpp>
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/numerics/solution_transfer.h>

#include <meltpooldg/utilities/cut_util.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <iomanip>
#include <sstream>
#include <vector>


namespace MeltPoolDG::CutUtil
{
  template <int dim, typename Number>
  SolutionTransferOperator<dim, Number>::SolutionTransferOperator(const Number gamma_degree_0,
                                                                  const Number gamma_degree_1,
                                                                  const Number gamma_degree_2,
                                                                  const bool   is_two_phase,
                                                                  const int    verbosity)
    : gamma_degree_0(gamma_degree_0)
    , gamma_degree_1(gamma_degree_1)
    , gamma_degree_2(gamma_degree_2)
    , is_two_phase(is_two_phase)
    , verbosity(verbosity)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}



  template <int dim, typename Number>
  void
  SolutionTransferOperator<dim, Number>::reinit(
    dealii::DoFHandler<dim>                                                  &dof_handler,
    dealii::Triangulation<dim>                                               &tria,
    const VectorType                                                         &solution,
    const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier_old,
    const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier,
    const std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> &reinit_vector,
    const std::function<void(const dealii::DoFHandler<dim> &)>               &reinit_matrix_free)
  {
    solution.update_ghost_values();

    fe_degree = dof_handler.get_fe_collection().max_degree();

    n_components_per_phase = dof_handler.get_fe_collection().n_components();
    if (is_two_phase)
      {
        AssertThrow(
          n_components_per_phase % 2 == 0,
          dealii::ExcMessage(
            "For a two phase problem, both phases must contain the same number of solution components."));
        n_components_per_phase /= 2;
      }

    // Check if the finite element is of type FE_DGQ or FE_Q.
    // Assert if different FE types are defined in the hp::FECollection object.
    {
      unsigned int dg_counter = 0;
      unsigned int cg_counter = 0;
      if (n_components_per_phase == 1 and not is_two_phase)
        for (const auto &fe : dof_handler.get_fe_collection())
          {
            if (typeid(fe) == typeid(dealii::FE_Nothing<dim>))
              continue;
            if (typeid(fe) == typeid(dealii::FE_DGQ<dim>))
              ++dg_counter;
            else if (typeid(fe) == typeid(dealii::FE_Q<dim>))
              ++cg_counter;
            else
              AssertThrow(false, dealii::ExcMessage("You did not provide a support FE type."));
          }
      else // multi-component or two phase
        for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
          {
            const auto &fe_system =
              dynamic_cast<const dealii::FESystem<dim> &>(dof_handler.get_fe_collection()[i]);
            for (unsigned int j = 0; j < fe_system.n_components(); ++j)
              {
                const auto &fe = fe_system.get_sub_fe(j, 1);

                if (typeid(fe) == typeid(dealii::FE_Nothing<dim>))
                  continue;
                if (typeid(fe) == typeid(dealii::FE_DGQ<dim>))
                  ++dg_counter;
                else if (typeid(fe) == typeid(dealii::FE_Q<dim>))
                  ++cg_counter;
                else
                  AssertThrow(false, dealii::ExcMessage("You did not provide a support FE type."));
              }
          }

      if (dg_counter != 0 and cg_counter == 0)
        is_dg = true;
      else if (dg_counter == 0 and cg_counter != 0)
        is_dg = false;
      else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "Two different FE types in the hp::FECollection object are not supported."));
    }

    // in case of a single phase problem, check that no cells are 'outside'
    if (not is_two_phase)
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned())
            continue;

          const auto cell_location = mesh_classifier_old.location_to_level_set(cell);

          AssertThrow(
            cell_location != dealii::NonMatching::LocationToLevelSet::inside or
              cell->get_fe().n_dofs_per_cell() == 0,
            dealii::ExcMessage(
              "The active domain should have the FE index 'outside' for a single domain problem."));
        }

    // 1) update FE-index and distribute DoFs according to new state with the moved interface
    transfer_solution_constant_dofs(dof_handler,
                                    tria,
                                    solution,
                                    mesh_classifier_old,
                                    mesh_classifier,
                                    reinit_vector,
                                    reinit_matrix_free);

    // 2) set-up and solve system for ghost-penalty extrapolation to determine the values of the
    //    remaining undetermined DoFs
    extrapolate_solution_new_dofs(dof_handler, mesh_classifier, mesh_classifier_old, reinit_vector);
  }



  template <int dim, typename Number>
  void
  SolutionTransferOperator<dim, Number>::transfer_solution_constant_dofs(
    dealii::DoFHandler<dim>                                                  &dof_handler,
    dealii::Triangulation<dim>                                               &tria,
    const VectorType                                                         &solution,
    const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier_old,
    const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier,
    const std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> &reinit_vector,
    const std::function<void(const dealii::DoFHandler<dim> &)>               &reinit_matrix_free)
  {
    // update the future FE-index according to the new interface position
    CutUtil::set_fe_index<dim>(dof_handler, mesh_classifier, true /* set_future */);

    dealii::SolutionTransfer<dim, VectorType> solution_transfer(dof_handler,
                                                                true /*average_values*/);

    // prepare solution transfer
    solution_transfer.prepare_for_coarsening_and_refinement(solution);

    // needed to trigger hp refinement --> call call-back function created by
    // solution transfer
    tria.execute_coarsening_and_refinement();

    // distribute DoFs according to new state
    dof_handler.distribute_dofs(dof_handler.get_fe_collection());

    // reinitialize the matrix-free object (dof_handler has changed)
    if (reinit_matrix_free)
      reinit_matrix_free(dof_handler);

    // reinit new solution vector
    reinit_vector(new_solution, dof_handler);
    new_solution = 0.;

    // transfer values to new_solution
    solution_transfer.interpolate(new_solution);

    new_solution.update_ghost_values();

    // revert averaging process in solution transfer for cg-case
    // (TODO: fix this issue in dealii)
    if (!is_dg)
      {
        VectorType dof_counter;
        reinit_vector(dof_counter, dof_handler);
        dof_counter = 0.;

        VectorType dof_counter_fe_nothing;
        reinit_vector(dof_counter_fe_nothing, dof_handler);
        dof_counter_fe_nothing = 0.;

        std::vector<dealii::types::global_dof_index> dof_indices;
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (not cell->is_locally_owned() and not cell->is_ghost())
              continue;

            dof_indices.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(dof_indices);

            for (const auto i : dof_indices)
              if (dof_counter.in_local_range(i))
                dof_counter[i] += 1.0;

            const auto cell_location = mesh_classifier.location_to_level_set(cell);

            if (cell_location != dealii::NonMatching::LocationToLevelSet::intersected)
              continue;

            const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);

            const auto &fe = cell->get_fe();

            std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            if (cell_location_old == dealii::NonMatching::LocationToLevelSet::inside)
              for (unsigned int i = 0; i < n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (dof_counter_fe_nothing.in_local_range(dof_index))
                      dof_counter_fe_nothing[dof_index] += 1;
                  }
            else if ((cell_location_old == dealii::NonMatching::LocationToLevelSet::outside) &&
                     (is_two_phase == true))
              for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (dof_counter_fe_nothing.in_local_range(dof_index))
                      dof_counter_fe_nothing[dof_index] += 1;
                  }
          }

        for (const auto i : new_solution.locally_owned_elements())
          if ((dof_counter[i] - dof_counter_fe_nothing[i]) > 0)
            new_solution[i] *= dof_counter[i] / (dof_counter[i] - dof_counter_fe_nothing[i]);

        new_solution.update_ghost_values();
      }
  }



  template <int dim, typename Number>
  dealii::LinearAlgebra::distributed::Vector<Number>
  SolutionTransferOperator<dim, Number>::mark_dofs_for_gp_extrapolation(
    const dealii::DoFHandler<dim>                  &dof_handler,
    const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
    const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old) const
  {
    VectorType flags_dofs_gp_extrapolation;
    flags_dofs_gp_extrapolation.reinit(dof_handler.locally_owned_dofs(),
                                       dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                       dof_handler.get_communicator());
    flags_dofs_gp_extrapolation = 0.;

    // a) Loop over intersected cells that previously did not belong to the
    //    subdomain and collect the DoFs
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() and not cell->is_ghost())
          continue;

        const auto cell_location = mesh_classifier.location_to_level_set(cell);

        if (cell_location != dealii::NonMatching::LocationToLevelSet::intersected)
          continue;

        const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);

        const auto &fe = cell->get_fe();

        std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);

        if (cell_location_old == dealii::NonMatching::LocationToLevelSet::inside)
          for (unsigned int i = 0; i < n_components_per_phase; ++i)
            for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
              {
                const auto dof_index =
                  local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                  flags_dofs_gp_extrapolation[dof_index] = 1;
              }
        else if ((cell_location_old == dealii::NonMatching::LocationToLevelSet::outside) &&
                 (is_two_phase == true))
          for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
            for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
              {
                const auto dof_index =
                  local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                  flags_dofs_gp_extrapolation[dof_index] = 1;
              }
      }

    flags_dofs_gp_extrapolation.update_ghost_values();

    // b) For continuous elements, loop over cells belonging to the subdomain and remove DoFs
    //    from extrapolation. Additionally, if both the old and new cell location are intersected,
    //    remove DoFs from extrapolation.
    if (!is_dg)
      {
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (not cell->is_locally_owned() and not cell->is_ghost())
              continue;

            const auto cell_location = mesh_classifier.location_to_level_set(cell);

            const auto &fe = cell->get_fe();

            std::vector<dealii::types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            if (cell_location == dealii::NonMatching::LocationToLevelSet::outside)
              for (unsigned int i = 0; i < n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                      flags_dofs_gp_extrapolation[dof_index] = 0;
                  }
            else if ((cell_location == dealii::NonMatching::LocationToLevelSet::inside) &&
                     (is_two_phase == true))
              for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
                for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                  {
                    const auto dof_index =
                      local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                    if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                      flags_dofs_gp_extrapolation[dof_index] = 0;
                  }
            else if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected)
              {
                const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);
                if (cell_location_old == dealii::NonMatching::LocationToLevelSet::intersected)
                  {
                    for (unsigned int i = 0; i < n_components_per_phase; ++i)
                      for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                        {
                          const auto dof_index =
                            local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                          if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                            flags_dofs_gp_extrapolation[dof_index] = 0;
                        }

                    if (is_two_phase == true)
                      {
                        for (unsigned int i = n_components_per_phase;
                             i < 2 * n_components_per_phase;
                             ++i)
                          for (unsigned int q = 0; q < fe.get_sub_fe(i, 1).n_dofs_per_cell(); ++q)
                            {
                              const auto dof_index =
                                local_dof_indices[fe.component_to_system_index(i /*component*/, q)];
                              if (flags_dofs_gp_extrapolation.in_local_range(dof_index))
                                flags_dofs_gp_extrapolation[dof_index] = 0;
                            }
                      }
                  }
              }
          }
      }

    return flags_dofs_gp_extrapolation;
  }



  template <int dim, typename Number>
  dealii::AffineConstraints<Number>
  SolutionTransferOperator<dim, Number>::create_constraints_gp_extrapolation(
    const dealii::DoFHandler<dim> &dof_handler,
    const VectorType              &flags_dofs_gp_extrapolation) const
  {
    flags_dofs_gp_extrapolation.update_ghost_values();

    const dealii::Utilities::MPI::Partitioner partitioner_dof(
      dof_handler.locally_owned_dofs(),
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_communicator());

    // Check if the partitioner of new_solution is globally compatible with partitioner_dof.
    // If not, transfer new_solution to default vector stucture.
    // The full set of ghost-values is required for setting the constraints.
    VectorType new_solution_including_all_ghosts;
    const bool is_globally_compatible =
      partitioner_dof.is_globally_compatible(*new_solution.get_partitioner());
    if (!is_globally_compatible)
      {
        new_solution_including_all_ghosts.reinit(dof_handler.locally_owned_dofs(),
                                                 dealii::DoFTools::extract_locally_relevant_dofs(
                                                   dof_handler),
                                                 dof_handler.get_communicator());
        new_solution_including_all_ghosts = new_solution;
        new_solution_including_all_ghosts.update_ghost_values();
      }

    dealii::AffineConstraints<Number> constraints_gp;
    constraints_gp.reinit(dealii::DoFTools::extract_locally_relevant_dofs(dof_handler));

    std::vector<bool> dof_processed(partitioner_dof.locally_owned_size() +
                                      partitioner_dof.n_ghost_indices(),
                                    false);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned() and not cell->is_ghost())
          continue;

        std::vector<dealii::types::global_dof_index> dof_indices(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);

        dealii::Vector<Number> extrapolate_dofs(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_values(flags_dofs_gp_extrapolation, extrapolate_dofs);

        for (unsigned int i = 0; i < extrapolate_dofs.size(); ++i)
          {
            const auto local_dof_idx = partitioner_dof.global_to_local(dof_indices[i]);
            AssertIndexRange(local_dof_idx, dof_processed.size());

            if (dof_processed[local_dof_idx] == false)
              {
                // constrain entries when they are not extrapolated
                if (extrapolate_dofs[i] == 0)
                  {
                    if (!is_globally_compatible)
                      constraints_gp.add_constraint(
                        dof_indices[i], {}, new_solution_including_all_ghosts[dof_indices[i]]);
                    else
                      constraints_gp.add_constraint(dof_indices[i],
                                                    {},
                                                    new_solution[dof_indices[i]]);
                  }
                dof_processed[local_dof_idx] = true;
              }
          }
      }

    Assert(constraints_gp.is_consistent_in_parallel(
             dealii::Utilities::MPI::all_gather(dof_handler.get_communicator(),
                                                dof_handler.locally_owned_dofs()),
             dealii::DoFTools::extract_locally_active_dofs(dof_handler),
             dof_handler.get_communicator()),
           dealii::ExcInternalError());

    constraints_gp.close();

    return constraints_gp;
  }



  template <int dim, typename Number>
  void
  SolutionTransferOperator<dim, Number>::extrapolate_solution_new_dofs(
    const dealii::DoFHandler<dim>                                            &dof_handler,
    const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier,
    const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier_old,
    const std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> &reinit_vector)
  {
    // identify and mark DoFs, which require ghost-penalty extrapolation
    const VectorType flags_dofs_gp_extrapolation =
      mark_dofs_for_gp_extrapolation(dof_handler, mesh_classifier, mesh_classifier_old);

    // check if DoFs have to be extrapolated, otherwise no gp-extrapolation is required
    const Number sum_gp_extrap = flags_dofs_gp_extrapolation.l1_norm();

    // set inhomogeneous Dirichlet constraints for DoFs which do no not require ghost-penalty
    // extrapolation
    const dealii::AffineConstraints<Number> constraints_gp =
      create_constraints_gp_extrapolation(dof_handler, flags_dofs_gp_extrapolation);

    // set-up sparsity pattern
    dealii::TrilinosWrappers::SparsityPattern dsp;
    dsp.reinit(dof_handler.locally_owned_dofs(),
               dof_handler.locally_owned_dofs(),
               dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
               dof_handler.get_communicator());

    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints_gp, false);

    dsp.compress();

    if (sum_gp_extrap > 0)
      {
        std::ostringstream str;
        str << "Number of new DoFs: " << std::setw(15) << sum_gp_extrap;
        Journal::print_line(pcout, str.str(), "cut solution transfer");
      }
    if (verbosity >= 2)
      {
        std::ostringstream str;
        str << "Number of nonzero elements: " << std::setw(15) << dsp.n_nonzero_elements();
        Journal::print_line(pcout, str.str(), "cut solution transfer");
      }

    dealii::TrilinosWrappers::SparseMatrix sparse_matrix;
    sparse_matrix.reinit(dsp);

    VectorType rhs;
    reinit_vector(rhs, dof_handler);
    rhs = 0.;

    dealii::hp::QCollection<dim - 1> face_quadrature;
    face_quadrature.push_back(dealii::QGauss<dim - 1>(fe_degree + 1));

    const unsigned int invalid = dealii::numbers::invalid_unsigned_int;

    // fill matrix
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned())
          continue;

        const auto   cell_location    = mesh_classifier.location_to_level_set(cell);
        const Number cell_side_length = cell->minimum_vertex_distance();

        if (cell_location != dealii::NonMatching::LocationToLevelSet::intersected)
          continue;

        // definition of lambda-function for ghost-penalty evaluation
        const auto eval_ghost_penalty = [&](dealii::FEValuesExtractors::Scalar &u_extractor,
                                            dealii::NonMatching::LocationToLevelSet
                                              inactive_location) {
          dealii::UpdateFlags update_flags =
            dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors;
          if (is_dg)
            update_flags = dealii::update_values | update_flags;
          if (fe_degree == 2)
            update_flags = update_flags | dealii::update_hessians;

          dealii::FEInterfaceValues<dim> fe_interface_values(dof_handler.get_fe_collection(),
                                                             face_quadrature,
                                                             update_flags);

          for (const unsigned int f : cell->face_indices())
            {
              // check if current face is a ghost-penalty face
              if (not CutUtil::face_has_ghost_penalty(mesh_classifier, cell, f, inactive_location))
                continue;

              fe_interface_values.reinit(
                cell, f, invalid, cell->neighbor(f), cell->neighbor_of_neighbor(f), invalid);

              const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();

              // Determine prefactor, for ghost-penalty face term evaluation.
              // Prefactor 0.5 is used for faces, which are visited twice.
              Number prefactor = 1.;
              if (CutUtil::is_new_intersected_face(
                    mesh_classifier, mesh_classifier_old, cell, f, inactive_location))
                prefactor = 0.5;

              // initialize local ghost-penalty constraint matrix and local rhs
              dealii::FullMatrix<Number> local_ghost_penalty_matrix(n_interface_dofs,
                                                                    n_interface_dofs);
              dealii::Vector<Number>     local_rhs(n_interface_dofs);

              // compute entries of local ghost-penalty constraint matrix
              for (const auto i : fe_interface_values.dof_indices())
                for (const auto j : fe_interface_values.dof_indices())
                  for (unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
                    {
                      const dealii::Tensor<1, dim> normal = fe_interface_values.normal(q);

                      // contributions from 1. normal derivative jump
                      local_ghost_penalty_matrix(i, j) +=
                        prefactor * normal *
                        fe_interface_values[u_extractor].jump_in_gradients(i, q) * normal *
                        fe_interface_values[u_extractor].jump_in_gradients(j, q) *
                        cell_side_length * gamma_degree_1 * fe_interface_values.JxW(q);

                      if (is_dg)
                        {
                          // contributions from 0. normal derivative jump
                          local_ghost_penalty_matrix(i, j) +=
                            prefactor * fe_interface_values[u_extractor].jump_in_values(i, q) *
                            fe_interface_values[u_extractor].jump_in_values(j, q) /
                            cell_side_length * gamma_degree_0 * fe_interface_values.JxW(q);
                        }

                      if (fe_degree == 2)
                        {
                          // contributions from 2. normal derivative jump
                          local_ghost_penalty_matrix(i, j) +=
                            prefactor *
                            (normal * fe_interface_values[u_extractor].jump_in_hessians(i, q) *
                             normal /*double contraction*/) *
                            (normal * fe_interface_values[u_extractor].jump_in_hessians(j, q) *
                             normal /*double contraction*/) *
                            dealii::Utilities::fixed_power<3>(cell_side_length) * gamma_degree_2 *
                            fe_interface_values.JxW(q);
                        }
                    }

              // distribute local ghost-penalty constraint matrix and local_rhs to
              // global system
              const std::vector<dealii::types::global_dof_index> local_interface_dof_indices =
                fe_interface_values.get_interface_dof_indices();

              constraints_gp.distribute_local_to_global(local_ghost_penalty_matrix,
                                                        local_rhs,
                                                        local_interface_dof_indices,
                                                        sparse_matrix,
                                                        rhs);
            }
        };

        const auto cell_location_old = mesh_classifier_old.location_to_level_set(cell);

        if (cell_location_old == dealii::NonMatching::LocationToLevelSet::inside)
          for (unsigned int i = 0; i < n_components_per_phase; ++i)
            {
              dealii::FEValuesExtractors::Scalar u(i);
              eval_ghost_penalty(u, cell_location_old);
            }
        else if ((cell_location_old == dealii::NonMatching::LocationToLevelSet::outside) &&
                 (is_two_phase == true))
          for (unsigned int i = n_components_per_phase; i < 2 * n_components_per_phase; ++i)
            {
              dealii::FEValuesExtractors::Scalar u(i);
              eval_ghost_penalty(u, cell_location_old);
            }
      }

    sparse_matrix.compress(dealii::VectorOperation::add);
    rhs.compress(dealii::VectorOperation::add);

    // solve system
    dealii::ReductionControl solver_control(1000, 1e-10, 1e-10);

    dealii::SolverCG<VectorType> solver(solver_control);
    solver.solve(sparse_matrix, new_solution, rhs, dealii::PreconditionIdentity());

    if (verbosity >= 2)
      {
        std::ostringstream str;
        str << "Ghost-penaly extrapolation system solved in " << solver_control.last_step()
            << " iterations.";
        Journal::print_line(pcout, str.str(), "cut_solution_transfer");
      }

    // apply constraints
    constraints_gp.distribute(new_solution);
  }



  template class SolutionTransferOperator<1, double>;
  template class SolutionTransferOperator<2, double>;
  template class SolutionTransferOperator<3, double>;
} // namespace MeltPoolDG::CutUtil
