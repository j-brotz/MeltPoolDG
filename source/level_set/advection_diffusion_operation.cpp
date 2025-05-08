#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/level_set/advection_diffusion_operation.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <variant>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  AdvectionDiffusionOperation<dim, number>::AdvectionDiffusionOperation(
    const ScratchData<dim, dim, number>                                &scratch_data_in,
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &dirichlet_bc_in,
    const AdvectionDiffusionData<number>                               &advec_diff_data_in,
    const TimeIntegration::TimeIterator<number>                        &time_iterator,
    const VectorType                                                   &advection_velocity,
    const unsigned int                                                  advec_diff_dof_idx_in,
    const unsigned int advec_diff_hanging_nodes_dof_idx_in,
    const unsigned int advec_diff_quad_idx_in,
    const unsigned int velocity_dof_idx_in)
    : scratch_data(scratch_data_in)
    , dirichlet_bc(dirichlet_bc_in)
    , time_iterator(time_iterator)
    , advection_velocity(advection_velocity)
    , advec_diff_dof_idx(advec_diff_dof_idx_in)
    , advec_diff_quad_idx(advec_diff_quad_idx_in)
    , advec_diff_hanging_nodes_dof_idx(advec_diff_hanging_nodes_dof_idx_in)
    , velocity_dof_idx(velocity_dof_idx_in)
    , solution_history(std::max(advec_diff_data_in.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
  {
    this->advec_diff_data = advec_diff_data_in;
    preconditioner =
      make_preconditioner<dim, number, AdvectionDiffusionOperator<dim, number>, VectorType>(
        this->advec_diff_data.linear_solver.preconditioner_type, advec_diff_operator.get());
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    if (solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().zero_out_ghost_values();
    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(advec_diff_dof_idx),
                                     initial_field_function,
                                     solution_history.get_current_solution());

    scratch_data.get_constraint(advec_diff_dof_idx)
      .distribute(solution_history.get_current_solution());

    // TODO: set to zero
    solution_history.set_recent_old_solution(solution_history.get_current_solution());

    // update ghost values
    solution_history.update_ghost_values();
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, advec_diff_dof_idx); });

    scratch_data.initialize_dof_vector(solution_advected_field_extrapolated, advec_diff_dof_idx);

    scratch_data.initialize_dof_vector(user_rhs, advec_diff_dof_idx);
    scratch_data.initialize_dof_vector(rhs, advec_diff_dof_idx);

    // TODO: setup sparsity pattern of system matrix only if the latter is needed for computing the
    // preconditioner
    if (preconditioner.is_initialized())
      preconditioner.reinit(scratch_data, advec_diff_dof_idx);

    if (advec_diff_operator)
      advec_diff_operator->reinit();
  }


  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::init_time_advance()
  {
    if (!advec_diff_operator)
      create_operator(advection_velocity);

    // pass time increment to operator TODO: pass time iterator directly to operator
    advec_diff_operator->reset_time_increment(time_iterator.get_current_time_increment());

    if (this->advec_diff_data.enable_time_dependent_bc)
      {
        MeltPoolDG::Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim, number>(
          const_cast<ScratchData<dim, dim, number> &>(scratch_data),
          dirichlet_bc,
          advec_diff_dof_idx,
          advec_diff_hanging_nodes_dof_idx);
      }

    // create inflow/outflow constraints if requested
    this->create_inflow_outflow_constraints();

    // compute RHS
    // TODO: also include it for matrix-based to this place (?)
    if (this->advec_diff_data.linear_solver.do_matrix_free &&
        this->advec_diff_data.predictor.type == PredictorType::least_squares_projection)
      {
        const bool velocity_update_ghosts = !advection_velocity.has_ghost_elements();
        if (velocity_update_ghosts)
          advection_velocity.update_ghost_values();

        const bool solution_update_ghosts =
          !solution_history.get_current_solution().has_ghost_elements();
        if (solution_update_ghosts)
          solution_history.get_current_solution().update_ghost_values();

        rhs = user_rhs;

        // apply dirichlet boundary values
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree<dim, number>(
          *advec_diff_operator,
          rhs,
          solution_history.get_current_solution(), //= old_solution for current time step,
          scratch_data,
          advec_diff_dof_idx,
          advec_diff_hanging_nodes_dof_idx,
          false /*don't zero out rhs*/,
          inflow_constraints_indices_and_values);

        if (solution_update_ghosts)
          solution_history.get_current_solution().zero_out_ghost_values();

        if (velocity_update_ghosts)
          advection_velocity.zero_out_ghost_values();
      }

    if (!predictor)
      predictor = std::make_unique<Predictor<VectorType, number>>(this->advec_diff_data.predictor,
                                                                  solution_history,
                                                                  &time_iterator);

    predictor->vmult(*advec_diff_operator, solution_advected_field_extrapolated, rhs);

    // apply hanging node constraints to predictor
    scratch_data.get_constraint(advec_diff_dof_idx)
      .distribute(solution_history.get_current_solution());

    this->ready_for_time_advance = true;
  }



  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::solve(const bool do_finish_time_step)
  {
    const ScopedName         scope_n("advection_diffusion::solve");
    const TimerOutput::Scope scope_t(scratch_data.get_timer(), scope_n);

    if (not this->ready_for_time_advance)
      init_time_advance();

    const bool velocity_update_ghosts = !advection_velocity.has_ghost_elements();
    if (velocity_update_ghosts)
      advection_velocity.update_ghost_values();

    const bool solution_update_ghosts =
      !solution_history.get_recent_old_solution().has_ghost_elements();
    if (solution_update_ghosts)
      solution_history.get_recent_old_solution().update_ghost_values();

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(2),
      [&]() -> number {
        return VectorTools::compute_norm<dim, number>(advection_velocity,
                                                      scratch_data,
                                                      velocity_dof_idx,
                                                      advec_diff_quad_idx);
      },
      "velocity",
      "advection_diffusion");

    int iter = 0;

    advec_diff_operator->pre();

    if (this->advec_diff_data.linear_solver.do_matrix_free)
      {
        rhs = user_rhs;

        // apply dirichlet boundary values
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree<dim, number>(
          *advec_diff_operator,
          rhs,
          solution_history.get_recent_old_solution(),
          scratch_data,
          advec_diff_dof_idx,
          advec_diff_hanging_nodes_dof_idx,
          false /*don't zero out rhs*/,
          inflow_constraints_indices_and_values);

        advec_diff_operator->enable_pre_post();
        preconditioner.update();

        solution_history.get_current_solution().zero_out_ghost_values();
        iter = LinearSolver::solve<VectorType>(*advec_diff_operator,
                                               solution_history.get_current_solution(),
                                               rhs,
                                               this->advec_diff_data.linear_solver,
                                               preconditioner,
                                               "advection_diffusion_operation");

        advec_diff_operator->disable_pre_post();
      }
    else
      {
        rhs = 0.0;
        advec_diff_operator->compute_system_matrix_and_rhs(
          solution_history.get_recent_old_solution(), rhs);

        rhs += user_rhs;
        scratch_data.get_constraint(advec_diff_dof_idx)
          .distribute(rhs); //@todo: this could be avoided by introducing a zero_out inside
                            // compute_system_matrix_and_rhs

        preconditioner.update(&advec_diff_operator->get_system_matrix());
        iter = LinearSolver::solve<VectorType>(advec_diff_operator->get_system_matrix(),
                                               solution_history.get_current_solution(),
                                               rhs,
                                               this->advec_diff_data.linear_solver,
                                               preconditioner,
                                               "advection_diffusion_operation");
      }
    // inflow/outflow
    if (!inflow_outflow_bc.empty())
      {
        for (unsigned int i = 0; i < inflow_constraints_indices_and_values.first.size(); ++i)
          solution_history.get_current_solution().local_element(
            inflow_constraints_indices_and_values.first[i]) =
            inflow_constraints_indices_and_values.second[i];
      }

    scratch_data.get_constraint(advec_diff_dof_idx)
      .distribute(solution_history.get_current_solution());


    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(3),
      [&]() -> number { return advec_diff_operator->get_system_matrix().frobenius_norm(); },
      "matrix",
      "advection_diffusion",
      6 /*precision*/,
      "F");

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(3),
      [&]() -> number { return rhs.l2_norm(); },
      "rhs",
      "advection_diffusion",
      6 /*precision*/,
      "l2");
    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(3),
      [&]() -> number { return solution_history.get_current_solution().l2_norm(); },
      "src",
      "advection_diffusion",
      6 /*precision*/,
      "l2");

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(1),
      [&]() -> number {
        return MeltPoolDG::VectorTools::compute_norm<dim, number>(
          solution_history.get_current_solution(),
          scratch_data,
          advec_diff_dof_idx,
          advec_diff_quad_idx);
      },
      "advected field",
      "advection_diffusion",
      10 /*precision*/
    );

    Journal::print_line(scratch_data.get_pcout(3),
                        "     * GMRES: i = " + std::to_string(iter),
                        "advection_diffusion");

    if (solution_update_ghosts)
      solution_history.get_recent_old_solution().zero_out_ghost_values();

    if (velocity_update_ghosts)
      advection_velocity.zero_out_ghost_values();

    if (do_finish_time_step)
      {
        this->finish_time_advance();
      }

    IterationMonitor<number>::add_linear_iterations(scope_n, iter);

    // update ghost values of solution
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::create_inflow_outflow_constraints()
  {
    if (!inflow_outflow_bc.empty())
      {
        AssertThrow(
          this->advec_diff_data.linear_solver.do_matrix_free == true,
          ExcMessage(
            "Inflow/outflow boundary conditions are only available for the matrix-free computation."));

        // 1) collect relevant indices and values subject to inflow
        inflow_constraints_indices_and_values.first.clear();
        inflow_constraints_indices_and_values.second.clear();

        const auto &partitioner =
          scratch_data.get_matrix_free().get_vector_partitioner(advec_diff_dof_idx);

        FEFaceValues<dim> vel_eval(
          scratch_data.get_mapping(),
          scratch_data.get_fe(velocity_dof_idx),
          Quadrature<dim - 1>(
            scratch_data.get_fe(advec_diff_dof_idx).get_unit_face_support_points()),
          update_quadrature_points | update_values | update_normal_vectors);
        const FEValuesExtractors::Vector velocities(0);

        advection_velocity.update_ghost_values();

        VectorType advection_velocity_compatible;
        advection_velocity_compatible.reinit(
          scratch_data.get_dof_handler(velocity_dof_idx).locally_owned_dofs(),
          DoFTools::extract_locally_relevant_dofs(scratch_data.get_dof_handler(velocity_dof_idx)),
          scratch_data.get_mpi_comm());
        advection_velocity_compatible.copy_locally_owned_data_from(advection_velocity);
        advection_velocity_compatible.update_ghost_values();

        for (const auto &cell : scratch_data.get_triangulation()
                                  .active_cell_iterators()) //|
                                                            // IteratorFilters::LocallyOwnedCell())
          {
            if (cell->is_locally_owned() || cell->is_ghost())
              {
                TriaIterator<DoFCellAccessor<dim, dim, false>> vel_dof_cell(
                  &scratch_data.get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &scratch_data.get_dof_handler(velocity_dof_idx));

                TriaIterator<DoFCellAccessor<dim, dim, false>> advec_diff_dof_cell(
                  &scratch_data.get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &scratch_data.get_dof_handler(advec_diff_dof_idx));

                unsigned int face_index = 0;
                for (const auto &face : cell->face_iterators())
                  {
                    if (face->at_boundary() && inflow_outflow_bc.contains(face->boundary_id()))
                      {
                        std::vector<Tensor<1, dim>> local_velocity(vel_eval.n_quadrature_points,
                                                                   Tensor<1, dim>());

                        std::vector<types::global_dof_index> face_dof_indices(
                          vel_eval.n_quadrature_points);

                        vel_eval.reinit(vel_dof_cell, face_index);
                        vel_eval[velocities].get_function_values(advection_velocity_compatible,
                                                                 local_velocity);

                        advec_diff_dof_cell->face(face_index)->get_dof_indices(face_dof_indices);

                        for (const auto q : vel_eval.quadrature_point_indices())
                          {
                            // inflow
                            if (scalar_product(vel_eval.normal_vector(q), local_velocity[q]) <= 0)
                              {
                                if (partitioner->in_local_range(face_dof_indices[q]) or
                                    partitioner->is_ghost_entry(face_dof_indices[q]))
                                  {
                                    const auto local_index =
                                      partitioner->global_to_local(face_dof_indices[q]);

                                    inflow_constraints_indices_and_values.first.emplace_back(
                                      local_index);
                                    inflow_constraints_indices_and_values.second.emplace_back(
                                      inflow_outflow_bc[face->boundary_id()]->value(
                                        vel_eval.quadrature_point(q)));
                                  }
                              }
                          }
                      }
                    ++face_index;
                  }
              }
          }

        // delete duplicated entries
        UtilityFunctions::remove_duplicates(inflow_constraints_indices_and_values.first,
                                            inflow_constraints_indices_and_values.second);

        // set indices in operator
        dynamic_cast<AdvectionDiffusionOperator<dim, number> *>(advec_diff_operator.get())
          ->set_inflow_outflow_bc(inflow_constraints_indices_and_values.first);
      }
  }



  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperation<dim, number>::get_advected_field() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperation<dim, number>::get_advected_field()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperation<dim, number>::get_advected_field_old() const
  {
    return solution_history.get_recent_old_solution();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperation<dim, number>::get_advected_field_old()
  {
    return solution_history.get_recent_old_solution();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperation<dim, number>::get_user_rhs()
  {
    return user_rhs;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperation<dim, number>::get_user_rhs() const
  {
    return user_rhs;
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(advec_diff_dof_idx),
                             solution_history.get_current_solution(),
                             "advected_field");
    data_out.add_data_vector(scratch_data.get_dof_handler(advec_diff_dof_idx),
                             user_rhs,
                             "advec_diff_user_rhs");
  }


  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::set_inflow_outflow_bc(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> inflow_outflow_bc_)
  {
    inflow_outflow_bc = inflow_outflow_bc_;
  }



  template <int dim, typename number>
  void
  AdvectionDiffusionOperation<dim, number>::create_operator(const VectorType &advection_velocity)
  {
    advec_diff_operator =
      std::make_unique<AdvectionDiffusionOperator<dim, number>>(scratch_data,
                                                                advection_velocity,
                                                                this->advec_diff_data,
                                                                advec_diff_dof_idx,
                                                                advec_diff_quad_idx,
                                                                velocity_dof_idx);
    /*
     *  In case of a matrix-based simulation, set up the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */
    advec_diff_operator->reinit();
    preconditioner =
      make_preconditioner<dim, number, AdvectionDiffusionOperator<dim, number>, VectorType>(
        this->advec_diff_data.linear_solver.preconditioner_type, advec_diff_operator.get());
    preconditioner.reinit(scratch_data, advec_diff_dof_idx);
  }

  template class AdvectionDiffusionOperation<1, double>;
  template class AdvectionDiffusionOperation<2, double>;
  template class AdvectionDiffusionOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
