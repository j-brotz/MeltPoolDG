#include <meltpooldg/linear_algebra/preconditioner_trilinos_factory.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  ReinitializationOperation<dim>::ReinitializationOperation(
    const ScratchData<dim>             &scratch_data_in,
    const ReinitializationData<double> &reinit_data,
    const NormalVectorData<double>     &normal_vec_data,
    const int                           ls_n_subdivisions_in,
    const TimeIterator<double>         &time_iterator,
    const unsigned int                  reinit_dof_idx_in,
    const unsigned int                  reinit_quad_idx_in,
    const unsigned int                  ls_dof_idx_in,
    const unsigned int                  normal_dof_idx_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data)
    , ls_n_subdivisions(ls_n_subdivisions_in)
    , time_iterator(time_iterator)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
    , solution_history(reinit_data.predictor.n_old_solution_vectors)
  {
    if (normal_vec_data.implementation == "meltpooldg")
      {
        normal_vector_operation = std::make_shared<NormalVectorOperation<dim>>(scratch_data_in,
                                                                               normal_vec_data,
                                                                               solution_level_set,
                                                                               normal_dof_idx,
                                                                               reinit_quad_idx,
                                                                               ls_dof_idx);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (normal_vec_data.implementation == "adaflo")
      {
        normal_vector_operation = std::make_shared<NormalVectorOperationAdaflo<dim>>(
          scratch_data_in,
          ls_dof_idx_in,
          normal_dof_idx,
          reinit_quad_idx,
          solution_level_set,
          normal_vec_data,
          reinit_data.interface_thickness_parameter.value / ls_n_subdivisions);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());

    /*
     *   create reinitialization operator. This class supports matrix-based
     *   and matrix-free computation.
     */
    create_operator();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, reinit_dof_idx); });

    scratch_data.initialize_dof_vector(solution_level_set, ls_dof_idx);
    scratch_data.initialize_dof_vector(delta_psi_extrapolated, reinit_dof_idx);

    scratch_data.initialize_dof_vector(rhs, reinit_dof_idx);

    update_operator();
    normal_vector_operation->reinit();
    reinit_operator->reinit();

    if (reinit_data.linear_solver.do_matrix_free)
      {
        /*
         * setup sparsity pattern of system matrix only if the latter is
         * needed for computing the preconditioner
         */
        preconditioner_matrixfree->reinit();
      }
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::set_initial_condition(const VectorType &solution_level_set_in)
  {
    /*
     *    copy the given solution into the member variable
     */
    solution_level_set.zero_out_ghost_values();
    solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
    /*
     *    update the normal vector field corresponding to the given solution of the
     *    level set; the normal vector field is called by reference within the
     *    operator class
     */
    normal_vector_operation->solve();
    /*
     * precompute preconditioner system matrix
     */
    if (reinit_data.linear_solver.do_matrix_free)
      update_preconditioner_matrixfree = true;
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::update_dof_idx(const unsigned int &reinit_dof_idx_in)
  {
    reinit_dof_idx = reinit_dof_idx_in;
    create_operator();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::init_time_advance()
  {
    reinit_operator->reset_time_increment(time_iterator.get_current_time_increment());

    // compute RHS
    // TODO: also include it for matrix-based to this place (?)
    if (reinit_data.linear_solver.do_matrix_free &&
        reinit_data.predictor.type == PredictorType::least_squares_projection)
      {
        reinit_operator->create_rhs(rhs, solution_level_set);
      }

    if (!predictor)
      predictor = std::make_unique<Predictor<VectorType, double>>(reinit_data.predictor,
                                                                  solution_history,
                                                                  &time_iterator);

    predictor->vmult(*reinit_operator, delta_psi_extrapolated, rhs);

    // apply hanging node constraints to predictor
    scratch_data.get_constraint(reinit_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.get_current_solution().zero_out_ghost_values();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::solve()
  {
    ScopedName         sc("reinitialization::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    const bool normal_update_ghosts = !get_normal_vector().has_ghost_elements();
    if (normal_update_ghosts)
      get_normal_vector().update_ghost_values();

    const bool ls_update_ghosts = !solution_level_set.has_ghost_elements();
    if (ls_update_ghosts)
      solution_level_set.update_ghost_values();

    init_time_advance();

    int iter = 0;

    if (reinit_data.linear_solver.do_matrix_free)
      {
        AssertThrow(preconditioner_matrixfree, ExcNotImplemented());

        reinit_operator->create_rhs(rhs, solution_level_set);

        if (reinit_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          {
            if (update_preconditioner_matrixfree)
              {
                diag_preconditioner_matrixfree =
                  preconditioner_matrixfree->compute_diagonal_preconditioner();
                update_preconditioner_matrixfree = false;
              }

            iter = LinearSolver::solve<VectorType>(*reinit_operator,
                                                   solution_history.get_current_solution(),
                                                   rhs,
                                                   reinit_data.linear_solver,
                                                   *diag_preconditioner_matrixfree,
                                                   "reinitialization_operation");
          }
        else
          {
            if (update_preconditioner_matrixfree)
              {
                trilinos_preconditioner_matrixfree =
                  preconditioner_matrixfree->compute_trilinos_preconditioner();
                update_preconditioner_matrixfree = false;
              }

            iter = LinearSolver::solve<VectorType>(*reinit_operator,
                                                   solution_history.get_current_solution(),
                                                   rhs,
                                                   reinit_data.linear_solver,
                                                   *trilinos_preconditioner_matrixfree,
                                                   "reinitialization_operation");
          }
      }
    else
      {
        reinit_operator->get_system_matrix() = 0;
        reinit_operator->assemble_matrixbased(solution_level_set,
                                              reinit_operator->get_system_matrix(),
                                              rhs);

        auto preconditioner = Preconditioner::get_preconditioner_trilinos(
          reinit_operator->get_system_matrix(), reinit_data.linear_solver.preconditioner_type);
        iter = LinearSolver::solve<VectorType>(reinit_operator->get_system_matrix(),
                                               solution_history.get_current_solution(),
                                               rhs,
                                               reinit_data.linear_solver,
                                               *preconditioner,
                                               "reinitialization_operation");

        Journal::print_formatted_norm(
          scratch_data.get_pcout(0),
          [&]() -> double { return reinit_operator->get_system_matrix().frobenius_norm(); },
          "matrix",
          "reinitialization",
          15 /*precision*/,
          "F");
      }
    scratch_data.get_constraint(reinit_dof_idx).distribute(solution_history.get_current_solution());

    if (normal_update_ghosts)
      get_normal_vector().zero_out_ghost_values();

    // copy the delta_psi to the DoFHandler of the level set
    VectorType delta_level_set;
    scratch_data.initialize_dof_vector(delta_level_set, ls_dof_idx);
    delta_level_set.copy_locally_owned_data_from(solution_history.get_current_solution());

    solution_level_set.zero_out_ghost_values();
    solution_level_set += delta_level_set;

    // update ghost values of solution
    solution_level_set.update_ghost_values();
    max_change_level_set = solution_history.get_current_solution().linfty_norm();

    Journal::print_formatted_norm(
      scratch_data.get_pcout(1),
      [&]() -> double {
        return MeltPoolDG::VectorTools::compute_norm<dim>(rhs,
                                                          scratch_data,
                                                          reinit_dof_idx,
                                                          reinit_quad_idx);
      },
      "RHS",
      "reinitialization",
      15 /*precision*/
    );
    Journal::print_formatted_norm(
      scratch_data.get_pcout(0),
      [&]() -> double {
        return VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                              scratch_data,
                                              reinit_dof_idx,
                                              reinit_quad_idx);
      },
      "delta phi",
      "reinitialization",
      15 /*precision*/
    );

    Journal::print_formatted_norm(scratch_data.get_pcout(1),
                                  max_change_level_set,
                                  "delta phi",
                                  "reinitialization",
                                  15 /*precision*/,
                                  "∞ ",
                                  2);
    Journal::print_formatted_norm(
      scratch_data.get_pcout(1),
      [&]() -> double {
        return VectorTools::compute_norm<dim>(solution_level_set,
                                              scratch_data,
                                              reinit_dof_idx,
                                              reinit_quad_idx);
      },
      "phi",
      "reinitialization",
      15 /*precision*/
    );

    Journal::print_line(scratch_data.get_pcout(1),
                        "     * CG: i = " + std::to_string(iter),
                        "reinitialization");

    IterationMonitor::add_linear_iterations(sc, iter);
  }

  template <int dim>
  double
  ReinitializationOperation<dim>::get_max_change_level_set() const
  {
    return max_change_level_set;
  }

  template <int dim>
  const typename ReinitializationOperation<dim>::BlockVectorType &
  ReinitializationOperation<dim>::get_normal_vector() const
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim>
  const VectorType &
  ReinitializationOperation<dim>::get_level_set() const
  {
    return solution_level_set;
  }

  template <int dim>
  VectorType &
  ReinitializationOperation<dim>::get_level_set()
  {
    return solution_level_set;
  }

  template <int dim>
  typename ReinitializationOperation<dim>::BlockVectorType &
  ReinitializationOperation<dim>::get_normal_vector()
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    normal_vector_operation->attach_vectors(vectors);

    vectors.push_back(&solution_level_set);

    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx), get_level_set(), "psi");

    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::create_operator()
  {
    if (reinit_data.modeltype == "olsson2007")
      {
        reinit_operator = std::make_unique<OlssonOperator<dim, double>>(
          scratch_data,
          reinit_data,
          ls_n_subdivisions,
          normal_vector_operation->get_solution_normal_vector(),
          reinit_dof_idx,
          reinit_quad_idx,
          ls_dof_idx,
          normal_dof_idx);
      }
    /*
     * add your desired operators here
     *
     * else if (reinit_data.reinitmodel == "my_model")
     *    ....
     */
    else
      AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."));

    if (reinit_data.linear_solver.do_matrix_free)
      preconditioner_matrixfree = std::make_shared<
        Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
        scratch_data,
        reinit_dof_idx,
        reinit_data.linear_solver.preconditioner_type,
        *reinit_operator);
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::update_operator()
  {
    if (!reinit_data.linear_solver.do_matrix_free)
      reinit_operator->initialize_matrix_based(scratch_data);
  }

  template class ReinitializationOperation<1>;
  template class ReinitializationOperation<2>;
  template class ReinitializationOperation<3>;
} // namespace MeltPoolDG::LevelSet
