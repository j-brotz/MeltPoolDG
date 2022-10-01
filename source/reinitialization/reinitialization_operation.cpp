#include <meltpooldg/linear_algebra/preconditioner_trilinos_factory.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::Reinitialization
{
  template <int dim>
  ReinitializationOperation<dim>::ReinitializationOperation(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &                     data_in,
    const TimeIterator<double> &                   time_iterator,
    const unsigned int                             reinit_dof_idx_in,
    const unsigned int                             reinit_quad_idx_in,
    const unsigned int                             ls_dof_idx_in,
    const unsigned int                             normal_dof_idx_in)
    : reinit_data(data_in.reinit)
    , scratch_data(scratch_data_in)
    , time_iterator(time_iterator)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
  {
    scratch_data->initialize_dof_vector(solution_level_set, ls_dof_idx_in);

    AssertThrow(reinit_data.linear_solver.solver_type == LinearSolverType::CG ||
                  reinit_data.linear_solver.do_matrix_free == false,
                ExcMessage(
                  "The matrix-free reinitialization_operation only supports the CG solver type."));
    /*
     *    initialize normal_vector_field
     */
    AssertThrow(data_in.normal_vec.linear_solver.do_matrix_free ==
                  data_in.reinit.linear_solver.do_matrix_free,
                ExcMessage("For the reinitialization problem both the "
                           " normal vector and the reinitialization operation have to be "
                           " computed either matrix-based or matrix-free."));

    if (data_in.normal_vec.implementation == "meltpooldg")
      {
        normal_vector_operation = std::make_shared<NormalVector::NormalVectorOperation<dim>>();

        normal_vector_operation->initialize(
          scratch_data_in, data_in, normal_dof_idx, reinit_quad_idx, ls_dof_idx);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (data_in.normal_vec.implementation == "adaflo")
      {
        AssertThrow(data_in.normal_vec.linear_solver.do_matrix_free, ExcNotImplemented());

        normal_vector_operation =
          std::make_shared<NormalVector::NormalVectorOperationAdaflo<dim>>(*scratch_data_in,
                                                                           ls_dof_idx_in,
                                                                           normal_dof_idx,
                                                                           reinit_quad_idx,
                                                                           solution_level_set,
                                                                           data_in);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *   create reinitialization operator. This class supports matrix-based
     *   and matrix-free computation.
     */
    create_operator();
    scratch_data->initialize_dof_vector(delta_psi_vec, reinit_dof_idx);
    scratch_data->initialize_dof_vector(delta_psi_vec_old, reinit_dof_idx);
    scratch_data->initialize_dof_vector(rhs, reinit_dof_idx);
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::reinit()
  {
    scratch_data->initialize_dof_vector(solution_level_set, ls_dof_idx);
    scratch_data->initialize_dof_vector(delta_psi_vec, reinit_dof_idx);
    scratch_data->initialize_dof_vector(delta_psi_vec_old, reinit_dof_idx);
    scratch_data->initialize_dof_vector(rhs, reinit_dof_idx);

    update_operator();
    normal_vector_operation->reinit();

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
    scratch_data->initialize_dof_vector(solution_level_set, ls_dof_idx);
    solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
    /*
     *    update the normal vector field corresponding to the given solution of the
     *    level set; the normal vector field is called by reference within the
     *    operator class
     */
    normal_vector_operation->solve(solution_level_set);
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
    if (reinit_data.predictor == PredictorType::none)
      {
        delta_psi_vec_old.copy_locally_owned_data_from(delta_psi_vec);
        delta_psi_vec = 0.0;
      }
    else if (reinit_data.predictor == PredictorType::linear_extrapolation)
      {
        VectorType delta_psi_extrapolated;
        scratch_data->initialize_dof_vector(delta_psi_extrapolated, reinit_dof_idx);

        // TODO: use old time increment
        UtilityFunctions::compute_linear_predictor(delta_psi_vec,
                                                   delta_psi_vec_old,
                                                   delta_psi_extrapolated,
                                                   time_iterator.get_current_time_increment(),
                                                   time_iterator.get_old_time_increment());

        delta_psi_vec_old.copy_locally_owned_data_from(delta_psi_vec);
        delta_psi_vec.copy_locally_owned_data_from(delta_psi_extrapolated);

        // apply hanging node constraints to predictor
        scratch_data->get_constraint(reinit_dof_idx).distribute(delta_psi_vec);
      }
    reinit_operator->reset_time_increment(time_iterator.get_current_time_increment());
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::solve()
  {
    get_normal_vector().update_ghost_values();
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
                                                   delta_psi_vec,
                                                   rhs,
                                                   reinit_data.linear_solver,
                                                   *diag_preconditioner_matrixfree);
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
                                                   delta_psi_vec,
                                                   rhs,
                                                   reinit_data.linear_solver,
                                                   *trilinos_preconditioner_matrixfree);
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
                                               delta_psi_vec,
                                               rhs,
                                               reinit_data.linear_solver,
                                               *preconditioner);

        Journal::print_formatted_norm(scratch_data->get_pcout(0),
                                      reinit_operator->get_system_matrix().frobenius_norm(),
                                      "matrix",
                                      "reinitialization",
                                      15 /*precision*/,
                                      "F");
      }
    scratch_data->get_constraint(reinit_dof_idx).distribute(delta_psi_vec);
    solution_level_set.zero_out_ghost_values();
    get_normal_vector().zero_out_ghost_values();


    // copy the delta_psi to the DoFHandler of the level set
    VectorType delta_level_set;
    scratch_data->initialize_dof_vector(delta_level_set, ls_dof_idx);
    delta_level_set.copy_locally_owned_data_from(delta_psi_vec);

    solution_level_set += delta_level_set;

    max_change_level_set = delta_psi_vec.linfty_norm();

    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  MeltPoolDG::VectorTools::compute_L2_norm<dim>(
                                    rhs, *scratch_data, reinit_dof_idx, reinit_quad_idx),
                                  "RHS",
                                  "reinitialization",
                                  15 /*precision*/
    );
    Journal::print_formatted_norm(scratch_data->get_pcout(0),
                                  VectorTools::compute_L2_norm<dim>(
                                    delta_psi_vec, *scratch_data, reinit_dof_idx, reinit_quad_idx),
                                  "delta phi",
                                  "reinitialization",
                                  15 /*precision*/
    );

    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  max_change_level_set,
                                  "delta phi",
                                  "reinitialization",
                                  15 /*precision*/,
                                  "∞ ",
                                  2);
    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  VectorTools::compute_L2_norm<dim>(solution_level_set,
                                                                    *scratch_data,
                                                                    reinit_dof_idx,
                                                                    reinit_quad_idx),
                                  "phi",
                                  "reinitialization",
                                  15 /*precision*/
    );

    Journal::print_line(scratch_data->get_pcout(1),
                        "     * CG: i = " + std::to_string(iter),
                        "reinitialization");
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
    vectors.push_back(&solution_level_set);
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data->get_dof_handler(reinit_dof_idx), get_level_set(), "psi");

    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data->get_dof_handler(reinit_dof_idx),
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
          *scratch_data,
          reinit_data,
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

    /*
     *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */

    if (!reinit_data.linear_solver.do_matrix_free)
      reinit_operator->initialize_matrix_based(*scratch_data);

    if (reinit_data.linear_solver.do_matrix_free)
      {
        preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          *scratch_data,
          reinit_dof_idx,
          reinit_data.linear_solver.preconditioner_type,
          *reinit_operator);
        /*
         * setup sparsity pattern of system matrix only if the latter is
         * needed for computing the preconditioner
         */
        preconditioner_matrixfree->reinit();
      }
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::update_operator()
  {
    if (!reinit_data.linear_solver.do_matrix_free)
      reinit_operator->initialize_matrix_based(*scratch_data);
  }

  template class ReinitializationOperation<1>;
  template class ReinitializationOperation<2>;
  template class ReinitializationOperation<3>;
} // namespace MeltPoolDG::Reinitialization
