#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_factory.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::initialize(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const AdvectionDiffusionData<double> &         advec_diff_data_in,
    const unsigned int                             advec_diff_dof_idx_in,
    const unsigned int                             advec_diff_hanging_nodes_dof_idx_in,
    const unsigned int                             advec_diff_quad_idx_in,
    const unsigned int                             velocity_dof_idx_in)
  {
    scratch_data                     = scratch_data_in;
    advec_diff_dof_idx               = advec_diff_dof_idx_in;
    advec_diff_quad_idx              = advec_diff_quad_idx_in;
    advec_diff_hanging_nodes_dof_idx = advec_diff_hanging_nodes_dof_idx_in;
    velocity_dof_idx                 = velocity_dof_idx_in;
    /*
     *  set the advection diffusion data
     */
    this->advec_diff_data = advec_diff_data_in;

    if (this->advec_diff_data.linear_solver.do_matrix_free)
      {
        preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          *scratch_data,
          advec_diff_dof_idx,
          this->advec_diff_data.linear_solver.preconditioner_type,
          *advec_diff_operator);

        preconditioner_matrixfree->reinit();
      }
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::set_initial_condition(
    const Function<dim> &initial_field_function,
    const VectorType &   initial_velocity)
  {
    (void)initial_velocity; // @todo: delete
    scratch_data->initialize_dof_vector(solution_advected_field, advec_diff_dof_idx);
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(advec_diff_dof_idx),
                                     initial_field_function,
                                     solution_advected_field);

    scratch_data->get_constraint(advec_diff_dof_idx).distribute(solution_advected_field);
    scratch_data->initialize_dof_vector(solution_advected_field_old, advec_diff_dof_idx);

    solution_advected_field_old.copy_locally_owned_data_from(solution_advected_field);
    scratch_data->initialize_dof_vector(user_rhs, advec_diff_dof_idx);
    scratch_data->initialize_dof_vector(rhs, advec_diff_dof_idx);
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::reinit()
  {
    scratch_data->initialize_dof_vector(solution_advected_field, advec_diff_dof_idx);
    scratch_data->initialize_dof_vector(solution_advected_field_old, advec_diff_dof_idx);
    scratch_data->initialize_dof_vector(user_rhs, advec_diff_dof_idx);
    scratch_data->initialize_dof_vector(rhs, advec_diff_dof_idx);
    /*
     *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */
    if (!this->advec_diff_data.linear_solver.do_matrix_free)
      advec_diff_operator->initialize_matrix_based(*scratch_data);

    if (this->advec_diff_data.linear_solver.do_matrix_free && preconditioner_matrixfree)
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
  AdvectionDiffusionOperation<dim>::init_time_advance(const double dt)
  {
    VectorType level_set_extrapolated;
    scratch_data->initialize_dof_vector(level_set_extrapolated, advec_diff_dof_idx);

    UtilityFunctions::compute_linear_predictor(
      solution_advected_field, solution_advected_field_old, level_set_extrapolated, dt, dt);

    solution_advected_field_old.copy_locally_owned_data_from(solution_advected_field);
    solution_advected_field.copy_locally_owned_data_from(level_set_extrapolated);

    // apply hanging node constraints to predictor
    scratch_data->get_constraint(advec_diff_dof_idx).distribute(solution_advected_field);
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::solve(const double dt, const VectorType &advection_velocity)
  {
    init_time_advance(dt);
    rhs = 0.0;

    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  VectorTools::compute_L2_norm<dim>(advection_velocity,
                                                                    *scratch_data,
                                                                    velocity_dof_idx,
                                                                    advec_diff_quad_idx),
                                  "velocity",
                                  "advection_diffusion");

    advection_velocity.update_ghost_values();
    solution_advected_field_old.update_ghost_values();

    if (!advec_diff_operator)
      create_operator(advection_velocity);

    advec_diff_operator->reset_time_increment(dt);

    int iter = 0;

    if (this->advec_diff_data.linear_solver.do_matrix_free)
      {
        /*
         * apply dirichlet boundary values
         */
        rhs = user_rhs;

        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree(
          *advec_diff_operator,
          rhs,
          solution_advected_field_old,
          *scratch_data,
          advec_diff_dof_idx,
          advec_diff_hanging_nodes_dof_idx,
          false /*zero_out rhs*/);

        if (this->advec_diff_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          {
            auto diag_preconditioner_matrixfree =
              preconditioner_matrixfree->compute_diagonal_preconditioner();

            iter = LinearSolver::solve<VectorType>(*advec_diff_operator,
                                                   solution_advected_field,
                                                   rhs,
                                                   this->advec_diff_data.linear_solver,
                                                   *diag_preconditioner_matrixfree);
          }
        else
          {
            auto trilinos_preconditioner_matrixfree =
              preconditioner_matrixfree->compute_trilinos_preconditioner();

            iter = LinearSolver::solve<VectorType>(*advec_diff_operator,
                                                   solution_advected_field,
                                                   rhs,
                                                   this->advec_diff_data.linear_solver,
                                                   *trilinos_preconditioner_matrixfree);
          }
      }
    else
      {
        advec_diff_operator->assemble_matrixbased(solution_advected_field_old,
                                                  advec_diff_operator->get_system_matrix(),
                                                  rhs);

        rhs += user_rhs;
        scratch_data->get_constraint(advec_diff_dof_idx)
          .distribute(rhs); //@todo: this could be avoided by introducing a zero_out inside
                            // assemble_matrixbased

        auto preconditioner = Preconditioner::get_preconditioner_trilinos(
          advec_diff_operator->get_system_matrix(),
          this->advec_diff_data.linear_solver.preconditioner_type);
        iter = LinearSolver::solve<VectorType>(advec_diff_operator->get_system_matrix(),
                                               solution_advected_field,
                                               rhs,
                                               this->advec_diff_data.linear_solver,
                                               *preconditioner);
      }

    scratch_data->get_constraint(advec_diff_dof_idx).distribute(solution_advected_field);

    Journal::print_formatted_norm(scratch_data->get_pcout(2),
                                  advec_diff_operator->get_system_matrix().frobenius_norm(),
                                  "matrix",
                                  "advection_diffusion",
                                  6 /*precision*/,
                                  "F");

    Journal::print_formatted_norm(scratch_data->get_pcout(2),
                                  rhs.l2_norm(),
                                  "rhs",
                                  "advection_diffusion",
                                  6 /*precision*/,
                                  "l2");
    Journal::print_formatted_norm(scratch_data->get_pcout(2),
                                  solution_advected_field.l2_norm(),
                                  "src",
                                  "advection_diffusion",
                                  6 /*precision*/,
                                  "l2");

    Journal::print_formatted_norm(
      scratch_data->get_pcout(0),
      MeltPoolDG::VectorTools::compute_L2_norm<dim>(
        solution_advected_field, *scratch_data, advec_diff_dof_idx, advec_diff_quad_idx),
      "advected field",
      "advection_diffusion",
      10 /*precision*/
    );

    Journal::print_line(scratch_data->get_pcout(2),
                        "     * GMRES: i = " + std::to_string(iter),
                        "advection_diffusion");

    solution_advected_field_old.zero_out_ghost_values();
    advection_velocity.zero_out_ghost_values();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperation<dim>::get_advected_field() const
  {
    return solution_advected_field;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperation<dim>::get_advected_field()
  {
    return solution_advected_field;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperation<dim>::get_advected_field_old() const
  {
    return solution_advected_field_old;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperation<dim>::get_advected_field_old()
  {
    return solution_advected_field_old;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperation<dim>::get_user_rhs()
  {
    return user_rhs;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperation<dim>::get_user_rhs() const
  {
    return user_rhs;
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solution_advected_field.update_ghost_values();
    vectors.push_back(&solution_advected_field);
    solution_advected_field_old.update_ghost_values();
    vectors.push_back(&solution_advected_field_old);
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data->get_dof_handler(advec_diff_dof_idx),
                             solution_advected_field,
                             "advected_field");
    data_out.add_data_vector(scratch_data->get_dof_handler(advec_diff_dof_idx),
                             user_rhs,
                             "advec_diff_user_rhs");
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::create_operator(const VectorType &advection_velocity)
  {
    advec_diff_operator = std::make_unique<AdvectionDiffusionOperator<dim>>(*scratch_data,
                                                                            advection_velocity,
                                                                            this->advec_diff_data,
                                                                            advec_diff_dof_idx,
                                                                            advec_diff_quad_idx,
                                                                            velocity_dof_idx);
    /*
     *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */
    if (!this->advec_diff_data.linear_solver.do_matrix_free)
      advec_diff_operator->initialize_matrix_based(*scratch_data);
    /*
     * initialize preconditioner matrix-free
     */
    if (this->advec_diff_data.linear_solver.do_matrix_free)
      {
        preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          *scratch_data,
          advec_diff_dof_idx,
          this->advec_diff_data.linear_solver.preconditioner_type,
          *advec_diff_operator);

        preconditioner_matrixfree->reinit();
      }
  }

  template class AdvectionDiffusionOperation<1>;
  template class AdvectionDiffusionOperation<2>;
  template class AdvectionDiffusionOperation<3>;
} // namespace MeltPoolDG::AdvectionDiffusion
