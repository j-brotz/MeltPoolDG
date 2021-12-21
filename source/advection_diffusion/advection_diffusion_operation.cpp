#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/linear_solve.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::initialize(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &                     data_in,
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
    this->advec_diff_data = data_in.advec_diff;
    /*
     *  set the parameters for the advection_diffusion problem
     */
    set_advection_diffusion_parameters(data_in);
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::set_initial_condition(
    const Function<dim> &initial_field_function,
    const VectorType &   initial_velocity)
  {
    (void)initial_velocity;
    reinit();
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(advec_diff_dof_idx),
                                     initial_field_function,
                                     solution_advected_field);

    scratch_data->get_constraint(advec_diff_dof_idx).distribute(solution_advected_field);
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::reinit()
  {
    scratch_data->initialize_dof_vector(solution_advected_field, advec_diff_dof_idx);
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::solve(const double dt, const VectorType &advection_velocity)
  {
    advection_velocity.update_ghost_values();

    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  VectorTools::compute_L2_norm<dim>(advection_velocity,
                                                                    *scratch_data,
                                                                    velocity_dof_idx,
                                                                    advec_diff_quad_idx),
                                  "velocity",
                                  "advection_diffusion");

    create_operator(advection_velocity);

    VectorType src, rhs;

    scratch_data->initialize_dof_vector(src, advec_diff_dof_idx);
    scratch_data->initialize_dof_vector(rhs, advec_diff_dof_idx);

    advec_diff_operator->reset_time_increment(dt);

    int iter = 0;

    solution_advected_field.update_ghost_values();

    if (this->advec_diff_data.linear_solver.do_matrix_free)
      {
        /*
         * apply dirichlet boundary values
         */
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree(
          *advec_diff_operator,
          rhs,
          solution_advected_field,
          *scratch_data,
          advec_diff_dof_idx,
          advec_diff_hanging_nodes_dof_idx);

        iter = LinearSolve::solve<VectorType>(*advec_diff_operator,
                                              src,
                                              rhs,
                                              this->advec_diff_data.linear_solver.solver_type);
      }
    else
      {
        //@todo: which preconditioner?
        // TrilinosWrappers::PreconditionAMG preconditioner;
        // TrilinosWrappers::PreconditionAMG::AdditionalData data;

        // preconditioner.initialize(system_matrix, data);
        advec_diff_operator->assemble_matrixbased(solution_advected_field,
                                                  advec_diff_operator->get_system_matrix(),
                                                  rhs);
        iter = LinearSolve::solve<VectorType>(advec_diff_operator->get_system_matrix(),
                                              src,
                                              rhs,
                                              this->advec_diff_data.linear_solver.solver_type);
      }

    scratch_data->get_constraint(advec_diff_dof_idx).distribute(src);

    solution_advected_field.copy_locally_owned_data_from(src);
    solution_advected_field.update_ghost_values();

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
                                  src.l2_norm(),
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
    return solution_advected_field;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperation<dim>::get_advected_field_old()
  {
    return solution_advected_field;
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solution_advected_field.update_ghost_values();
    vectors.push_back(&solution_advected_field);
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data->get_dof_handler(advec_diff_dof_idx),
                             solution_advected_field,
                             "advected_field");
  }

  template <int dim>
  void
  AdvectionDiffusionOperation<dim>::set_advection_diffusion_parameters(
    const Parameters<double> &data_in)
  {
    this->advec_diff_data = data_in.advec_diff; //@todo is this really needed?
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
  }

  template class AdvectionDiffusionOperation<1>;
  template class AdvectionDiffusionOperation<2>;
  template class AdvectionDiffusionOperation<3>;
} // namespace MeltPoolDG::AdvectionDiffusion
