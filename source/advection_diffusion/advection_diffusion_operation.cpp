#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
//

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/utilities/linearsolve.hpp>
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
    dealii::VectorTools::project(scratch_data->get_mapping(),
                                 scratch_data->get_dof_handler(advec_diff_dof_idx),
                                 scratch_data->get_constraint(advec_diff_dof_idx),
                                 scratch_data->get_quadrature(advec_diff_quad_idx),
                                 initial_field_function,
                                 solution_advected_field);
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

    scratch_data->get_pcout() << "|vel|= " << advection_velocity.l2_norm() << std::endl;

    create_operator(advection_velocity);

    VectorType src, rhs;

    scratch_data->initialize_dof_vector(src, advec_diff_dof_idx);
    scratch_data->initialize_dof_vector(rhs, advec_diff_dof_idx);

    advec_diff_operator->set_time_increment(dt);

    int iter = 0;

    solution_advected_field.update_ghost_values();

    if (this->advec_diff_data.do_matrix_free)
      {
        /*
         * apply dirichlet boundary values
         */
        advec_diff_operator->create_rhs_and_apply_dirichlet_mf(rhs,
                                                               solution_advected_field,
                                                               *scratch_data,
                                                               advec_diff_dof_idx,
                                                               advec_diff_hanging_nodes_dof_idx);

        iter = LinearSolve::solve<VectorType, SolverGMRES<VectorType>, OperatorBase<dim, double>>(
          *advec_diff_operator, src, rhs);
      }
    else
      {
        //@todo: which preconditioner?
        // TrilinosWrappers::PreconditionAMG preconditioner;
        // TrilinosWrappers::PreconditionAMG::AdditionalData data;

        // preconditioner.initialize(system_matrix, data);
        advec_diff_operator->assemble_matrixbased(solution_advected_field,
                                                  advec_diff_operator->system_matrix,
                                                  rhs);
        iter = LinearSolve::solve<VectorType, SolverGMRES<VectorType>, SparseMatrixType>(
          advec_diff_operator->system_matrix, src, rhs);
      }

    scratch_data->get_constraint(advec_diff_dof_idx).distribute(src);

    solution_advected_field.copy_locally_owned_data_from(src);
    solution_advected_field.update_ghost_values();

    scratch_data->get_pcout(2) << "|matrix|= "
                               << advec_diff_operator->system_matrix.frobenius_norm() << std::endl;
    scratch_data->get_pcout(2) << "|rhs|= " << rhs.l2_norm() << std::endl;
    scratch_data->get_pcout(2) << "|src|= " << src.l2_norm() << std::endl;

    scratch_data->get_pcout(1) << "| GMRES: i=" << std::setw(5) << std::left << iter;

    const auto &pcout = scratch_data->get_pcout();
    pcout << "\t |ϕ|2 = " << std::setw(15) << std::left << std::setprecision(10)
          << MeltPoolDG::VectorTools::compute_L2_norm<dim>(solution_advected_field,
                                                           *scratch_data,
                                                           advec_diff_dof_idx,
                                                           advec_diff_quad_idx)
          << std::endl;
    advection_velocity.zero_out_ghosts();
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
    solution_advected_field.update_ghost_values();
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
    if (!this->advec_diff_data.do_matrix_free)
      advec_diff_operator->initialize_matrix_based(*scratch_data);
  }

  template class AdvectionDiffusionOperation<1>;
  template class AdvectionDiffusionOperation<2>;
  template class AdvectionDiffusionOperation<3>;
} // namespace MeltPoolDG::AdvectionDiffusion
