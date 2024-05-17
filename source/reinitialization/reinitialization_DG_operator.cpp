#include <meltpooldg/reinitialization/reinitialization_DG_operator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{

  template <int dim, typename Number>
  ReinitilizationDGOperator<dim, Number>::ReinitilizationDGOperator(
    const MeltPoolDG::ScratchData<dim> &scratch_data_in,
    const ReinitializationData<double> &reinit_data_in,
    const unsigned int                  reinit_dof_idx_in,
    const unsigned int                  reinit_quad_idx_in)
    : scratch_data(scratch_data_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , reinit_data(reinit_data_in)
    , RI_DG_diffusion_operator(scratch_data_in,
                               reinit_data_in,
                               reinit_dof_idx_in,
                               reinit_quad_idx_in)

    , RI_grad_operator(scratch_data_in, reinit_dof_idx_in, reinit_quad_idx_in)
  {
    if (reinit_data_in.reinitilization_DG_specific_data.IMEX_integration_scheme !=
        TimeIntegrators::not_initialized)
      IMEX_integration = TimeIntegratorConcretization::concretize(
        reinit_data_in.reinitilization_DG_specific_data.IMEX_integration_scheme,
        RI_DG_diffusion_operator,
        scratch_data_in,
        reinit_dof_idx_in,
        reinit_quad_idx_in,
        reinit_data_in.linear_solver);
  }


  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::prepare_operator(const VectorType &solution)
  {
    Number const min_vertex_distance = scratch_data.get_min_cell_size();
    compute_godunov_gradient(solution);
    compute_smoothed_signum(solution, min_vertex_distance);
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::apply_diffusion_implicit(
    const Number                                  time,
    const Number                                  time_step,
    TimeIntegration::SolutionHistory<VectorType> &solution_history,
    VectorType                                   &rhs) const
  {
    IMEX_integration->perform_time_step(time, time_step, solution_history, rhs);
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::apply_operator(Number const      time,
                                                         VectorType       &dst,
                                                         VectorType const &src)
  {
    compute_godunov_hamiltonian(src);

    scratch_data.get_matrix_free().cell_loop(
      &ReinitilizationDGOperator<dim, Number>::local_apply_domain_num_Hamiltonian,
      this,
      dst,
      num_Hamiltonian,
      true);


    if (reinit_data.reinitilization_DG_specific_data.IMEX_integration_scheme ==
        TimeIntegrators::not_initialized)
      {
        RI_DG_diffusion_operator.apply_operator(time, num_Hamiltonian, src);
        RI_DG_diffusion_operator.apply_dirichlet_boundary_operator(time, num_Hamiltonian, src);


        dst.add(1.0, num_Hamiltonian);
      }
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::reinit()
  {
    scratch_data.initialize_dof_vector(num_Hamiltonian, reinit_dof_idx);
    scratch_data.initialize_dof_vector(signum_smoothed, reinit_dof_idx);
    scratch_data.initialize_dof_vector(God_grad, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_x_l, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_x_r, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_y_l, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_y_r, reinit_dof_idx);

    if constexpr (dim == 3)
      {
        scratch_data.initialize_dof_vector(grad_z_l, reinit_dof_idx);
        scratch_data.initialize_dof_vector(grad_z_r, reinit_dof_idx);
      }
    if (reinit_data.reinitilization_DG_specific_data.IMEX_integration_scheme !=
        TimeIntegrators::not_initialized)
      IMEX_integration->reinit();

    RI_DG_diffusion_operator.compute_viscosity_value();
    RI_DG_diffusion_operator.compute_penalty_parameter();
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::compute_godunov_hamiltonian(const VectorType &solution)
  {
    Number const gradient_goal = 1.;

    if (reinit_data.reinitilization_DG_specific_data.use_const_gradient_in_RI == false)
      {
        compute_godunov_gradient(solution);
      }

    num_Hamiltonian = God_grad;
    num_Hamiltonian.add(-gradient_goal);
    num_Hamiltonian.scale(signum_smoothed);
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::local_apply_domain_num_Hamiltonian(
    const MatrixFree<dim, Number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, Number> eval(data, reinit_dof_idx, reinit_quad_idx);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u = eval.get_value(q);
            // minus sign because the term is shifted to the right-hand side of the RI equation
            eval.submit_value(-u, q);
          }

        eval.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::compute_godunov_gradient(const VectorType &solution)
  {
    {
      // compute local upwind and downwind gradients
      //  x-direction
      RI_grad_operator.template apply<false, 0>(solution, grad_x_l);
      RI_grad_operator.template apply<true, 0>(solution, grad_x_r);
      if constexpr (dim != 1)
        {
          // y-direction
          RI_grad_operator.template apply<false, 1>(solution, grad_y_l);
          RI_grad_operator.template apply<true, 1>(solution, grad_y_r);
        }

      if constexpr (dim == 3)
        {
          // z-direction
          RI_grad_operator.template apply<false, 2>(solution, grad_z_l);
          RI_grad_operator.template apply<true, 2>(solution, grad_z_r);
        }
    }

    DEAL_II_OPENMP_SIMD_PRAGMA
    for (unsigned int i = 0; i < grad_x_l.locally_owned_size(); ++i)
      {
        double argument_one = std::max(std::min(grad_x_l.local_element(i), 0.0) *
                                         std::min(grad_x_l.local_element(i), 0.0),
                                       std::max(grad_x_r.local_element(i), 0.0) *
                                         std::max(grad_x_r.local_element(i), 0.0));

        if constexpr (dim != 1)
          {
            argument_one += std::max(std::min(grad_y_l.local_element(i), 0.0) *
                                       std::min(grad_y_l.local_element(i), 0.0),
                                     std::max(grad_y_r.local_element(i), 0.0) *
                                       std::max(grad_y_r.local_element(i), 0.0));
          }

        if constexpr (dim == 3)
          {
            argument_one += std::max(std::min(grad_z_l.local_element(i), 0.0) *
                                       std::min(grad_z_l.local_element(i), 0.0),
                                     std::max(grad_z_r.local_element(i), 0.0) *
                                       std::max(grad_z_r.local_element(i), 0.0));
          }


        double argument_two = std::max(std::max(grad_x_l.local_element(i), 0.0) *
                                         std::max(grad_x_l.local_element(i), 0.0),
                                       std::min(grad_x_r.local_element(i), 0.0) *
                                         std::min(grad_x_r.local_element(i), 0.0));

        if constexpr (dim != 1)
          {
            argument_two += std::max(std::max(grad_y_l.local_element(i), 0.0) *
                                       std::max(grad_y_l.local_element(i), 0.0),
                                     std::min(grad_y_r.local_element(i), 0.0) *
                                       std::min(grad_y_r.local_element(i), 0.0));
          }

        if constexpr (dim == 3)
          {
            argument_two += std::max(std::max(grad_z_l.local_element(i), 0.0) *
                                       std::max(grad_z_l.local_element(i), 0.0),
                                     std::min(grad_z_r.local_element(i), 0.0) *
                                       std::min(grad_z_r.local_element(i), 0.0));
          }

        if (solution.local_element(i) > 0.0)
          God_grad.local_element(i) = std::sqrt(argument_one);
        else
          {
            God_grad.local_element(i) = std::sqrt(argument_two);
          }
      }
  }


  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::compute_smoothed_signum(
    const VectorType &solution,
    const Number      min_vertex_distance) const
  {
    const auto  &data      = scratch_data.get_matrix_free();
    const Number fe_degree = ((static_cast<Number>(scratch_data.get_degree(reinit_dof_idx))));

    {
      // Use maximum element size for the calculation of the argument of tanh
      const dealii::VectorizedArray<Number> eta_vector = 2.0 * min_vertex_distance / fe_degree;


      FECellIntegrator<dim, 1, Number> source(data, reinit_dof_idx, reinit_quad_idx);
      FECellIntegrator<dim, 1, Number> God_grad_p(data, reinit_dof_idx, reinit_quad_idx);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          source.reinit(cell);
          source.read_dof_values(solution);
          God_grad_p.reinit(cell);
          God_grad_p.read_dof_values(God_grad);

          for (unsigned int q = 0; q < source.dofs_per_cell; ++q)
            {
              // calculate argument of tanh: (pi*phi)/(2*grad(phi)*max_cell_size/fe_degree)
              const auto arg = numbers::PI * source.get_dof_value(q) /
                               (eta_vector * std::abs(God_grad_p.get_dof_value(q)));

              const auto u = MeltPoolDG::VectorTools::tanh<dim>(arg);

              source.submit_dof_value(u, q);
            }

          source.set_dof_values(signum_smoothed);
        }
    }
  }


  template <int dim, typename Number>
  double
  ReinitilizationDGOperator<dim, Number>::get_viscosity() const
  {
    return RI_DG_diffusion_operator.get_viscosity();
  }


  template class ReinitilizationDGOperator<1>;
  template class ReinitilizationDGOperator<2>;
  template class ReinitilizationDGOperator<3>;
} // namespace MeltPoolDG::LevelSet
