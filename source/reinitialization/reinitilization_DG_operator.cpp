#include <meltpooldg/reinitialization/reinitilization_DG_operator.hpp>

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
    IMEX_integration =
      TimeIntegratorConcretization::concretize(reinit_data_in.IMEX_integration_scheme,
                                               RI_DG_diffusion_operator,
                                               scratch_data_in,
                                               reinit_dof_idx_in,
                                               reinit_quad_idx_in,
                                               reinit_data_in.linear_solver);
  }


  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::prepare_operator(VectorType &solution)
  {
    Number const min_vertex_distance = scratch_data.get_min_cell_size();
    Godunov_gradient(solution);
    Smoothed_signum(solution, min_vertex_distance);
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::apply_IMEX(
    Number const                                  time,
    Number const                                  time_step,
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
    Godunov_Hamiltonian(src);

    scratch_data.get_matrix_free().cell_loop(
      &ReinitilizationDGOperator<dim, Number>::local_apply_domain_num_Hamiltonian,
      this,
      dst,
      num_Hamiltonian,
      true);


    if (reinit_data.use_IMEX == false)
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
    initialize_dof_vector(num_Hamiltonian, reinit_dof_idx);
    initialize_dof_vector(Signum_smoothed, reinit_dof_idx);
    initialize_dof_vector(God_grad, reinit_dof_idx);
    initialize_dof_vector(grad_x_l, reinit_dof_idx);
    initialize_dof_vector(grad_x_r, reinit_dof_idx);
    initialize_dof_vector(grad_y_l, reinit_dof_idx);
    initialize_dof_vector(grad_y_r, reinit_dof_idx);

    if constexpr (dim == 3)
      {
        initialize_dof_vector(grad_z_l, reinit_dof_idx);
        initialize_dof_vector(grad_z_r, reinit_dof_idx);
      }
    IMEX_integration->reinit();
  }



  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::compute_viscosity_value()
  {
    RI_DG_diffusion_operator.compute_viscosity_value();
  }


  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::compute_penalty_parameter()
  {
    RI_DG_diffusion_operator.compute_penalty_parameter();
  }

  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::Godunov_Hamiltonian(const VectorType &solution)
  {
    Number const gradient_goal = 1.;

    if (reinit_data.use_const_gradient_in_RI == false)
      {
        Godunov_gradient(solution);
      }

    num_Hamiltonian = God_grad;
    num_Hamiltonian.add(-gradient_goal);
    num_Hamiltonian.scale(Signum_smoothed);
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
  ReinitilizationDGOperator<dim, Number>::Godunov_gradient(const VectorType &solution)
  {
    const auto &data = scratch_data.get_matrix_free();

    {
      // compute local upwind and downwind gradients
      //  x-direction
      RI_grad_operator.template apply_RI_grad<false, 0>(solution, grad_x_l);
      RI_grad_operator.template apply_RI_grad<true, 0>(solution, grad_x_r);
      // y-direction
      RI_grad_operator.template apply_RI_grad<false, 1>(solution, grad_y_l);
      RI_grad_operator.template apply_RI_grad<true, 1>(solution, grad_y_r);

      if constexpr (dim == 3)
        {
          // z-direction
          RI_grad_operator.template apply_RI_grad<false, 2>(solution, grad_z_l);
          RI_grad_operator.template apply_RI_grad<true, 2>(solution, grad_z_r);
        }
    }

    {
      const dealii::VectorizedArray<Number> zero_vector = 0;



      FECellIntegrator<dim, 1, Number> phi_grad_x_l(data, reinit_dof_idx, reinit_quad_idx);
      FECellIntegrator<dim, 1, Number> phi_grad_x_r(data, reinit_dof_idx, reinit_quad_idx);
      FECellIntegrator<dim, 1, Number> phi_grad_y_l(data, reinit_dof_idx, reinit_quad_idx);
      FECellIntegrator<dim, 1, Number> phi_grad_y_r(data, reinit_dof_idx, reinit_quad_idx);

      FECellIntegrator<dim, 1, Number> phi_grad_z_l(data, reinit_dof_idx, reinit_quad_idx);
      FECellIntegrator<dim, 1, Number> phi_grad_z_r(data, reinit_dof_idx, reinit_quad_idx);

      FECellIntegrator<dim, 1, Number> phi_sol(data, reinit_dof_idx, reinit_quad_idx);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi_grad_x_l.reinit(cell);
          phi_grad_x_l.read_dof_values(grad_x_l);
          phi_grad_x_r.reinit(cell);
          phi_grad_x_r.read_dof_values(grad_x_r);
          phi_grad_y_l.reinit(cell);
          phi_grad_y_l.read_dof_values(grad_y_l);
          phi_grad_y_r.reinit(cell);
          phi_grad_y_r.read_dof_values(grad_y_r);

          if constexpr (dim == 3)
            {
              phi_grad_z_l.reinit(cell);
              phi_grad_z_l.read_dof_values(grad_z_l);
              phi_grad_z_r.reinit(cell);
              phi_grad_z_r.read_dof_values(grad_z_r);
            }

          phi_sol.reinit(cell);
          phi_sol.read_dof_values(solution);

          for (unsigned int q = 0; q < phi_grad_x_l.dofs_per_cell; ++q)
            {
              auto argument_one =
                std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector)) *
                           (std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                         (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)) *
                           (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

                + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector)) *
                             (std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                           (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)) *
                             (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)));

              auto argument_two =
                std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector)) *
                           (std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                         (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)) *
                           (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

                + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector)) *
                             (std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                           (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)) *
                             (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)));
              if constexpr (dim == 3)
                {
                  argument_one +=
                    std::max((std::min(phi_grad_z_l.get_dof_value(q), zero_vector)) *
                               (std::min(phi_grad_z_l.get_dof_value(q), zero_vector)),
                             (std::max(phi_grad_z_r.get_dof_value(q), zero_vector)) *
                               (std::max(phi_grad_z_r.get_dof_value(q), zero_vector)));

                  argument_two +=
                    +std::max((std::max(phi_grad_z_l.get_dof_value(q), zero_vector)) *
                                (std::max(phi_grad_z_l.get_dof_value(q), zero_vector)),
                              (std::min(phi_grad_z_r.get_dof_value(q), zero_vector)) *
                                (std::min(phi_grad_z_r.get_dof_value(q), zero_vector)));
                }



              const auto u = compare_and_apply_mask<SIMDComparison::greater_than>(
                phi_sol.get_dof_value(q), 0., std::sqrt(argument_one), std::sqrt(argument_two)

              );
              phi_grad_x_l.submit_dof_value(u, q);
            }

          phi_grad_x_l.set_dof_values(God_grad);
        }
    }
  }


  template <int dim, typename Number>
  void
  ReinitilizationDGOperator<dim, Number>::Smoothed_signum(const VectorType &solution,
                                                          const Number min_vertex_distance) const
  {
    const auto  &data      = scratch_data.get_matrix_free();
    const Number fe_degree = (Number)scratch_data.get_degree(reinit_dof_idx);

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

              // tanh(x)=(e^x-e^(-x))/(e^x+e^(-x)), tanh(x) is not supported for VectorizedArray
              const auto u = (std::exp(arg) - std::exp(-arg)) / (std::exp(arg) + std::exp(-arg));

              source.submit_dof_value(u, q);
            }

          source.set_dof_values(Signum_smoothed);
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
