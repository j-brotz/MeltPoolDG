#include <meltpooldg/level_set/reinitialization_DG_operator.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitilizationDGOperator<dim, number>::ReinitilizationDGOperator(
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
    const ReinitializationData<number>              &reinit_data_in,
    const unsigned int                               reinit_dof_idx_in,
    const unsigned int                               reinit_quad_idx_in,
    const VectorType                                &curvature_in,
    const BlockVectorType                           &normal_vector_in)
    : scratch_data(scratch_data_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , reinit_data(reinit_data_in)
    , RI_DG_diffusion_operator(scratch_data_in,
                               reinit_data_in,
                               reinit_dof_idx_in,
                               reinit_quad_idx_in,
                               curvature_in,
                               normal_vector_in,
                               signum_smoothed)

    , RI_grad_operator(scratch_data_in, reinit_dof_idx_in, reinit_quad_idx_in)
  {
    if (reinit_data.reinitilization_DG_specific_data.IMEX_integration_data.integrator_type !=
        TimeIntegratorSchemes::not_initialized)
      IMEX_integration =
        std::shared_ptr<TimeIntegratorBase<number>>(time_integrator_factory<number>(
          RI_DG_diffusion_operator,
          reinit_data_in.reinitilization_DG_specific_data.IMEX_integration_data,
          reinit_data_in.linear_solver,
          scratch_data.get_timer()));
  }


  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::prepare_operator(const VectorType &solution)
  {
    number const min_vertex_distance = scratch_data.get_min_cell_size();
    compute_godunov_gradient(solution);
    compute_smoothed_signum(solution, min_vertex_distance);
    RI_DG_diffusion_operator.compute_diffusitivity_value();
    RI_DG_diffusion_operator.compute_penalty_parameter();
  }

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::apply_diffusion_implicit(
    const number                                  time,
    const number                                  time_step,
    TimeIntegration::SolutionHistory<VectorType, number> &solution_history) const
  {
    IMEX_integration->perform_time_step(time, time_step, solution_history);
  }

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::apply_operator(
    number const                                           time,
    VectorType                                            &dst,
    VectorType const                                      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    compute_godunov_hamiltonian(src);

    scratch_data.get_matrix_free().cell_loop(
      &ReinitilizationDGOperator<dim, number>::local_apply_domain_num_Hamiltonian,
      this,
      dst,
      num_Hamiltonian,
      true);

    scratch_data.get_matrix_free().cell_loop(
      &ReinitilizationDGOperator<dim, number>::local_apply_inverse_mass_matrix, this, dst, dst);



    if (reinit_data.reinitilization_DG_specific_data.IMEX_integration_data.integrator_type ==
        TimeIntegratorSchemes::not_initialized)
      {
        RI_DG_diffusion_operator.apply_operator(time, num_Hamiltonian, src);
        RI_DG_diffusion_operator.apply_dirichlet_boundary_operator(time, num_Hamiltonian, src);

        dst.add(1.0, num_Hamiltonian);
      }

    if (reinit_data.reinitilization_DG_specific_data.use_interface_movement_penalization)
      {
        scratch_data.get_matrix_free().cell_loop(
          &ReinitilizationDGOperator<dim, number>::interface_movement_penalty,
          this,
          num_Hamiltonian,
          src,
          true);

        scratch_data.get_matrix_free().cell_loop(
          &ReinitilizationDGOperator<dim, number>::local_apply_inverse_mass_matrix,
          this,
          num_Hamiltonian,
          num_Hamiltonian);

        dst.add(1.0, num_Hamiltonian);
      }
    func(0, dst.locally_owned_size());
  }

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FECellIntegrator<dim, 1, number> eval(data, reinit_dof_idx, reinit_quad_idx);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, number> inverse(eval);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);

        inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

        eval.set_dof_values(dst);
      }
  }


  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::reinit()
  {
    scratch_data.initialize_dof_vector(num_Hamiltonian, reinit_dof_idx);
    scratch_data.initialize_dof_vector(signum_smoothed, reinit_dof_idx);
    scratch_data.initialize_dof_vector(God_grad, reinit_dof_idx);
    scratch_data.initialize_dof_vector(sign_indicator_function, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_x_l, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_x_r, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_y_l, reinit_dof_idx);
    scratch_data.initialize_dof_vector(grad_y_r, reinit_dof_idx);

    if constexpr (dim == 3)
      {
        scratch_data.initialize_dof_vector(grad_z_l, reinit_dof_idx);
        scratch_data.initialize_dof_vector(grad_z_r, reinit_dof_idx);
      }
    if (reinit_data.reinitilization_DG_specific_data.IMEX_integration_data.integrator_type !=
        TimeIntegratorSchemes::not_initialized)
      IMEX_integration->reinit(grad_x_l); // TODO: Pass scratch data to initialize vectors

    RI_DG_diffusion_operator.compute_diffusitivity_value();
    RI_DG_diffusion_operator.compute_penalty_parameter();
  }

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::compute_godunov_hamiltonian(
    const VectorType &solution) const
  {
    number const gradient_goal = 1.;

    if (reinit_data.reinitilization_DG_specific_data.use_const_gradient_in_RI == false)
      {
        compute_godunov_gradient(solution);
      }

    num_Hamiltonian = God_grad;
    num_Hamiltonian.add(-gradient_goal);
    num_Hamiltonian.scale(signum_smoothed);
  }

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::local_apply_domain_num_Hamiltonian(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, number> eval(data, reinit_dof_idx, reinit_quad_idx);


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

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::interface_movement_penalty(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator<dim, 1, number> eval(data, reinit_dof_idx, reinit_quad_idx);
    FECellIntegrator<dim, 1, number> eval_sig_indicator(data, reinit_dof_idx, reinit_quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::values);

        eval_sig_indicator.reinit(cell);
        eval_sig_indicator.gather_evaluate(sign_indicator_function, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u              = eval.get_value(q);
            const auto sign_indicator = eval_sig_indicator.get_value(q);
            const auto flux =
              compare_and_apply_mask<SIMDComparison::greater_than>(u,
                                                                   VectorizedArray<number>(0.),
                                                                   VectorizedArray<number>(1.),
                                                                   VectorizedArray<number>(0.)) -
              compare_and_apply_mask<SIMDComparison::greater_than>(sign_indicator,
                                                                   VectorizedArray<number>(0.),
                                                                   VectorizedArray<number>(1.),
                                                                   VectorizedArray<number>(0.));
            eval.submit_value(-flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::compute_godunov_gradient(const VectorType &solution) const
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
        number argument_one = std::max(std::min(grad_x_l.local_element(i), 0.0) *
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


        number argument_two = std::max(std::max(grad_x_l.local_element(i), 0.0) *
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

        if (signum_smoothed.local_element(i) >= 0.0)
          God_grad.local_element(i) = std::sqrt(argument_one);
        else
          {
            God_grad.local_element(i) = std::sqrt(argument_two);
          }
      }
  }


  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::compute_smoothed_signum(
    const VectorType &solution,
    const number      min_vertex_distance) const
  {
    const auto  &data      = scratch_data.get_matrix_free();
    const number fe_degree = ((static_cast<number>(scratch_data.get_degree(reinit_dof_idx))));

    {
      // Use maximum element size for the calculation of the argument of tanh
      const dealii::VectorizedArray<number> eta_vector =
        reinit_data.reinitilization_DG_specific_data.signum_smoothness_paramater *
        min_vertex_distance / fe_degree;


      FECellIntegrator<dim, 1, number> source(data, reinit_dof_idx, reinit_quad_idx);
      FECellIntegrator<dim, 1, number> God_grad_p(data, reinit_dof_idx, reinit_quad_idx);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          source.reinit(cell);
          source.read_dof_values(solution);
          God_grad_p.reinit(cell);
          God_grad_p.read_dof_values(God_grad);

          for (unsigned int q = 0; q < source.dofs_per_cell; ++q)
            {
              if (reinit_data.reinitilization_DG_specific_data.hyperbolic_weighting_function_type ==
                  HyperbolicWeightingFunctionType::smoothed_signum)
                {
                  // calculate argument of tanh: (pi*phi)/(2*grad(phi)*max_cell_size/fe_degree)
                  const auto arg =
                    numbers::PI * source.get_dof_value(q) /
                    (eta_vector * std::abs(God_grad_p.get_dof_value(q)) +
                     reinit_data.reinitilization_DG_specific_data
                       .avoid_zero_division_smoothed_signum); // In case Godunov gradient is zero

                  const auto u = std::tanh(arg);
                  source.submit_dof_value(u, q);
                }
              else // The standard case is the level set before reinit
                {
                  source.submit_dof_value(source.get_dof_value(q), q);
                }
            }

          source.set_dof_values(signum_smoothed);
        }
    }
  }


  template <int dim, typename number>
  number
  ReinitilizationDGOperator<dim, number>::get_max_diffusitivity() const
  {
    return RI_DG_diffusion_operator.get_max_diffusitivity();
  }

  template <int dim, typename number>
  void
  ReinitilizationDGOperator<dim, number>::set_artificial_diffusitivity()
  {
    RI_DG_diffusion_operator.compute_diffusitivity_value();
    RI_DG_diffusion_operator.compute_penalty_parameter();
  }

  template class ReinitilizationDGOperator<1, double>;
  template class ReinitilizationDGOperator<2, double>;
  template class ReinitilizationDGOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
