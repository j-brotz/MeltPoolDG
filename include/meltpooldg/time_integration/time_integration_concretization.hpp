#pragma once

#include <deal.II/base/exceptions.h>

#include <meltpooldg/time_integration/low_storage_runge_kutta.hpp>
#include <meltpooldg/time_integration/one_step_theta.hpp>
#include <meltpooldg/time_integration/time_integration_setup.hpp>


// Namespace for Time Integrator Concretization
namespace MeltPoolDG::TimeIntegratorConcretization
{
  template <typename Operator, int dim, typename Number = double>
  inline std::shared_ptr<TimeIntegrationBase<dim>>
  concretize(TimeIntegrators                     time_integration_scheme,
             Operator                           &pde_operator,
             const MeltPoolDG::ScratchData<dim> &scratch_data_in_,
             const unsigned int                  dof_idx_in,
             const unsigned int                  quad_idx_in,
             const LinearSolverData<Number>     &linear_solver_data_in)
  {
    if (time_integration_scheme == TimeIntegrators::RK_stage_1_order_1)
      {
        return std::make_shared<
          LowStorageRungeKuttaIntegrator<Operator, dim, TimeIntegrators::RK_stage_1_order_1>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    if (time_integration_scheme == TimeIntegrators::RK_stage_2_order_2)
      {
        return std::make_shared<
          LowStorageRungeKuttaIntegrator<Operator, dim, TimeIntegrators::RK_stage_2_order_2>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    if (time_integration_scheme == TimeIntegrators::RK_stage_3_order_3)
      {
        return std::make_shared<
          LowStorageRungeKuttaIntegrator<Operator, dim, TimeIntegrators::RK_stage_3_order_3>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    if (time_integration_scheme == TimeIntegrators::RK_stage_5_order_4)
      {
        return std::make_shared<
          LowStorageRungeKuttaIntegrator<Operator, dim, TimeIntegrators::RK_stage_5_order_4>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    if (time_integration_scheme == TimeIntegrators::RK_stage_7_order_4)
      {
        return std::make_shared<
          LowStorageRungeKuttaIntegrator<Operator, dim, TimeIntegrators::RK_stage_7_order_4>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    if (time_integration_scheme == TimeIntegrators::RK_stage_9_order_5)
      {
        return std::make_shared<
          LowStorageRungeKuttaIntegrator<Operator, dim, TimeIntegrators::RK_stage_9_order_5>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }
    if (time_integration_scheme == TimeIntegrators::explicit_euler)
      {
        return std::make_shared<OneStepTheta<Operator, dim, TimeIntegrators::explicit_euler>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    if (time_integration_scheme == TimeIntegrators::implicit_euler)
      {
        return std::make_shared<OneStepTheta<Operator, dim, TimeIntegrators::implicit_euler>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    if (time_integration_scheme == TimeIntegrators::crank_nicolson)
      {
        return std::make_shared<OneStepTheta<Operator, dim, TimeIntegrators::crank_nicolson>>(
          pde_operator, scratch_data_in_, dof_idx_in, quad_idx_in, linear_solver_data_in);
      }

    // Done here additionally to the check in advection diffusion data. Since BDF2 is implemented
    // for the CG case
    AssertThrow(false,
                ExcMessage("The chosen time integration scheme bdf2 is not implemented for DG"));
  }

} // namespace MeltPoolDG::TimeIntegratorConcretization
