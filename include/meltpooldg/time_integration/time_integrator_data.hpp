#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <string>

namespace MeltPoolDG
{
  BETTER_ENUM(TimeIntegratorSchemes,
              int,
              not_initialized,
              LSRK_stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
              LSRK_stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
              LSRK_stage_7_order_4, /* Tselios, Simos, 2007 */
              LSRK_stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
              implicit_euler,
              explicit_euler,
              crank_nicolson,
              bdf_1,
              bdf_2,
              bdf_3,
              bdf_4,
              bdf_5,
              bdf_6)

  /**
   * Collection of all integrator parameters.
   */
  struct TimeIntegratorData
  {
    /**
     * Default constructor.
     */
    TimeIntegratorData() = default;

    /**
     * Constructor setting a custom time integration scheme. This can be used to change the default
     * time integration scheme, when no scheme is passed by the input file.
     *
     * @param scheme Name of the new default scheme.
     */
    explicit TimeIntegratorData(const TimeIntegratorSchemes scheme)
      : integrator_type(scheme)
    {}


    TimeIntegratorSchemes integrator_type = TimeIntegratorSchemes::not_initialized;

    NonlinearSolverData<double> nlsolver_data;
    LinearSolverData<double>    linear_solver_data;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("time integration");
      {
        prm.add_parameter("type", integrator_type, "Name of the time integration scheme.");
        nlsolver_data.add_parameters(prm);
        linear_solver_data.add_parameters(prm);
      }
      prm.leave_subsection();
    }
  };

} // namespace MeltPoolDG