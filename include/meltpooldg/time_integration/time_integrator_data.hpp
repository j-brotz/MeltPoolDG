#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <string>

namespace MeltPoolDG::TimeIntegration
{
  BETTER_ENUM(TimeIntegratorSchemes,
              int,
              not_initialized,
              RK_stage_1_order_1,
              RK_stage_2_order_2,
              RK_stage_3_order_3,
              RK_stage_4_order_4,
              LSRK_stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
              LSRK_stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
              LSRK_stage_7_order_4, /* Tselios, Simos, 2007 */
              LSRK_stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
              implicit_euler,
              explicit_euler,
              crank_nicolson,
              bdf_2)

  /**
   * Collection of all integrator parameters.
   */
  struct TimeIntegratorData
  {
    TimeIntegratorSchemes integrator_type = TimeIntegratorSchemes::LSRK_stage_5_order_4;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("time integration");
      {
        prm.add_parameter("type", integrator_type, "Name of the time integration scheme.");
      }
      prm.leave_subsection();
    }
  };

} // namespace MeltPoolDG::TimeIntegration