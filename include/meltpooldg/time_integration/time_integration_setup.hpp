#pragma once
#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  //// time integration schemes
  BETTER_ENUM(TimeIntegrators,
              int,
              not_initialized,
              RK_stage_1_order_1,
              RK_stage_2_order_2,
              RK_stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
              RK_stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
              RK_stage_7_order_4, /* Tselios, Simos, 2007 */
              RK_stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
              implicit_euler,
              explicit_euler,
              crank_nicolson,
              bdf_2)
} // namespace MeltPoolDG
