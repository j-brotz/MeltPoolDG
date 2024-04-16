#pragma once

namespace MeltPoolDG
{
  // time integration schemes
  enum TimeIntegrators
  {
    stage_1_order_1,
    stage_2_order_2,
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
    implicit_Euler,
    explicit_Euler,
    crank_nicolson,
  };
} // namespace MeltPoolDG
