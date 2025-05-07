#include <meltpooldg/core/case_registration.hpp>

#include <cmath>
#include <vector>

#include "melt_front_propagation.templates.hpp"


/**
 * This simulation represents simple test examples for heat transfer with melt front propagation.
 * The problem is inspired by the Proell et al. [1] single track scan example.
 *
 * The slab has properties of Ti-6Al-4V, is initially below the solidus temperature and is subjected
 * to a Gusarov laser heat source [2] at x = 0.
 *
 * [1] Proell, S. D., Wall, W. A., & Meier, C. (2019). On phase change and latent heat models in
 * metal additive manufacturing process simulation. Advanced Modeling and Simulation in Engineering
 * Sciences, 7(1), 1-32. http://arxiv.org/abs/1906.06238
 *
 * [2] Gusarov, A. V., Yadroitsev, I., Bertrand, P., & Smurov, I. (2009). Model of Radiation and
 * Heat Transfer in Laser-Powder Interaction Zone at Selective Laser Melting. Journal of Heat
 * Transfer, 131(7), 1-10. https://doi.org/10.1115/1.3109245
 */

namespace MeltPoolDG::Simulation::MeltFrontPropagation
{

  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     1,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     2,
                                     double);
  MELTPOOLDG_REGISTER_MULTI_APP_CASE(Heat::HeatTransferCase,
                                     SimulationMeltFrontPropagation,
                                     "melt_front_propagation",
                                     3,
                                     double);

} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
