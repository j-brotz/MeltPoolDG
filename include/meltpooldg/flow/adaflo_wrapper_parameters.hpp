/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/

#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <adaflo/parameters.h>
#endif

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  namespace Flow
  {
    struct AdafloWrapperParameters
    {
      AdafloWrapperParameters() = default;

#ifdef MELT_POOL_DG_WITH_ADAFLO
      void
      parse_parameters(const std::string &parameter_filename);

      const FlowParameters &
      get_parameters() const;

      FlowParameters params;
#endif
    };
  } // namespace Flow
} // namespace MeltPoolDG
