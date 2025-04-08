#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <adaflo/parameters.h>
#endif

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG::Flow
{
  struct AdafloWrapperParameters
  {
    AdafloWrapperParameters() = default;

#ifdef MELT_POOL_DG_WITH_ADAFLO
    void
    parse_parameters(const std::string &parameter_filename);

    const adaflo::FlowParameters &
    get_parameters() const;

    adaflo::FlowParameters params;
#endif
  };
} // namespace MeltPoolDG::Flow
