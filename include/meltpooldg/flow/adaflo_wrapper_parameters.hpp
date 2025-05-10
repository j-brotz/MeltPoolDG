#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <adaflo/parameters.h>
#endif

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  struct AdafloWrapperParameters
  {
    AdafloWrapperParameters() = default;

#ifdef MELT_POOL_DG_WITH_ADAFLO
    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const MaterialData<number>                       material,
         const FiniteElementType                         &fe_type,
         const TimeIntegration::TimeSteppingData<number> &time_stepping);

    void
    check_input_parameters(const bool enable_evaporative_dilation_rate) const;

    const adaflo::FlowParameters &
    get_parameters() const;

    adaflo::FlowParameters params;
#endif
  };
} // namespace MeltPoolDG::Flow
