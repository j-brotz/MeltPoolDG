#pragma once

#ifdef MPDG_ENABLE_ADAFLO
#  include <adaflo/parameters.h>
#endif

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * @brief A struct providing the relevant wrapper data for coupling MeltPoolDG with Adaflo.
   */
  template <typename number>
  struct AdafloWrapperParameters
  {
    /**
     * @brief Default constructor.
     */
    AdafloWrapperParameters() = default;

#ifdef MPDG_ENABLE_ADAFLO
    /**
     * @brief Add Adaflo-wrapper-specific parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * @brief Finalizes and adjusts Adaflo parameters after reading user input.
     *
     * @param material The material data to derive density and viscosity from.
     * @param fe_type The finite element type, used to determine if a simplex mesh is used.
     * @param time_stepping Time integration parameters used to initialize Adaflo’s internal timing.
     */
    void
    post(const MaterialData<number>                       material,
         const FiniteElementType                         &fe_type,
         const TimeIntegration::TimeSteppingData<number> &time_stepping);

    /**
     * @brief Validates that Adaflo parameters are consistent with enabled physical models.
     *
     * Performs consistency checks on the input parameters, particularly in the case
     * where evaporative phase change is enabled. If phase change is considered,
     * the convective formulation of the momentum balance must be used.
     *
     * @param enable_evaporative_dilation_rate Flag indicating whether the evaporative dilation rate
     * (i.e., phase change) model is active.
     */
    void
    check_input_parameters(const bool enable_evaporative_dilation_rate) const;

    /**
     * @brief Returns a constant reference to the internal Adaflo flow parameters.
     *
     * @return const reference to the adaflo::FlowParameters object.
     */
    const adaflo::FlowParameters &
    get_parameters() const;

    /// Adaflo parameter struct
    adaflo::FlowParameters params;
#endif
  };
} // namespace MeltPoolDG::Flow
