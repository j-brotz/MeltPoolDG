#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/better_enum.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <vector>


namespace MeltPoolDG::HyperbolicPDETools
{
  /**
   * An enumeration of the available limiters.
   *
   * - `tvd_minmod`: Standard TVD (total variation diminishing) minmod limiter
   * - `tvb_minmod`: TVB (total variation bounded) minmod limiter, which relaxes the TVD criteria
   * near smooth extrema to avoid unnecessary clipping of physical peaks.
   */
  BETTER_ENUM(LimiterType, char, tvd_minmod, tvb_minmod)

  BETTER_ENUM(TroubledCellMarkingType,
              char,
              inter_cell_numerical_admissibility,
              local_cell_numerical_admissibility,
              physical_admissibility)

  /**
   * A struct to hold the data for the limiters.
   */
  template <typename number>
  struct LimiterData
  {
    /// Boolean flag indicating whether to apply the limiter or not.
    bool apply_limiter = false;

    /// The TVB constant used in the TVB minmod limiter.
    std::vector<number> tvb_constant;

    /// The types of troubled cell marking to use for the limiter.
    std::vector<TroubledCellMarkingType> troubled_cell_marking_types;

    /// The type of limiter to apply.
    LimiterType type = LimiterType::tvd_minmod;

    /**
     * Add the limiter parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the limiter parameters will be added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * Check the input parameters for validity.
     */
    void
    check_input_parameters() const;
  };
} // namespace MeltPoolDG::HyperbolicPDETools

template <typename number>
void
MeltPoolDG::HyperbolicPDETools::LimiterData<number>::add_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("limiter");
  {
    prm.add_parameter("apply", apply_limiter, "Whether to apply a limiter.");
    prm.add_parameter(
      "tvb constant",
      tvb_constant,
      "The TVB constant used in the TVB minmod limiter. The constant has the dimension of a second derivative.");
    prm.add_parameter("type", type, "The type of limiter to apply.");
    prm.add_parameter("troubled cell marking types",
                      troubled_cell_marking_types,
                      "The types of troubled cell marking to use for the limiter.");
  }
  prm.leave_subsection();
}

template <typename number>
void
MeltPoolDG::HyperbolicPDETools::LimiterData<number>::check_input_parameters() const
{
  AssertThrow(
    !apply_limiter or !troubled_cell_marking_types.empty(),
    dealii::ExcMessage(
      "If the limiter is applied, at least one troubled cell marking type must be specified."));
  AssertThrow(
    !(Utils::contains(troubled_cell_marking_types,
                      TroubledCellMarkingType::inter_cell_numerical_admissibility) and
      Utils::contains(troubled_cell_marking_types,
                      TroubledCellMarkingType::local_cell_numerical_admissibility)),
    dealii::ExcMessage(
      "Using both inter-cell and local cell numerical admissibility is not recommended as it adds computational complexity without providing significant benefits and hence it is not supported. Please choose one of the two marking types."));
}
