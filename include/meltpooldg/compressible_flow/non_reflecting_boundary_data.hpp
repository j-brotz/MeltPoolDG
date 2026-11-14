#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Parameters of the polynomial-correction Navier-Stokes characteristic boundary
   * conditions (PC-NSCBC), see Shehadi and van der Weide, Comput. Phys. Commun. 303 (2024)
   * 109194.
   *
   * The incoming wave amplitudes at a non-reflecting boundary are given by
   * \f[
   *   \mathcal{L}^+ = \sigma \mathcal{K} - \beta \mathcal{T}^+,
   * \f]
   * where \f$\mathcal{K}\f$ relaxes the boundary towards the prescribed target state on the time
   * scale \f$l / a\f$ and \f$\mathcal{T}^+\f$ contains the transverse terms. Setting \f$\sigma =
   * \beta = 0\f$ results in a perfectly non-reflecting boundary in the one-dimensional sense, which
   * is however not able to maintain the prescribed state. As in the reference, the transverse terms
   * are only considered at outflow boundaries.
   */
  template <typename number>
  struct NonReflectingBoundaryData
  {
    /// Reference length \f$l\f$ of the linear relaxation towards the prescribed state.
    number reference_length = 1.0;

    /// Normal relaxation coefficient \f$\sigma\f$ at outflow boundaries (static pressure outlet).
    /// The default value is the one recommended by Rudy and Strikwerda.
    number outflow_relaxation_coefficient = 0.28;

    /// Transverse relaxation coefficient \f$\beta\f$ at outflow boundaries.
    number outflow_transverse_relaxation_coefficient = 0.0;

    /// Normal relaxation coefficient \f$\sigma\f$ at inflow boundaries (static density and velocity
    /// inlet). The default value is the one of the reference.
    number inflow_relaxation_coefficient = 1.0;

    /**
     * @brief Add the parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("non reflecting boundary");
      {
        prm.add_parameter(
          "reference length",
          reference_length,
          "Reference length l of the linear relaxation of the polynomial-correction "
          "Navier-Stokes characteristic boundary conditions towards the prescribed state, "
          "which happens on the time scale l / a with the speed of sound a. The default "
          "corresponds to the reference, where l = 1 m is used for domains of a few tens of "
          "meters.");
        prm.add_parameter("outflow relaxation coefficient",
                          outflow_relaxation_coefficient,
                          "Normal relaxation coefficient sigma of the incoming acoustic wave at "
                          "non-reflecting outflow boundaries, which drives the pressure towards "
                          "the prescribed one. A value of zero gives a perfectly non-reflecting "
                          "outflow for one-dimensional waves, which however does not maintain the "
                          "prescribed pressure. Larger values increase the reflection of low "
                          "frequencies.");
        prm.add_parameter(
          "outflow transverse relaxation coefficient",
          outflow_transverse_relaxation_coefficient,
          "Transverse relaxation coefficient beta at non-reflecting outflow boundaries, which "
          "controls how much of the transverse terms is subtracted from the incoming wave "
          "amplitude. A value of zero neglects the transverse terms (NSCBC1D). Positive values "
          "reduce the reflection of waves which impinge on the boundary at a non-normal angle. "
          "Values close to one may become unstable, values up to 0.9 have been found stable.");
        prm.add_parameter(
          "inflow relaxation coefficient",
          inflow_relaxation_coefficient,
          "Normal relaxation coefficient sigma of the incoming waves at non-reflecting inflow "
          "boundaries, which drive the normal velocity, the density, the tangential velocity "
          "and the mass fractions towards the prescribed ones. Small values allow the inflow "
          "state to drift, in particular for waves which impinge on the inflow at a non-normal "
          "angle.");
      }
      prm.leave_subsection();
    }

    /**
     * @brief Check the parameters after reading user input.
     */
    void
    post() const
    {
      AssertThrow(reference_length > 0.,
                  dealii::ExcMessage("The reference length of the non-reflecting boundary "
                                     "conditions must be positive."));
      AssertThrow(outflow_relaxation_coefficient >= 0. and inflow_relaxation_coefficient >= 0.,
                  dealii::ExcMessage("The relaxation coefficients of the non-reflecting boundary "
                                     "conditions must not be negative."));
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
