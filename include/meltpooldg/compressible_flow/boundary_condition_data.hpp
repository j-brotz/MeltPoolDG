#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Data structure, which contains the coefficients of the boundary conditions of the
   * compressible flow solver that are shared by all boundaries of one type.
   *
   * The values that may vary along a boundary, e.g. the target pressure of an outflow, are not
   * part of this struct. They are given by the boundary functions of the simulation case.
   */
  template <typename number>
  struct BoundaryConditionData
  {
    /// Coefficients of the non-reflecting outflow, which are shared by all  non-reflecting outflow
    /// boundaries.
    struct NonReflectingOutflow
    {
      /// The reference length normal to the outlet (SI: m). It should be of the order of the domain
      /// size. The default 0 marks it as unset.
      number reference_length = 0.;

      /// The relaxation coefficient (dimensionless), which must be positive.
      number relaxation_coefficient = 0.28;

      /// The transverse coefficient (dimensionless), which must be positive.
      number transverse_coefficient = 0.;
    } non_reflecting_outflow;

    /**
     * @brief Add the parameters of the boundary conditions to the subsection "boundary
     * conditions" of the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("boundary conditions");
      {
        prm.enter_subsection("non reflecting outflow");
        {
          prm.add_parameter(
            "reference length",
            non_reflecting_outflow.reference_length,
            "Reference length l_x of the non-reflecting outflow normal to the outlet [m]. Only "
            "the ratio of the relaxation coefficient and the reference length enters the "
            "relaxation of the outlet pressure, so it should be of the order of the domain size. "
            "It must be positive if a non-reflecting outflow is used. The default 0 marks it as "
            "unset.");
          prm.add_parameter(
            "relaxation coefficient",
            non_reflecting_outflow.relaxation_coefficient,
            "Relaxation coefficient sigma of the non-reflecting outflow (dimensionless). With 0, "
            "the outlet is perfectly non-reflecting, but its mean pressure may drift. A larger "
            "value relaxes the outlet pressure faster towards the target pressure, but reflects "
            "more.",
            dealii::Patterns::Double(0.));
          prm.add_parameter(
            "transverse coefficient",
            non_reflecting_outflow.transverse_coefficient,
            "Transverse coefficient beta of the non-reflecting outflow (dimensionless). The "
            "default 0 neglects the transverse terms. A positive value reduces the reflection of obliquely incident waves.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
