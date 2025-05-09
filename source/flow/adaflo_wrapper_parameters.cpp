#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>

namespace MeltPoolDG::Flow
{
#ifdef MELT_POOL_DG_WITH_ADAFLO
  template <typename number>
  void
  AdafloWrapperParameters<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    // declare parameters
    prm.enter_subsection("flow");
    prm.enter_subsection("adaflo");
    params.add_parameters(prm);
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <typename number>
  void
  AdafloWrapperParameters<number>::post(
    const MaterialData<number>                       material,
    const FiniteElementType                         &fe_type,
    const TimeIntegration::TimeSteppingData<number> &time_stepping)
  {
    params.post();

    // WARNING: by setting the differences to a non-zero value we force
    //   adaflo to assume that we are running a simulation with variable
    //   coefficients, i.e., it allocates memory for the data structures
    //   variable_densities and variable_viscosities, which are accessed
    //   during NavierStokesMatrix::begin_densities() and
    //   NavierStokesMatrix::begin_viscosity(). However, we do not actually
    //   use these values, since we fill the density and viscosity
    //   differently.
    params.density_diff   = 1.0;
    params.viscosity_diff = 1.0;

    AssertThrow(params.density == 1.0, // 1.0 is the default value from adaflo
                dealii::ExcMessage(
                  "It seems that you specified the density parameter "
                  "within the adaflo section, which is ignored by MeltPoolDG. "
                  "Please use the > material: liquid: density: < section instead. "));

    AssertThrow(params.viscosity == 1.0, // 1.0 is the default value from adaflo
                dealii::ExcMessage(
                  "It seems that you specified the viscosity parameter "
                  "within the adaflo section, which is ignored by MeltPoolDG. "
                  "Please use the > material: liquid: viscosity: < section instead. "));

    if (material.gas.density > 0.0)
      {
        // adaflo assumes the parameter density to be the one of heaviside == 0
        params.density = material.gas.density;
      }
    if (material.gas.dynamic_viscosity > 0.0)
      {
        // adaflo assumes the parameter viscosity to be the one of heaviside == 0
        params.viscosity = material.gas.dynamic_viscosity;
      }

    /// synchronize time stepping schemes
    params.start_time           = time_stepping.start_time;
    params.end_time             = time_stepping.end_time;
    params.time_step_size_start = time_stepping.time_step_size;
    params.time_step_size_min   = 1e-16;
    params.time_step_size_max   = 1e10;

    params.use_simplex_mesh = fe_type == FiniteElementType::FE_SimplexP;
  }

  template <typename number>
  void
  AdafloWrapperParameters<number>::check_input_parameters(
    const bool enable_evaporative_dilation_rate) const
  {
    if (enable_evaporative_dilation_rate)
      {
        AssertThrow(
          params.beta_convective_term_momentum_balance == 0,
          dealii::ExcMessage(
            "For the consideration of phase change, the convective "
            "formulation of the momentum balance in the Navier-Stokes equations "
            "must be chosen: Navier-Stokes: adaflo: Navier-Stokes: {formulation convective "
            "term momentum balance: convective }"));
        //@todo: the following is kept as back-up

        // AssertThrow(material.two_phase_properties_transition_type ==
        // TwoPhasePropertiesTransitionType::consistent_with_evaporation,
        // ExcMessage(
        //"For the consideration of phase change, the density "
        //"has to be interpolated consistently with the continuity equation "
        //"including phase change."));
      }
  }

  template <typename number>
  const adaflo::FlowParameters &
  AdafloWrapperParameters<number>::get_parameters() const
  {
    return this->params;
  }
#endif
  template struct AdafloWrapperParameters<double>;

} // namespace MeltPoolDG::Flow
