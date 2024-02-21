#include <meltpooldg/radiative_transport/radiative_transport_data.hpp>


namespace MeltPoolDG::RadiativeTransport
{
  template <typename number>
  RadiativeTransportData<number>::RadiativeTransportData()
  {
    linear_solver.solver_type         = LinearSolverType::GMRES;
    linear_solver.preconditioner_type = PreconditionerType::ILU;
  }

  template <typename number>
  void
  RadiativeTransportData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("rte");
    {
      prm.add_parameter(
        "rte verbosity level",
        verbosity_level,
        "Sets the maximum verbosity level of the console output. The maximum level with respect to the "
        " base value is decisive.");
      prm.add_parameter("predictor type", predictor_type, "Choose a predictor type.");
      prm.add_parameter("absorptivity type",
                        absorptivity_type,
                        "Chooses the formulation of the absorptivity coefficient");
      prm.add_parameter("avoid singular matrix absorptivity",
                        avoid_singular_matrix_absorptivity,
                        "Minimum value for absorptivity to ensure a non-singular matrix for RTE.");
      linear_solver.add_parameters(prm);
      pseudo_time_stepping.add_parameters(prm);
      absorptivity_constant_data.add_parameters(prm);
      absorptivity_gradient_based_data.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  RadiativeTransportData<number>::AbsorptivityGradientBasedData::add_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("absorptivity");
    {
      prm.add_parameter("avoid div zero constant",
                        avoid_div_zero_constant,
                        "Sets the absorptivity of the gas phase.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  RadiativeTransportData<number>::AbsorptivityConstantData::add_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("absorptivity");
    {
      prm.add_parameter("absorptivity gas",
                        absorptivity_gas,
                        "Sets the absorptivity of the gas phase.");
      prm.add_parameter("absorptivity liquid",
                        absorptivity_liquid,
                        "Sets the absorptivity of the liquid phase.");
    }
    prm.leave_subsection();
  }

  template struct RadiativeTransportData<double>;
} // namespace MeltPoolDG::RadiativeTransport