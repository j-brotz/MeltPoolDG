#include <deal.II/base/exceptions.h>

#include <meltpooldg/heat/heat_data.hpp>

namespace MeltPoolDG::Heat
{
  template <typename number>
  HeatData<number>::HeatData()
  {
    linear_solver.solver_type         = LinearSolverType::GMRES;
    linear_solver.preconditioner_type = PreconditionerType::DiagonalReduced;
    predictor.type                    = PredictorType::linear_extrapolation;
  }

  template <typename number>
  void
  HeatData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("heat");
    {
      fe.add_parameters(prm);

      prm.add_parameter("enable time dependent bc",
                        enable_time_dependent_bc,
                        "Set this parameter to true to enable time-dependent bc.");
      prm.add_parameter(
        "use volume-specific thermal capacity for phase interpolation",
        use_volume_specific_thermal_capacity_for_phase_interpolation,
        "Perform phase interpolation via the volumetric thermal capacity (product of density "
        " and capacity) instead of interpolating density and thermal capacity individually.");

      prm.enter_subsection("radiative boundary condition");
      {
        prm.add_parameter("emissivity", radiation.emissivity, "Emissivity.");
        prm.add_parameter("temperature infinity",
                          radiation.temperature_infinity,
                          "Infinity temperature.");
      }
      prm.leave_subsection();

      prm.enter_subsection("convective boundary condition");
      {
        prm.add_parameter("convection coefficient",
                          convection.convection_coefficient,
                          "Convection coefficient.");
        prm.add_parameter("temperature infinity",
                          convection.temperature_infinity,
                          "Infinity temperature.");
      }
      prm.leave_subsection();

      nlsolve.add_parameters(prm);
      linear_solver.add_parameters(prm);
      predictor.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  HeatData<number>::post(const FiniteElementData &base_fe_data,
                         const unsigned int       base_verbosity_level)
  {
    fe.post(base_fe_data);

    // sync verbosity level with base verbosity if not set
    if (nlsolve.verbosity_level == -1)
      nlsolve.verbosity_level = base_verbosity_level;

    predictor.post();
  }

  template <typename number>
  void
  HeatData<number>::check_input_parameters(const FiniteElementData &base_fe_data) const
  {
    fe.check_input_parameters(base_fe_data);
  }

  template struct HeatData<double>;
} // namespace MeltPoolDG::Heat
