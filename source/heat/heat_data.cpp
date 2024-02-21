#include <deal.II/base/exceptions.h>

#include <meltpooldg/heat/heat_data.hpp>

namespace MeltPoolDG::Heat
{
  template <typename number>
  HeatData<number>::HeatData()
  {
    linear_solver.solver_type         = LinearSolverType::GMRES;
    linear_solver.preconditioner_type = PreconditionerType::DiagonalReduced;
  }

  template <typename number>
  void
  HeatData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("heat");
    {
      prm.add_parameter("degree", degree, "Defines the interpolation degree");
      prm.add_parameter("n q points 1d", n_q_points_1d, "Defines the number of quadrature points");
      prm.add_parameter(
        "n subdivisions",
        n_subdivisions,
        "Set the number of subdivisions for the finite element of the level set operation.");
      prm.add_parameter("heat convection coefficient",
                        convection_coefficient,
                        "Convection coefficient for the radiative boundary condition");
      prm.add_parameter("heat emissivity",
                        emissivity,
                        "Emissivity for the radiative boundary condition");
      prm.add_parameter("heat temperature infinity",
                        temperature_infinity,
                        "Infinity temperature for the conductive and radiative boundary condition");

      // TODO: use heat.nlsolve.add_parameters() routine
      prm.add_parameter("heat nlsolve max nonlinear iterations",
                        nlsolve.max_nonlinear_iterations,
                        "Set the number of maximum nonlinear iterations with standard tolerances.");
      prm.add_parameter(
        "heat nlsolve field correction tolerance",
        nlsolve.field_correction_tolerance,
        "Set the tolerance for the maximum allowed correction of the unknown field.");
      prm.add_parameter(
        "heat nlsolve residual tolerance",
        nlsolve.residual_tolerance,
        "Set the tolerance for the maximum allowed residual of the nonlinear system.");
      prm.add_parameter(
        "heat nlsolve max nonlinear iterations alt",
        nlsolve.max_nonlinear_iterations_alt,
        "Set the number of maximum nonlinear iterations with alternative tolerances.");
      prm.add_parameter(
        "heat nlsolve field correction tolerance alt",
        nlsolve.field_correction_tolerance_alt,
        "Set the alternative tolerance for the maximum allowed correction of the unknown field.");
      prm.add_parameter(
        "heat nlsolve residual tolerance alt",
        nlsolve.residual_tolerance_alt,
        "Set the alternative tolerance for the maximum allowed residual of the nonlinear system.");

      // override default value
      predictor.type = PredictorType::linear_extrapolation;
      predictor.add_parameters(prm);
      prm.add_parameter("enable time dependent bc",
                        enable_time_dependent_bc,
                        "Set this parameter to true to enable time-dependent bc.");
      prm.add_parameter(
        "use volume-specific thermal capacity for phase interpolation",
        use_volume_specific_thermal_capacity_for_phase_interpolation,
        "Perform phase interpolation via the volumetric thermal capacity (product of density "
        " and capacity) instead of interpolating density and thermal capacity individually.");
      // add deprecated status
      prm.declare_alias("use volume-specific thermal capacity for phase interpolation",
                        "interpolate rho times cp",
                        true);
      delta_approximation_phase_weighted.add_parameters(prm);
      linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  HeatData<number>::post(const unsigned int          base_degree,
                         const unsigned int          base_verbosity_level,
                         const MaterialData<number> &material)
  {
    // TODO check in interpolate rho times cp

    // set automatic weights of asymmetric delta functions, if requested
    delta_approximation_phase_weighted.set_parameters(material);

    // set heat degree equal to base degree if it is not set
    if (degree < 1)
      degree = base_degree;

    if (n_q_points_1d < 1)
      n_q_points_1d = degree + 1;

    // sync verbosity level with base verbosity if not set
    if (nlsolve.verbosity_level == -1)
      nlsolve.verbosity_level = base_verbosity_level;
  }

  template <typename number>
  void
  HeatData<number>::check_input_parameters(const bool base_do_simplex,
                                           const int  ls_n_subdivisions) const
  {
    AssertThrow(
      (!base_do_simplex || (ls_n_subdivisions == 1 && n_subdivisions == 1)),
      ExcMessage(
        "If you use a simplex mesh, n_subdivisions for the level set and the heat equation must be 1."));
    AssertThrow((n_subdivisions == 1 || degree == 1),
                ExcMessage("If you use n_subdivisions for the heat equation, degree must be 1."));
  }

  template struct HeatData<double>;
} // namespace MeltPoolDG::Heat