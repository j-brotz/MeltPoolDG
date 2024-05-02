#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  AdvectionDiffusionData<number>::AdvectionDiffusionData()
  {
    linear_solver.solver_type         = LinearSolverType::GMRES;
    linear_solver.preconditioner_type = PreconditionerType::Diagonal;
  }
  template <typename number>
  void
  AdvectionDiffusionData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("advection diffusion");
    {
      fe.add_parameters(prm);

      prm.enter_subsection("convection stabilization");
      {
        prm.add_parameter("type", conv_stab.type, "Defines the type for convection stabilization.");

        prm.add_parameter(
          "coefficient",
          conv_stab.coefficient,
          "Defines the stabilization coefficient for convection. (default velocity-dependent).");
      }
      prm.leave_subsection();
      prm.add_parameter("diffusivity",
                        diffusivity,
                        "Defines the diffusivity for the advection diffusion equation ");
      prm.add_parameter(
        "time integration scheme",
        time_integration_scheme,
        "Determines the time integration scheme.",
        Patterns::Selection(
          "explicit_euler|implicit_euler|crank_nicolson|bdf_2|RK_stage_1_order_1|RK_stage_2_order_2|RK_stage_3_order_3|RK_stage_5_order_4|RK_stage_7_order_4|RK_stage_9_order_5"));
      prm.add_parameter(
        "implementation",
        implementation,
        "Choose the corresponding implementation of the advection diffusion operation.",
        Patterns::Selection("meltpooldg|adaflo"));
      prm.add_parameter("enable time dependent bc",
                        enable_time_dependent_bc,
                        "Set this parameter to true to enable time-dependent bc.");
      predictor.add_parameters(prm);
      linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  AdvectionDiffusionData<number>::post(const FiniteElementData &base_fe_data)
  {
    predictor.post();
    fe.post(base_fe_data);
  }

  template <typename number>
  void
  AdvectionDiffusionData<number>::check_input_parameters() const
  {
    AssertThrow(linear_solver.do_matrix_free || implementation == "meltpooldg",
                ExcNotImplemented());

    AssertThrow(!enable_time_dependent_bc || implementation == "meltpooldg", ExcNotImplemented());

    AssertThrow((conv_stab.type == ConvectionStabilizationType::SUPG &&
                 linear_solver.do_matrix_free && implementation == "meltpooldg") ||
                  conv_stab.type == ConvectionStabilizationType::none,
                ExcNotImplemented());
    AssertThrow(
      predictor.type != PredictorType::least_squares_projection || linear_solver.do_matrix_free,
      ExcMessage(
        "For matrix-based advection-diffusion solver least squares projection is not supported."));

    AssertThrow(diffusivity >= 0.0,
                ExcMessage("Advection diffusion operator: diffusivity is smaller than zero!"));

    // Cross check for DG since for DG only matrix free is supported
    if ((fe.type == FiniteElementType::FE_DGQ) && (!linear_solver.do_matrix_free))
      {
        AssertThrow(false, ExcMessage("For the DG element only matrix free is implemented."));
      }
  }

  template struct AdvectionDiffusionData<double>;
} // namespace MeltPoolDG::LevelSet
