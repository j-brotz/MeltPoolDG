#include <meltpooldg/radiative_transport/pseudo_rte_data.hpp>

#include <limits>


namespace MeltPoolDG::RadiativeTransport
{
  template <typename number>
  PseudoTimeSteppingData<number>::PseudoTimeSteppingData()
  {
    linear_solver.solver_type         = LinearSolverType::CG;
    linear_solver.preconditioner_type = PreconditionerType::ILU;
    linear_solver.monitor_type        = LinearSolverMonitorType::none;
    time_stepping_data.max_n_steps    = 1;
    time_stepping_data.time_step_size = 0.0;
    time_stepping_data.end_time       = std::numeric_limits<number>::max();
  }

  template <typename number>
  void
  PseudoTimeSteppingData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("pseudo time stepping");
    {
      prm.add_parameter("diffusion term scaling",
                        diffusion_term_scaling,
                        "Scaling parameter of diffusion term.");
      prm.add_parameter("advection term scaling",
                        advection_term_scaling,
                        "Scaling parameter of advection term.");
      prm.add_parameter(
        "pseudo time scaling",
        pseudo_time_scaling,
        "Determine the pseudo-time step as the product of this scaling and minimum cell size.");
      prm.add_parameter("rel tolerance", rel_tolerance, "Pseudo-time stepping relative tolerance.");

      time_stepping_data.add_parameters(prm);
      linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template struct PseudoTimeSteppingData<double>;
} // namespace MeltPoolDG::RadiativeTransport