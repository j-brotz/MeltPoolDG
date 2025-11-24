#include <meltpooldg/flow/adaflo_wrapper_wrapper.hpp>

/*
template <int dim, typename Number>
MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<dim, Number>::IncompressibleFlowSolverWrapper(
  const FlowData<IncompressibleFlowSolverParameters> &flow_data,
  dealii::Triangulation<dim>                         &triangulation,
  const dealii::Mapping<dim>                         &mapping)
  : navier_stokes(std::make_unique<adaflo::NavierStokes<dim>>(mapping,
                                                              flow_data.flow_parameters.parameters,
                                                              triangulation))
  , timer(timer)
{}


template <int dim, typename Number>
void
MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<dim, Number>::reinit()
{}


template <int dim, typename Number>
void
MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<dim, Number>::solve(
  const Number                  dt,
  [[maybe_unused]] const Number time)
{
  dealii::TimerOutput::Scope timer_section(timer, "Solver: Navier_Stokes_Solver");
  update_penalty_term();
  navier_stokes->get_constraints_u().set_zero(navier_stokes->user_rhs.block(0));
  navier_stokes->get_constraints_p().set_zero(navier_stokes->user_rhs.block(1));
  navier_stokes->time_stepping.set_time_step(dt);
  navier_stokes->advance_time_step();
}


template <int dim, typename Number>
void
IncompressibleFlowSolverWrapper<dim, Number>::update_penalty_term()
{}


template <int dim, typename Number>
void
MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<dim,
                                                  Number>::prepare_for_coarsening_and_refinement()
{
  this->navier_stokes->prepare_coarsening_and_refinement();
}


template <int dim, typename Number>
void
MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<dim, Number>::execute_coarsening_and_refinement()
{
  this->navier_stokes->distribute_dofs();
  this->navier_stokes->initialize_data_structures();
  this->navier_stokes->initialize_matrix_free();
}


template <int dim, typename number>
void
MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<dim, number>::attach_output_vectors(
  GenericDataOut<dim, number> &data_out) const
{
  // pressure
  std::vector<std::string> names = {"pressure"};
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation = {dealii::DataComponentInterpretation::component_is_scalar};
  output.add_to_data_out(names,
                         data_component_interpretation,
                         this->navier_stokes->get_dof_handler_p(),
                         this->navier_stokes->solution.block(1));

  // velocity
  names.clear();
  data_component_interpretation.clear();
  names.insert(names.end(), dim, "velocity");
  data_component_interpretation.insert(
    data_component_interpretation.end(),
    dim,
    dealii::DataComponentInterpretation::component_is_part_of_vector);
  output.add_to_data_out(names,
                         data_component_interpretation,
                         this->navier_stokes->get_dof_handler_u(),
                         this->navier_stokes->solution.block(0));

  // user right-hand side number
  if (output_data.flow_output.output_velocity_user_rhs)
    {
      names.clear();
      names.insert(names.end(), dim, "velocity_user_rhs");
      output.add_to_data_out(names,
                             data_component_interpretation,
                             this->navier_stokes->get_dof_handler_u(),
                             this->navier_stokes->user_rhs.block(0));
    }
}

template class MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<1, double>;
template class MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<2, double>;
template class MeltPoolDG::Flow::IncompressibleFlowSolverWrapper<3, double>;
*/