#pragma once

#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/simulation_base.hpp>

#include "parameters_base.hpp"

namespace MeltPoolDG
{
  using namespace dealii;

  // TODO: how can this template type be omited?
  template <int dim, typename SimulationType = MeltPoolCase<dim>>
  class ProblemBase
  {
  public:
    virtual ~ProblemBase()
    {}

    virtual void
    run(std::shared_ptr<SimulationType> base_in) = 0;

  protected:
    // TODO: Move into free function
    void
    add_problem_specific_parameters(const std::string &parameter_file)
    {
      ParameterHandler prm_problem_specific;
      add_parameters(prm_problem_specific);

      std::ifstream file;
      file.open(parameter_file);

      if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "json")
        prm_problem_specific.parse_input_from_json(file, true);
      else if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "prm")
        prm_problem_specific.parse_input(parameter_file);
      else
        AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file ending"));

      add_and_parse_parameters(
        parameter_file, [this](ParameterHandler &prm) { add_parameters(prm); }, false);
    }

    /**
     * Add parameters to the parameter handler.
     *
     * This function is intended to be overriden by derived classes.
     * If so, add_problem_specific_parameters() must be called from
     * the derived class.
     */
    virtual void
    add_parameters(dealii::ParameterHandler &)
    {
      AssertThrow(false,
                  ExcMessage(
                    "If problem specific parameters should be added, "
                    "add_parameters() has to be overriden in the derived class. Abort..."));
    }
  };
} // namespace MeltPoolDG
