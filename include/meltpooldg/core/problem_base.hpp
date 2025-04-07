#pragma once

#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/simulation_base.hpp>

#include "parameters_base.hpp"

namespace MeltPoolDG
{
  // TODO: how can this template type be omited?
  template <int dim, typename number, typename SimulationType = MeltPoolCase<dim, number>>
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
      add_and_parse_parameters(
        parameter_file, [this](dealii::ParameterHandler &prm) { add_parameters(prm); }, false);
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
                  dealii::ExcMessage(
                    "If problem specific parameters should be added, "
                    "add_parameters() has to be overriden in the derived class. Abort..."));
    }
  };
} // namespace MeltPoolDG
