#pragma once
#include <meltpooldg/interface/simulation_base.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  class ProblemBase
  {
  public:
    virtual ~ProblemBase()
    {}

    virtual void
    run(std::shared_ptr<SimulationBase<dim>> base_in) = 0;

    virtual std::string
    get_name() = 0;

    virtual void
    perform_convergence_study(){};

  protected:
    void
    add_problem_specific_parameters(const std::string &parameter_file)
    {
      /*
       * Read user-defined parameters.
       */
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
