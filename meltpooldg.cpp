#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>

#include <meltpooldg/interface/problem_selector.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/revision.hpp>

#include <iostream>

#include "simulations/simulation_selector.hpp"

namespace MeltPoolDG
{
  namespace Simulation
  {
    template <typename number = double>
    void
    run_simulation(const std::string parameter_file, const MPI_Comm mpi_communicator)
    {
      Parameters<number> parameters;
      parameters.process_parameters_file(parameter_file);

      // print GIT hashes if verbosity level >= 1
      if (parameters.base.verbosity_level >= 1)
        {
          dealii::ConditionalOStream pcout(std::cout,
                                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

          Journal::print_decoration_line(pcout);
          pcout << "  - deal.II ("
                << "branch: " << DEAL_II_GIT_BRANCH << "; "
                << "revision: " << DEAL_II_GIT_REVISION << "; short: " << DEAL_II_GIT_SHORTREV
                << ")" << std::endl;
          pcout << "  - MeltPoolDG ("
                << "branch: " << LOCAL_GIT_BRANCH "; "
                << "revision: " << LOCAL_GIT_REVISION << "; short: " << LOCAL_GIT_SHORTREV << ")"
                << std::endl;
          Journal::print_decoration_line(pcout);
        }

      const auto dim = parameters.base.dimension;

      try
        {
          if (dim == 1)
            {
              auto sim = SimulationSelector<1>::get_simulation(parameters.base.application_name,
                                                               parameter_file,
                                                               mpi_communicator);
              sim->create();
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
                  parameters.base.do_print_parameters)
                parameters.print_parameters(std::cout, false /*print_details*/);
              auto problem = ProblemSelector<1>::get_problem(parameters.base.problem_name);
              problem->run(sim);
            }
          else if (dim == 2)
            {
              auto sim = SimulationSelector<2>::get_simulation(parameters.base.application_name,
                                                               parameter_file,
                                                               mpi_communicator);
              sim->create();
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
                  parameters.base.do_print_parameters)
                parameters.print_parameters(std::cout, false /*print_details*/);
              auto problem = ProblemSelector<2>::get_problem(parameters.base.problem_name);
              problem->run(sim);
            }
          else if (dim == 3)
            {
              auto sim = SimulationSelector<3>::get_simulation(parameters.base.application_name,
                                                               parameter_file,
                                                               mpi_communicator);
              sim->create();
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
                  parameters.base.do_print_parameters)
                parameters.print_parameters(std::cout, false /*print_details*/);
              auto problem = ProblemSelector<3>::get_problem(parameters.base.problem_name);
              problem->run(sim);
            }
          else
            {
              AssertThrow(false, ExcMessage("Dimension must be 1, 2 or 3."));
            }
        }
      catch (std::exception &exc)
        {
          std::cerr << std::endl
                    << std::endl
                    << "----------------------------------------------------" << std::endl;
          std::cerr << "Exception on processing: " << std::endl
                    << exc.what() << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------" << std::endl;
        }
      catch (...)
        {
          std::cerr << std::endl
                    << std::endl
                    << "----------------------------------------------------" << std::endl;
          std::cerr << "Unknown exception!" << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------" << std::endl;
        }
    }

  } // namespace Simulation
} // namespace MeltPoolDG

std::string
concatenate_strings(const int argc, char **argv)
{
  std::string result = std::string(argv[0]);

  for (int i = 0; i < argc; ++i)
    result = result + " " + std::string(argv[i]);

  return result;
}

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace MeltPoolDG;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;
  // check command line arguments
  if (argc == 1)
    {
      if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        std::cout << "ERROR: No .json parameter files has been provided!" << std::endl;
      return 1;
    }
  else if (argc == 2)
    {
      input_file = std::string(argv[argc - 1]);
      Simulation::run_simulation(input_file, mpi_comm);
    }
  else if (argc == 3 &&
           ((std::string(argv[1]) == "--help") || (std::string(argv[1]) == "--help-detail")))
    {
      input_file = std::string(argv[argc - 1]);

      Parameters<double> parameters;
      parameters.process_parameters_file(input_file);

      if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        parameters.print_parameters(std::cout,
                                    std::string(argv[1]) == "--help-detail" /*print_details*/);
      return 0;
    }
  else
    AssertThrow(false, ExcMessage("no input file specified"));

  return 0;
}
