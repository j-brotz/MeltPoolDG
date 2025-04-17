#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/problem_selector.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/revision.hpp>

#include <iostream>
#include <string>

#include "simulations/simulation_selector.hpp"

namespace MeltPoolDG
{
  namespace Simulation
  {
    template <typename number = double>
    void
    run_simulation(const std::string parameter_file, const MPI_Comm mpi_communicator)
    {
      using namespace dealii;
      unsigned int dim          = 0;
      std::string  problem_type = "not_initialized";
      std::string  case_name    = "not_initialized";

      {
        // The parameters will be read here to get information on the selection variables, i.e.,
        // the dimension, the case_name and the problem_name.
        ParameterHandler   prm;
        Parameters<number> parameters;
        parameters.process_parameters_file(prm, parameter_file);

        // print number of processes and GIT hashes if verbosity level >= 3
        if (parameters.base.verbosity_level >= 3)
          {
            dealii::ConditionalOStream pcout(std::cout,
                                             Utilities::MPI::this_mpi_process(mpi_communicator) ==
                                               0);
            Journal::print_decoration_line(pcout);
            Journal::print_line(
              pcout,
              "Running MeltPoolDG on " +
                std::to_string(Utilities::MPI::n_mpi_processes(mpi_communicator)) + " ranks.");

            Journal::print_decoration_line(pcout);
            pcout << "  - deal.II:" << std::endl
                  << "      * branch: " << DEAL_II_GIT_BRANCH << std::endl
                  << "      * revision: " << DEAL_II_GIT_REVISION << std::endl
                  << "      * short: " << DEAL_II_GIT_SHORTREV << std::endl;
            pcout << "  - MeltPoolDG:" << std::endl
                  << "      * branch: " << LOCAL_GIT_BRANCH << std::endl
                  << "      * revision: " << LOCAL_GIT_REVISION << std::endl
                  << "      * short: " << LOCAL_GIT_SHORTREV << std::endl;
            Journal::print_decoration_line(pcout);
          }

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
            parameters.base.do_print_parameters)
          {
            parameters.print_parameters(prm, std::cout, false /*print_details*/);
          }

        dim          = parameters.base.dimension;
        problem_type = parameters.base.problem_name;
        case_name    = parameters.base.case_name;
      }

      try
        {
          if (dim == 1)
            {
              auto sim = SimulationSelector<1, number>::get_simulation(case_name,
                                                                       parameter_file,
                                                                       mpi_communicator);
              sim->create();
              auto problem = ProblemSelector<1, number>::get_problem(problem_type, std::move(sim));
              problem->run();
            }
          else if (dim == 2)
            {
              auto sim = SimulationSelector<2, number>::get_simulation(case_name,
                                                                       parameter_file,
                                                                       mpi_communicator);
              sim->create();
              auto problem = ProblemSelector<2, number>::get_problem(problem_type, std::move(sim));
              problem->run();
            }
          else if (dim == 3)
            {
              auto sim = SimulationSelector<3, number>::get_simulation(case_name,
                                                                       parameter_file,
                                                                       mpi_communicator);
              sim->create();
              auto problem = ProblemSelector<3, number>::get_problem(problem_type, std::move(sim));
              problem->run();
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

      ParameterHandler   prm;
      Parameters<double> parameters;
      parameters.process_parameters_file(prm, input_file);

      if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        parameters.print_parameters(prm,
                                    std::cout,
                                    std::string(argv[1]) == "--help-detail" /*print_details*/);

      return 0;
    }
  else
    AssertThrow(false, ExcMessage("no input file specified"));

  return 0;
}
