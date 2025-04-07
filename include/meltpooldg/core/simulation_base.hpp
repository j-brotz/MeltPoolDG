#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/case_factory.hpp>
#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/revision.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "parameters_base.hpp"

namespace MeltPoolDG
{
  /**
   * @brief Base class for managing a simulation case in a parallel computing environment.
   *
   * This class handles the setup and management of boundary conditions, field functions, and
   * spatial discretization in a parallel environment using MPI. It is intended to be extended by
   * more specific simulation classes.
   *
   * @tparam dim Spatial dimension.
   * @tparam P Parameter object type.
   * @tparam spacedim Space dimension.
   */
  template <int dim, typename number, int spacedim = dim>
  class SimulationCaseBase
  {
  public:
    // Triangulation object. Needs to be filled inside create_spatial_discretization()
    std::shared_ptr<dealii::Triangulation<dim, spacedim>> triangulation;

    // Path to the parameter file
    const std::string parameter_file;

    // MPI communicator for parallel execution
    // TODO: make private
    const MPI_Comm mpi_communicator;

    SimulationCaseBase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : parameter_file(parameter_file_in)
      , mpi_communicator(mpi_communicator_in)
    {}

    /**
     * @brief Main setup function to initialize the simulation.
     *
     * This function should be called after the constructor is called and the
     * parameters are parsed. Then, the spatial discretization,
     * boundary conditions, and field functions are created, which may depend
     * on the input parameters.
     */
    void
    create()
    {
      parse_simulation_specific_parameters();
      create_spatial_discretization();
      AssertThrow(
        this->triangulation,
        dealii::ExcMessage(
          "It seems that your SimulationCaseBase object does not contain"
          " a valid triangulation object. A shared_ptr to your triangulation"
          " must be specified as follows for a serialized triangulation "
          " this->triangulation = std::make_shared<Triangulation<dim>>(); "
          " or for a parallel triangulation "
          " this->triangulation = std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator); "));
      set_boundary_conditions();
      set_field_conditions();
    }

    /**
     * @brief Retrieve boundary conditions of a specific type for an operation.
     *
     * @param type The type of the boundary condition to retrieve.
     * @param operation_name The name of the operation whose boundary conditions are requested.
     * @param is_optional    A flag indicating whether the bc is optional.
     *                       Defaults to `true`, requiring the bc not to be present.
     * @return A map of boundary IDs to their corresponding boundary functions.
     *
     * @throws dealii::ExcMessage If no boundary conditions are found for the specified operation.
     */
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    get_boundary_condition(const std::string &type,
                           const std::string &operation_name,
                           const bool         is_optional = true) const
    {
      AssertThrow(is_optional || boundary_conditions_map.contains(operation_name),
                  dealii::ExcMessage(
                    "BC for " + operation_name +
                    " not found. "
                    "Did you forget to register the operation via "
                    "attach_boundary_condition({id, function} ,operation_name, function)?"));

      if (!boundary_conditions_map.contains(operation_name))
        return std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>();
      else
        return boundary_conditions_map.at(operation_name)->get_bc_of_type(type, is_optional);
    }

    /**
     * @brief Retrieve the boundary condition object for a specific operation.
     *
     * @param operation_name The name of the operation whose boundary condition object is requested.
     * @return A shared pointer to the BoundaryConditions object for the operation.
     *
     * @throws dealii::ExcMessage If no boundary conditions are found for the specified operation.
     */
    std::shared_ptr<BoundaryConditionManager<dim, number>>
    get_boundary_condition_manager(const std::string &operation_name,
                                   const bool         is_optional = true) const
    {
      AssertThrow(is_optional || boundary_conditions_map.contains(operation_name),
                  dealii::ExcMessage(
                    "BC for " + operation_name +
                    " not found. "
                    "Did you forget to register the operation via "
                    "attach_boundary_condition({id, function} ,operation_name, function)?"));

      if (!boundary_conditions_map.contains(operation_name))
        return std::make_shared<BoundaryConditionManager<dim, number>>();
      else
        return boundary_conditions_map.at(operation_name);
    }

    /**
     * @brief Retrieve the type of a boundary condition for a specific boundary ID and operation.
     *
     * @param boundary_id The boundary ID for which the type is requested.
     * @param operation_name The name of the operation associated with the boundary condition.
     * @return The type of the boundary condition as a string.
     *
     * @throws dealii::ExcMessage If no boundary conditions are found for the specified operation.
     */
    std::string
    get_boundary_condition_type(dealii::types::boundary_id boundary_id,
                                const std::string         &operation_name) const
    {
      AssertThrow(!boundary_conditions_map[operation_name].empty(),
                  dealii::ExcMessage(
                    "BC for " + operation_name +
                    " not found. "
                    "Did you forget to register the operation via "
                    "attach_boundary_condition({id, function} ,operation_name, function)?"));

      return boundary_conditions_map.at(operation_name)->get_type(boundary_id);
    }

    /**
     * @brief Update the time for all boundary conditions.
     *
     * Sets the current time for all boundary conditions managed by this class.
     *
     * @param time The current simulation time.
     */
    void
    set_time_boundary_conditions(const number time)
    {
      for (const auto &[operation, bc] : boundary_conditions_map)
        bc->set_time(time);
    }

    /**
     * Getter functions for field function of @p type for @p operation_name.
     * If the field function for the specified @p type and @p operation_name
     * cannot be found, behavior depends on the value of @p is_optional:
     * - If @p is_optional is set to `true`, the function will return a `nullptr`
     *   to indicate that the field function is not required and can be omitted.
     * - If @p is_optional is `false`, an exception may be thrown
     *   to indicate that the requested field function is missing.
     *
     * @param type           The category of the field function (e.g., "temperature", "velocity").
     * @param operation_name The name of the operation or context where the field function is used
     *                       (e.g., "initial_condition", "boundary_condition").
     * @param is_optional    A flag indicating whether the field function is optional.
     *                       Defaults to `false`, requiring the field function to be present.
     *
     */
    std::shared_ptr<dealii::Function<dim>>
    get_field_function(const std::string &type,
                       const std::string &operation_name,
                       const bool         is_optional = false)
    {
      auto field_conditions = field_functions[operation_name];

      AssertThrow(is_optional || (!field_conditions.empty() && field_conditions[type]),
                  ExcFieldNotAttached(type, operation_name));
      if (!field_conditions.empty() && field_conditions[type])
        return field_conditions.at(type);
      else // is_optional = true
        return nullptr;
    }
    /**
     * @brief Retrieve the initial condition for a given operation.
     *
     * @param operation_name The operation name for which the initial condition is requested.
     * @param is_optional Flag indicating if the initial condition is optional.
     * @return A shared pointer to the initial condition function.
     */
    std::shared_ptr<dealii::Function<dim>>
    get_initial_condition(const std::string &operation_name, const bool is_optional = false)
    {
      return get_field_function("initial_condition", operation_name, is_optional);
    }

    /**
     * @brief Get the periodic boundary condition manager.
     *
     * @return The periodic boundary condition manager.
     */
    const PeriodicBoundaryConditions<dim> &
    get_periodic_bc() const
    {
      return periodic_boundary_conditions;
    }

    /**
     * @brief Perform specific postprocessing (can be overridden by derived classes).
     *
     * @note This function needs to be called within the specific problem.
     *
     * @param generic_data_out The postprocessing data.
     */
    virtual void
    do_postprocessing([[maybe_unused]] const GenericDataOut<dim, number> &generic_data_out) const
    {
      // do nothing default
    }

  protected:
    // Methods to be implemented in derived classes

    /**
     * @brief Pure virtual function to create the spatial discretization.
     *
     * Derived classes must implement this function to define how the spatial discretization is
     * created.
     */
    virtual void
    create_spatial_discretization() = 0;

    /**
     * @brief Pure virtual function to set the boundary conditions.
     *
     * Derived classes must implement this function to define how boundary conditions are set.
     */
    virtual void
    set_boundary_conditions() = 0;

    /**
     * @brief Pure virtual function to set the field conditions.
     *
     * Derived classes must implement this function to define how field conditions are set.
     */
    virtual void
    set_field_conditions() = 0;

    /**
     * @brief Add simulation-specific parameters (can be overridden).
     *
     * This function can be overridden in derived classes to add custom simulation parameters to the
     * handler. The return value can be used to print the parameters to the
     * terminal output.
     *
     * @param prm ParameterHandler object to which parameters can be added.
     */
    virtual bool
    add_simulation_specific_parameters(dealii::ParameterHandler &)
    {
      return false;
      // default: do nothing
    }

    /**
     * Attaches a field function with a specified type and operation name.
     *
     * This function stores a provided field function in an internal dictionary,
     * allowing it to be later retrieved by a combination of its @p type and
     * @p operation_name. The @p type typically represents the category of the
     * field function (e.g., "prescribed_velocity"), while
     * @p operation_name indicates the specific operation or context to which the
     * function applies (e.g., "heat" or "level_set").
     *
     * By using this function, various field functions can be organized and managed
     * for different types and operations, enabling efficient lookup and reuse.
     *
     * @tparam FunctionType The type of the field function being attached, typically
     *                      a derived class of `dealii::Function<dim>`.
     * @param function      A `std::shared_ptr` to the field function object that
     *                      should be associated with the specified @p type and
     *                      @p operation_name.
     * @param type          A string representing the field function category
     *                      (e.g., "prescribed_velocity").
     * @param operation_name A string specifying the operation or context where
     *                       this field function is applicable (e.g., "level_set").
     */
    void
    attach_field_function(std::shared_ptr<dealii::Function<dim>> function,
                          const std::string                     &type,
                          const std::string                     &operation_name)
    {
      field_functions[operation_name][type] = function;
    }

    /**
     * Overload function to attach initial conditions. See documentation of
     * attach_field_function.
     */
    void
    attach_initial_condition(std::shared_ptr<dealii::Function<dim>> initial_function,
                             const std::string                     &operation_name)
    {
      attach_field_function(initial_function, "initial_condition", operation_name);
    }

    /**
     * @brief Attach a boundary condition for a specific operation.
     *
     * This method associates a boundary condition, specified by its ID and function,
     * with a given operation name. If the operation has no associated boundary conditions
     * yet, a new boundary condition object is created.
     *
     * @param id_and_function A pair consisting of a boundary ID and a shared pointer to the boundary function.
     * @param type The type of the boundary condition (e.g., Dirichlet, Neumann, etc.).
     * @param operation_name The name of the operation for which the boundary condition is being attached.
     */
    void
    attach_boundary_condition(
      std::pair<const dealii::types::boundary_id, const std::shared_ptr<dealii::Function<dim>>>
                         id_and_function,
      const std::string &type,
      const std::string &operation_name)
    {
      // create boundary conditions object for a given operation_name if it has not yet been created
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] =
          std::make_shared<BoundaryConditionManager<dim, number>>();

      boundary_conditions_map[operation_name]->attach_boundary_condition(id_and_function, type);
    }

    /**
     * @brief Attach a boundary condition with a dummy function for a specific operation.
     *
     * This overload simplifies attaching a boundary condition by omitting the function.
     * A placeholder function is used internally.
     *
     * @param id The boundary ID to be associated with the condition.
     * @param type The type of the boundary condition.
     * @param operation_name The name of the operation for which the boundary condition is being attached.
     */
    void
    attach_boundary_condition(const dealii::types::boundary_id id,
                              const std::string               &type,
                              const std::string               &operation_name)
    {
      attach_boundary_condition({id, nullptr /*dummy_function*/}, type, operation_name);
    }

    /**
     * Attach periodic boundary condition.
     *
     * @param direction refers to the space direction in which periodicity is enforced. When
     * matching periodic faces this vector component is ignored.
     */
    void
    attach_periodic_boundary_condition(const dealii::types::boundary_id id_in,
                                       const dealii::types::boundary_id id_out,
                                       const int                        direction)
    {
      AssertThrow(this->triangulation,
                  dealii::ExcMessage("You try to pass periodic faces but the triangulation "
                                     "is still empty."));

      periodic_boundary_conditions.attach_boundary_condition(id_in, id_out, direction);

      // distribute periodic bc to the triangulation
      std::vector<
        dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>>
        periodic_faces;

      dealii::GridTools::collect_periodic_faces(
        *triangulation, id_in, id_out, direction, periodic_faces);
      triangulation->add_periodicity(periodic_faces);
    }

  private:
    // Manager for periodic boundary conditions
    PeriodicBoundaryConditions<dim> periodic_boundary_conditions;

    // nested dictionary for storing field functions as
    // field_functions[<operation_name>][<type>] = <function>.
    // The <operation_name> and <type> are specified in the respective problems.
    std::map<std::string, std::map<std::string, std::shared_ptr<dealii::Function<dim>>>>
      field_functions;

    // This map organizes boundary conditions hierarchically based on operation names.
    std::map<std::string, std::shared_ptr<BoundaryConditionManager<dim, number>>>
      boundary_conditions_map;

    /**
     * @brief Parses simulation-specific parameters.
     */
    void
    parse_simulation_specific_parameters()
    {
      bool enable_print = false;
      add_and_parse_parameters(
        parameter_file,
        [this, &enable_print](dealii::ParameterHandler &prm) {
          enable_print = add_simulation_specific_parameters(prm);
        },
        enable_print && (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0));
    }
  };

  // TODO: move somewhere else
  template <int dim, typename number>
  class MeltPoolCase : public SimulationCaseBase<dim, number>
  {
    // Simulation parameters object.
  public:
    Parameters<number> parameters;

    MeltPoolCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }
  };


  template <typename ParametersType,
            template <int, typename>
            class CaseType,
            template <int, typename>
            class ProblemType>
  void
  run_simulation(const std::string &parameter_file, const MPI_Comm mpi_communicator)
  {
    unsigned int dim = 0;
    std::string  number;
    std::string  case_name;

    // Read and process parameters
    {
      dealii::ParameterHandler prm;
      ParametersType           parameters;
      parameters.process_parameters_file(prm, parameter_file);

      // Print number of processes and GIT hashes if verbosity level >= 3
      if (parameters.base.verbosity_level >= 3)
        {
          dealii::ConditionalOStream pcout(
            std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
          Journal::print_decoration_line(pcout);
          Journal::print_line(pcout,
                              "Running simulation on " +
                                std::to_string(
                                  dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)) +
                                " ranks.");
          Journal::print_decoration_line(pcout);
          pcout << "  - deal.II:" << std::endl
                << "      * branch: " << DEAL_II_GIT_BRANCH << std::endl
                << "      * revision: " << DEAL_II_GIT_REVISION << std::endl
                << "      * short: " << DEAL_II_GIT_SHORTREV << std::endl;
          Journal::print_decoration_line(pcout);
        }

      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
          parameters.base.do_print_parameters)
        parameters.print_parameters(prm, std::cout, false /*print_details*/);

      dim       = parameters.base.dimension;
      number    = parameters.base.number;
      case_name = parameters.base.case_name;
    }

    // Try to run the simulation based on the dimension
    try
      {
        if (number == "double")
          {
            if (dim == 1)
              {
                auto sim =
                  SimulationCaseFactory<CaseType<1, double>>::create_simulation(case_name,
                                                                                parameter_file,
                                                                                mpi_communicator);
                sim->create();
                auto problem = std::make_unique<ProblemType<1, double>>(std::move(sim));
                problem->run();
              }
            else if (dim == 2)
              {
                auto sim =
                  SimulationCaseFactory<CaseType<2, double>>::create_simulation(case_name,
                                                                                parameter_file,
                                                                                mpi_communicator);
                sim->create();
                auto problem = std::make_unique<ProblemType<2, double>>(std::move(sim));
                problem->run();
              }
            else if (dim == 3)
              {
                auto sim =
                  SimulationCaseFactory<CaseType<3, double>>::create_simulation(case_name,
                                                                                parameter_file,
                                                                                mpi_communicator);
                sim->create();
                auto problem = std::make_unique<ProblemType<3, double>>(std::move(sim));
                problem->run();
              }
            else
              {
                AssertThrow(false, dealii::ExcMessage("Dimension must be 1, 2, or 3."));
              }
          }
        else
          {
            AssertThrow(false,
                        dealii::ExcMessage("Currently, explicit template instantiations are "
                                           "only done for floating point number format 'double'."));
          }
      }
    catch (std::exception &exc)
      {
        std::cerr << "\n\n----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
      }
    catch (...)
      {
        std::cerr << "\n\n----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
      }
  }

  template <typename Parameters,
            template <int, typename>
            class Case,
            template <int, typename>
            class Problem>
  void
  default_main(int argc, char *argv[], MPI_Comm mpi_comm)
  {
    // Ensure at least one input file is provided
    if (argc < 2)
      {
        AssertThrow(false, dealii::ExcMessage("no input file specified"));
        return;
      }

    std::string input_file = argv[argc - 1]; // Last argument as input file

    // Handle help options
    if (argc == 3 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "--help-detail"))
      {
        dealii::ParameterHandler prm;
        Parameters               parameters;
        parameters.process_parameters_file(prm, input_file);

        if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
          parameters.print_parameters(prm, std::cout, std::string(argv[1]) == "--help-detail");

        return;
      }

    AssertThrow(argc < 4,
                dealii::ExcMessage(
                  "The provided number of command line parameters is not supported."));

    run_simulation<Parameters, Case, Problem>(input_file, mpi_comm);
  }
} // namespace MeltPoolDG
