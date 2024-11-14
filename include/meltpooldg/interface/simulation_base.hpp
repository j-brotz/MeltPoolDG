#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/exceptions.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/periodic_boundary_conditions.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, int spacedim = dim>
  class SimulationBase
  {
  public:
    SimulationBase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : parameter_file(parameter_file_in)
      , mpi_communicator(mpi_communicator_in)
    {
      set_parameters();
    }

    virtual ~SimulationBase() = default;

    /**
     * add simulation-specific parameters to the parameter handler
     * @note they will be available after create() has been
     * called.
     */
    virtual void
    add_simulation_specific_parameters(dealii::ParameterHandler &)
    {
      return;
      // default: do nothing
    }

    virtual void
    set_parameters()
    {
      /*
       * read parameters defined in parameters.hpp
       */
      dealii::ParameterHandler prm;
      this->parameters.process_parameters_file(prm, this->parameter_file);
    }

    virtual void
    set_boundary_conditions() = 0;

    virtual void
    set_field_conditions() = 0;

    virtual void
    create_spatial_discretization() = 0;

    virtual void
    create()
    {
      set_simulation_specific_parameters();
      create_spatial_discretization();
      AssertThrow(
        this->triangulation,
        dealii::ExcMessage(
          "It seems that your SimulationBase object does not contain"
          " a valid triangulation object. A shared_ptr to your triangulation"
          " must be specified as follows for a serialized triangulation "
          " this->triangulation = std::make_shared<Triangulation<dim>>(); "
          " or for a parallel triangulation "
          " this->triangulation = std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator); "));
      set_boundary_conditions();
      set_field_conditions();
    }

    /**
     * This function may be overwritten for output purposes in your simulation case.
     * It is called after every time step.
     */
    virtual void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const
    {
      (void)generic_data_out;
      // do nothing default
    }

    /**
     * Read the simulation specific parameters
     */
    void
    set_simulation_specific_parameters()
    {
      /*
       * read user-defined parameters
       */
      dealii::ParameterHandler prm_simulation_specific;
      prm_simulation_specific.clear();
      add_simulation_specific_parameters(prm_simulation_specific);

      std::ifstream file;
      file.open(parameter_file);

      if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "json")
        prm_simulation_specific.parse_input_from_json(file, true);
      else if (parameter_file.substr(parameter_file.find_last_of(".") + 1) == "prm")
        prm_simulation_specific.parse_input(parameter_file);
      else
        AssertThrow(false,
                    dealii::ExcMessage("Parameterhandler cannot handle current file ending"));

      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
          this->parameters.base.do_print_parameters)
        {
          std::cout << "Simulation-specific paramters:" << std::endl;
          print_parameters_external(prm_simulation_specific, std::cout, false /*print_details*/);
        }
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
    attach_field_function(std::shared_ptr<Function<dim>> function,
                          const std::string             &type,
                          const std::string             &operation_name)
    {
      field_functions[operation_name][type] = function;
    }

    /**
     * Wrapper function to attach initial conditions. See documentation of
     * attach_field_function.
     */
    void
    attach_initial_condition(std::shared_ptr<Function<dim>> initial_function,
                             const std::string             &operation_name)
    {
      attach_field_function(initial_function, "initial_condition", operation_name);
    }

    /**
     * Attach functions for boundary conditions
     */

    template <typename FunctionType>
    void
    attach_dirichlet_boundary_condition(dealii::types::boundary_id    id,
                                        std::shared_ptr<FunctionType> boundary_function,
                                        const std::string            &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      boundary_conditions_map[operation_name]->dirichlet_bc.attach(id, boundary_function);
    }

    template <typename FunctionType>
    void
    attach_neumann_boundary_condition(dealii::types::boundary_id    id,
                                      std::shared_ptr<FunctionType> boundary_function,
                                      const std::string            &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      AssertThrow(boundary_conditions_map[operation_name]->neumann_bc.count(id) == 0,
                  ExcBCAlreadyAssigned("Neumann"));

      boundary_conditions_map[operation_name]->neumann_bc[id] = boundary_function;
    }

    template <typename FunctionType>
    void
    attach_inflow_outflow_boundary_condition(dealii::types::boundary_id    id,
                                             std::shared_ptr<FunctionType> boundary_function,
                                             const std::string            &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      AssertThrow(boundary_conditions_map[operation_name]->inflow_outflow_bc.count(id) == 0,
                  ExcBCAlreadyAssigned("Inflow outflow"));

      boundary_conditions_map[operation_name]->inflow_outflow_bc[id] = boundary_function;
    }

    void
    attach_no_slip_boundary_condition(dealii::types::boundary_id id,
                                      const std::string         &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->no_slip_bc;

      AssertThrow(std::find(bc.begin(), bc.end(), id) == bc.end(), ExcBCAlreadyAssigned("no-slip"));

      bc.push_back(id);
    }

    void
    attach_open_boundary_condition(dealii::types::boundary_id id, const std::string &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      AssertThrow(boundary_conditions_map[operation_name]->open_boundary_bc.count(id) == 0,
                  ExcBCAlreadyAssigned("open BC"));

      boundary_conditions_map[operation_name]->open_boundary_bc[id] =
        std::make_shared<dealii::Functions::ZeroFunction<dim>>();
    }

    void
    attach_open_boundary_condition(dealii::types::boundary_id             id,
                                   std::shared_ptr<dealii::Function<dim>> boundary_function,
                                   const std::string                     &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      AssertThrow(boundary_conditions_map[operation_name]->open_boundary_bc.count(id) == 0,
                  ExcBCAlreadyAssigned("open BC"));

      boundary_conditions_map[operation_name]->open_boundary_bc[id] = boundary_function;
    }

    /**
     * This boundary condition fixes the pressure to a constant level. On the given boundary @p id,
     * an arbitrary point is selected for a constant dirichlet condition.
     *
     * @note The boundary @p id must not be a periodic boundary selected by
     *   attach_periodic_boundary_condition().
     */
    void
    attach_fix_pressure_constant_condition(dealii::types::boundary_id id,
                                           const std::string         &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->fix_pressure_constant;

      AssertThrow(std::find(bc.begin(), bc.end(), id) == bc.end(),
                  ExcBCAlreadyAssigned("fix pressure constant constant"));

      bc.push_back(id);
    }

    void
    attach_symmetry_boundary_condition(dealii::types::boundary_id id,
                                       const std::string         &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->symmetry_bc;

      AssertThrow(std::find(bc.begin(), bc.end(), id) == bc.end(),
                  ExcBCAlreadyAssigned("symmetry"));

      bc.push_back(id);
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
                  ExcMessage("You try to pass periodic faces but the triangulation "
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

    void
    attach_radiation_boundary_condition(dealii::types::boundary_id id,
                                        const std::string         &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->radiation_bc;

      AssertThrow(std::find(bc.begin(), bc.end(), id) == bc.end(),
                  ExcBCAlreadyAssigned("radiation"));

      bc.push_back(id);
    }

    void
    attach_convection_boundary_condition(dealii::types::boundary_id id,
                                         const std::string         &operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->convection_bc;

      AssertThrow(std::find(bc.begin(), bc.end(), id) == bc.end(),
                  ExcBCAlreadyAssigned("convection"));

      bc.push_back(id);
    }

    /*
     * getter functions
     */
    virtual MPI_Comm
    get_mpi_communicator() const
    {
      return this->mpi_communicator;
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
        return field_conditions[type];
      else // is_optional = true
        return nullptr;
    }

    std::shared_ptr<dealii::Function<dim>>
    get_initial_condition(const std::string &operation_name, const bool is_optional = false)
    {
      return get_field_function("initial_condition", operation_name, is_optional);
    }

    /**
     * Attach a new BoundaryConditions<dim> with id @p operation_name.
     */
    void
    attach_boundary_condition(const std::string &operation_name)
    {
      if (boundary_conditions_map.count(operation_name) == 0)
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();
    }

    /**
     * Getter functions for boundary conditions
     */
    const auto &
    get_bc(const std::string &operation_name) const
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();
      return boundary_conditions_map[operation_name];
    }

    std::shared_ptr<BoundaryConditions<dim>>
    get_bc(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          " not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));

      return boundary_conditions_map[operation_name];
    }

    const auto &
    get_dirichlet_bc(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          "not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));

      return boundary_conditions_map[operation_name]->dirichlet_bc;
    }

    const auto &
    get_neumann_bc(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          "not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));
      return boundary_conditions_map[operation_name]->neumann_bc;
    }

    const auto &
    get_inflow_outflow_bc(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          "not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));
      return boundary_conditions_map[operation_name]->inflow_outflow_bc;
    }

    const auto &
    get_open_boundary_bc(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          "not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));
      return boundary_conditions_map[operation_name]->open_boundary_bc;
    }

    const std::vector<dealii::types::boundary_id> &
    get_no_slip_id(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          "not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));
      return boundary_conditions_map[operation_name]->no_slip_bc;
    }

    const std::vector<dealii::types::boundary_id> &
    get_fix_pressure_constant_id(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          "not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));
      return boundary_conditions_map[operation_name]->fix_pressure_constant;
    }

    const std::vector<dealii::types::boundary_id> &
    get_symmetry_id(const std::string &operation_name)
    {
      AssertThrow(
        boundary_conditions_map[operation_name],
        dealii::ExcMessage(
          "BC for " + operation_name +
          "not found. "
          "Did you forget to register the operation via attach_boundary_condition(operation_name)?"));
      return boundary_conditions_map[operation_name]->symmetry_bc;
    }

    const auto &
    get_periodic_bc() const
    {
      return periodic_boundary_conditions;
    }

    const std::vector<dealii::types::boundary_id> &
    get_radiation_id(const std::string &operation_name)
    {
      return boundary_conditions_map[operation_name]->radiation_bc;
    }

    const std::vector<types::boundary_id> &
    get_convection_id(const std::string &operation_name)
    {
      return boundary_conditions_map[operation_name]->convection_bc;
    }

  public:
    const std::string                                     parameter_file;
    const MPI_Comm                                        mpi_communicator;
    Parameters<double>                                    parameters;
    std::shared_ptr<dealii::Triangulation<dim, spacedim>> triangulation;

  private:
    std::map<std::string, std::shared_ptr<BoundaryConditions<dim>>> boundary_conditions_map;
    PeriodicBoundaryConditions<dim>                                 periodic_boundary_conditions;

    // nested dictionary for storing field functions as
    // field_functions[<operation_name>][<type>] = <function>.
    // The <operation_name> and <type> are specified in the respective problems.
    std::map<std::string, std::map<std::string, std::shared_ptr<Function<dim>>>> field_functions;
  };
} // namespace MeltPoolDG
