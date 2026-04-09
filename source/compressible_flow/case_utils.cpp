#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <meltpooldg/compressible_flow/case_utils.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number                                         initial_time,
    const std::shared_ptr<dealii::Function<dim, number>> density,
    const std::shared_ptr<dealii::Function<dim, number>> velocity,
    const std::shared_ptr<dealii::Function<dim, number>> inner_energy,
    const std::shared_ptr<dealii::Function<dim, number>> mass_fractions)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + mass_fractions->n_components,
                                    initial_time)
    , density(density)
    , velocity(velocity)
    , inner_energy(inner_energy)
    , mass_fractions(mass_fractions)
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number       initial_time,
    const std::string &density_function_string,
    const std::string &velocity_function_string,
    const std::string &inner_energy_function_string,
    const std::string &mass_fraction_functions_string,
    const unsigned     n_mass_fractions)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + n_mass_fractions, initial_time)
    , density(std::make_shared<dealii::FunctionParser<dim>>(density_function_string))
    , velocity(std::make_shared<dealii::FunctionParser<dim>>(velocity_function_string))
    , inner_energy(std::make_shared<dealii::FunctionParser<dim>>(inner_energy_function_string))
    , mass_fractions(n_mass_fractions == 0 ? nullptr :
                                             std::make_shared<dealii::FunctionParser<dim>>(
                                               mass_fraction_functions_string))
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number       initial_time,
    const std::string &density_function_string,
    const std::string &velocity_function_string,
    const std::string &inner_energy_function_string)
    : dealii::Function<dim, number>(n_conserved_variables<dim>, initial_time)
    , density(std::make_shared<dealii::FunctionParser<dim>>(density_function_string))
    , velocity(std::make_shared<dealii::FunctionParser<dim>>(velocity_function_string))
    , inner_energy(std::make_shared<dealii::FunctionParser<dim>>(inner_energy_function_string))
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  void
  ConservativeVariablesFunction<dim, number>::set_time(const number new_time)
  {
    dealii::Function<dim>::set_time(new_time);
    density->set_time(new_time);
    velocity->set_time(new_time);
    inner_energy->set_time(new_time);

    // Only set time for species function if it exists, since this function can also be used for
    // single-species flow cases where no species function is provided.
    if (mass_fractions)
      mass_fractions->set_time(new_time);
  }

  template <int dim, typename number>
  number
  ConservativeVariablesFunction<dim, number>::value(const dealii::Point<dim, number> &loc,
                                                    const unsigned int component) const
  {
    const auto ramp_up_scaling = [&](const RampUpParameters param) -> number {
      if (this->get_time() > param.duration or param.duration == 0. or
          param.type == RampUpType::none)
        return 1.;

      switch (param.type)
        {
          case RampUpType::linear:
            return this->get_time() / param.duration;
            break;
          case RampUpType::exponential:
            return (std::exp(this->get_time() / param.duration) - 1) / (std::numbers::e - 1);
            break;
          case RampUpType::cosine:
            return 0.5 * (1 - std::cos(std::numbers::pi * this->get_time() / param.duration));
            break;
          default:
            AssertThrow(false, dealii::ExcInternalError());
        }
    };

    const auto compute_conserved_energy = [&](const dealii::Point<dim, number> &loc) -> number {
      dealii::Vector<number> velocity_vec(dim);
      velocity->vector_value(loc, velocity_vec);
      velocity_vec *= ramp_up_scaling(velocity_ramp_up);
      return density->value(loc, 0) * (inner_energy->value(loc, 0) + 0.5 * velocity_vec.norm_sqr());
    };

    if (component == 0)
      return density->value(loc, 0);
    else if (component > 0 and component <= dim)
      return density->value(loc, 0) * velocity->value(loc, component - 1) *
             ramp_up_scaling(velocity_ramp_up);
    else if (component == dim + 1)
      return compute_conserved_energy(loc);
    else if (component < this->n_components)
      return density->value(loc, 0) * mass_fractions->value(loc, component - dim - 2);
    else
      AssertThrow(false,
                  dealii::ExcMessage("Invalid component for conservative variable function: " +
                                     std::to_string(component) +
                                     ". Only components from within the range [0, " +
                                     std::to_string(this->n_components) + ") are valid."));
  }

  template <int dim, typename number>
  std::shared_ptr<dealii::Function<dim>>
  InputDefinedBoundaryCondition<dim, number>::create_boundary_function(const number start_time,
                                                                       const int    n_species) const
  {
    switch (type)
      {
          case (BoundaryConditionType::inflow): {
            auto inflow_bc = std::make_shared<
              MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<dim, number>>(
              start_time, density, velocity, energy, species_mass_fractions, n_species - 1);

            inflow_bc->set_velocity_ramp_up(inflow_velocity_ramp_up.duration,
                                            inflow_velocity_ramp_up.type);
            return inflow_bc;
          }
          case (BoundaryConditionType::subsonic_outflow_fixed_energy): {
            auto energy_boundary_function =
              std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
            energy_boundary_function->initialize(
              dealii::FunctionParser<dim>::default_variable_names(),
              energy,
              typename dealii::FunctionParser<dim>::ConstMap());
            return energy_boundary_function;
          }
          case (BoundaryConditionType::subsonic_outflow_fixed_pressure): {
            auto pressure_boundary_function =
              std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
            pressure_boundary_function->initialize(
              dealii::FunctionParser<dim>::default_variable_names(),
              pressure,
              typename dealii::FunctionParser<dim>::ConstMap());
            return pressure_boundary_function;
          }
        default:
          return std::make_shared<dealii::Functions::ConstantFunction<dim>>(0.);
      }
  }

  template <int dim, typename number>
  void
  InputDefinedBoundaryCondition<dim, number>::add_user_boundary_condition_parameters(
    dealii::ParameterHandler &prm,
    const std::string        &boundary_location)
  {
    prm.enter_subsection(boundary_location);
    {
      prm.add_parameter("type", type, "Type of the boundary.");
      prm.add_parameter(
        "density",
        density,
        "Prescribed density at the boundary (depending on type of boundary condition).");
      prm.add_parameter(
        "velocity",
        velocity,
        "Prescribed velocity at the boundary (depending on type of boundary condition). "
        "Velocity components are separated by semicolon, e.g., '1.0; 0.0; 0.0' for 3D.");
      prm.add_parameter(
        "pressure",
        pressure,
        "Prescribed pressure at the boundary (depending on type of boundary condition).");
      prm.add_parameter(
        "energy",
        energy,
        "Prescribed energy at the boundary (depending on type of boundary condition).");
      prm.add_parameter(
        "species mass fractions",
        species_mass_fractions,
        "Prescribed mass fractions of species at the boundary (only used for multi-component flow cases).");
      prm.add_parameter(
        "inflow velocity ramp-up duration",
        inflow_velocity_ramp_up.duration,
        "In the case of an inflow boundary, the time ramp-up duration of the inflow velocity.");
      prm.add_parameter(
        "inflow velocity ramp-up type",
        inflow_velocity_ramp_up.type,
        "In the case of an inflow boundary, the ramp-up type of the inflow velocity. "
        "Supported options are 'linear', 'exponential' and 'cosine'.");
    }
    prm.leave_subsection();
  }

  template <int dim, typename number>
  void
  add_hyper_rectangle_custom_boundary_condition_parameters(
    dealii::ParameterHandler                                        &prm,
    std::array<InputDefinedBoundaryCondition<dim, number>, 2 * dim> &boundary_conditions)
  {
    if constexpr (dim >= 1)
      {
        boundary_conditions[0].add_user_boundary_condition_parameters(prm, "boundary x min");
        boundary_conditions[1].add_user_boundary_condition_parameters(prm, "boundary x max");
      }
    if constexpr (dim >= 2)
      {
        boundary_conditions[2].add_user_boundary_condition_parameters(prm, "boundary y min");
        boundary_conditions[3].add_user_boundary_condition_parameters(prm, "boundary y max");
      }
    if constexpr (dim == 3)
      {
        boundary_conditions[4].add_user_boundary_condition_parameters(prm, "boundary z min");
        boundary_conditions[5].add_user_boundary_condition_parameters(prm, "boundary z max");
      }
  }

  template <int dim, typename number>
  std::shared_ptr<dealii::Function<dim, number>>
  InputDefinedInitialCondition<dim, number>::create_initial_condition_function(
    const number initial_time,
    const int    n_species) const
  {
    return std::make_shared<
      MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<dim, number>>(
      initial_time, density, velocity, inner_energy, species_mass_fractions, n_species - 1);
  }

  template <int dim, typename number>
  void
  InputDefinedInitialCondition<dim, number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("initial conditions");
    {
      prm.add_parameter("density", density, "Initial density field value or spatial distribution.");
      prm.add_parameter(
        "velocity",
        velocity,
        "Initial velocity field. "
        "Specify components separated by semicolons, e.g., '1.0; 0.0; 0.0' for a 3D simulation.");
      prm.add_parameter("energy",
                        inner_energy,
                        "Initial specific inner energy field value or distribution.");
      prm.add_parameter(
        "species mass fractions",
        species_mass_fractions,
        "Initial species mass fractions field value or distribution (only used for multi-component flow cases).");
    }
    prm.leave_subsection();
  }



  template <int dim, typename number>
  void
  InputDefinedSubdividedHyperRectangleDomain<dim, number>::create_triangulation(
    std::shared_ptr<dealii::Triangulation<dim>> &triangulation,
    const MPI_Comm                              &mpi_communicator,
    const unsigned                               n_global_refinements)
  {
    auto create_point_from_container =
      []<typename T>(const T &container) -> dealii::Point<dim, number> {
      dealii::Point<dim, number> point;
      if constexpr (dim == 1)
        point = dealii::Point<dim, number>(container[0]);
      else if constexpr (dim == 2)
        point = dealii::Point<dim, number>(container[0], container[1]);
      else if constexpr (dim == 3)
        point = dealii::Point<dim, number>(container[0], container[1], container[2]);
      return point;
    };

    if constexpr (dim == 1)
      triangulation =
        std::make_shared<dealii::parallel::shared::Triangulation<dim>>(mpi_communicator);
    else
      triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(mpi_communicator);

    dealii::Point<dim, number> dimensions = create_point_from_container(domain_dimensions);

    dealii::GridGenerator::subdivided_hyper_rectangle(
      *triangulation, domain_base_discretization, dealii::Point<dim, number>(), dimensions, true);

    AssertThrow(periodic_boundaries.size() == dim,
                dealii::ExcMessage("Invalid size for periodic boundaries vector. "
                                   "Expected size " +
                                   std::to_string(dim) + ", but got size " +
                                   std::to_string(periodic_boundaries.size()) + "."));

    for (unsigned int d = 0; d < dim; ++d)
      {
        if (periodic_boundaries[d])
          {
            std::vector<dealii::GridTools::PeriodicFacePair<
              typename dealii::Triangulation<dim>::cell_iterator>>
              periodicity_vector;

            dealii::GridTools::collect_periodic_faces(
              *(triangulation), 2 * d, 2 * d + 1, d, periodicity_vector);
            triangulation->add_periodicity(periodicity_vector);
          }
      }

    triangulation->refine_global(n_global_refinements);

    // Perform additional mesh refinement in the user-defined region of interest
    if (region_of_interest_refinement_times > 0)
      {
        AssertThrow(region_of_interest_corner[0].size() == dim &&
                      region_of_interest_corner[1].size() == dim,
                    dealii::ExcMessage("Invalid size for region-of-interest corner points. "
                                       "Both points must have size " +
                                       std::to_string(dim) + ", but got sizes " +
                                       std::to_string(region_of_interest_corner[0].size()) +
                                       " and " +
                                       std::to_string(region_of_interest_corner[1].size()) + "."));


        std::vector<AMR::AMRRegion<dim, number>> regions;
        regions.emplace_back(dealii::BoundingBox<dim, number>(
          {create_point_from_container(region_of_interest_corner[0]),
           create_point_from_container(region_of_interest_corner[1])}));
        for (unsigned i = 0; i < region_of_interest_refinement_times; ++i)
          {
            AMR::set_refinement_flags_in_regions<dim, number>(*triangulation, regions);
            triangulation->execute_coarsening_and_refinement();
          }
      }
  }

  template <int dim, typename number>
  void
  InputDefinedSubdividedHyperRectangleDomain<dim, number>::add_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("domain");
    {
      prm.add_parameter(
        "size",
        domain_dimensions,
        "Physical dimensions of the computational domain. "
        "Specify comma-separated values: the first for the x-direction, the second for the y-direction, "
        "and optionally the third for the z-direction.");
      prm.add_parameter(
        "base grid resolution",
        domain_base_discretization,
        "Number of base cells in each spatial direction. "
        "Provide comma-separated values: the first for the x-direction, the second for the y-direction, "
        "and optionally the third for the z-direction. "
        "Note: the final mesh resolution may differ depending on other parameters "
        "(e.g., global refinements).");
      prm.add_parameter(
        "periodic boundary conditions",
        periodic_boundaries,
        "Specify whether the boundaries in each direction are periodic. "
        "Provide comma-separated boolean values (true/false) for each direction, e.g., 'true; false; false' for a 3D simulation with periodicity only in the x-direction.");

      prm.enter_subsection("region of interest");
      {
        prm.add_parameter(
          "first corner",
          region_of_interest_corner[0],
          "Coordinates of the bottom corner of the region of interest. "
          "Provide comma-separated values: the first for the x-coordinate, the second for the y-coordinate, "
          "and optionally the third for the z-coordinate.");
        prm.add_parameter(
          "second corner",
          region_of_interest_corner[1],
          "Coordinates of the top corner of the region of interest. "
          "Provide comma-separated values: the first for the x-coordinate, the second for the y-coordinate, "
          "and optionally the third for the z-coordinate.");
        prm.add_parameter(
          "refinements",
          region_of_interest_refinement_times,
          "Number of additional refinement levels to apply inside the defined region. "
          "A value of zero disables region-specific refinement.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

} // namespace MeltPoolDG::CompressibleFlow

template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<1, double>;
template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<2, double>;
template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<3, double>;

template struct MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<1, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<2, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<3, double>;

template struct MeltPoolDG::CompressibleFlow::InputDefinedSubdividedHyperRectangleDomain<1, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedSubdividedHyperRectangleDomain<2, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedSubdividedHyperRectangleDomain<3, double>;

template void
MeltPoolDG::CompressibleFlow::add_hyper_rectangle_custom_boundary_condition_parameters<1, double>(
  dealii::ParameterHandler &prm,
  std::array<MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<1, double>, 2>
    &boundary_conditions);
template void
MeltPoolDG::CompressibleFlow::add_hyper_rectangle_custom_boundary_condition_parameters<2, double>(
  dealii::ParameterHandler &prm,
  std::array<MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<2, double>, 4>
    &boundary_conditions);
template void
MeltPoolDG::CompressibleFlow::add_hyper_rectangle_custom_boundary_condition_parameters<3, double>(
  dealii::ParameterHandler &prm,
  std::array<MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<3, double>, 6>
    &boundary_conditions);

template struct MeltPoolDG::CompressibleFlow::InputDefinedInitialCondition<1, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedInitialCondition<2, double>;
template struct MeltPoolDG::CompressibleFlow::InputDefinedInitialCondition<3, double>;
