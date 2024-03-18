#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

/**
 * This simulation is mainly meant to test the functionality of RTE
 */

namespace MeltPoolDG::Simulation::RadiativeTransport
{
  BETTER_ENUM(InterfaceCase, char, straight, single_powder_particle)

  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  template <int dim>
  class LevelSetHeaviside : public Function<dim>
  {
  public:
    LevelSetHeaviside(const InterfaceCase              interface_case_in,
                      const std::pair<double, double> &interface_case_info_in,
                      const double                     epsilon_cell_in)
      : Function<dim>(1)
      , eps(epsilon_cell_in)
      , interface_case(interface_case_in)
      , interface_case_info(interface_case_info_in)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      if (interface_case == InterfaceCase::straight)
        {
          const auto y            = p[dim - 1];
          const auto current_time = this->get_time();
          return UtilityFunctions::CharacteristicFunctions::heaviside(
            level +
              interface_case_info.first *
                ((current_time < interface_case_info.second) ? current_time :
                                                               interface_case_info.second) /
                interface_case_info.second -
              y,
            eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid
        }
      else if (interface_case == InterfaceCase::single_powder_particle)
        {
          // say we inherit from a straight interface case
          const auto y = p[dim - 1];

          double straight_value = UtilityFunctions::CharacteristicFunctions::heaviside(
            level - y,
            eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid

          Point<dim> sphere_center;
          sphere_center[dim - 1] = interface_case_info.first;

          // now add a powder particle with gradient:
          const Functions::SignedDistance::Sphere<dim> distance_sphere(sphere_center,
                                                                       interface_case_info.second);
          double                                       powder_particle_value =
            UtilityFunctions::CharacteristicFunctions::heaviside(-distance_sphere.value(p), eps);
          return std::max(straight_value, powder_particle_value);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
          return 0.0;
        }
      return 0.0;
    }

  private:
    const double              eps                 = 0.1;
    const double              level               = 0.0;
    InterfaceCase             interface_case      = InterfaceCase::straight;
    std::pair<double, double> interface_case_info = std::pair<double, double>(0., 0.);
  };



  template <int dim>
  class RadiativeTransportSimulation : public SimulationBase<dim>
  {
  public:
    RadiativeTransportSimulation(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
      , cell_repetitions(dim, 1)
    {}

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("domain x min",
                          domain_x_min,
                          "minimum x coordinate of simulation domain");
        prm.add_parameter("domain y min",
                          domain_y_min,
                          "minimum y coordinate of simulation domain");
        prm.add_parameter("domain x max",
                          domain_x_max,
                          "maximum x coordinate of simulation domain");
        prm.add_parameter("domain y max",
                          domain_y_max,
                          "maximum y coordinate of simulation domain");
        prm.add_parameter("cell repetitions",
                          cell_repetitions,
                          "cell repetitions per dim applied before global refinement or amr");

        prm.add_parameter("power",
                          power_in,
                          "Sets the intensity scale of the laser source. Is a scalar value");
        prm.add_parameter("source center", center_in, "location of the heat source center");
        prm.add_parameter("source radius", radius_in, "heat source radius");

        prm.add_parameter(
          "interface case",
          interface_case,
          "kind of interface for this simulation. "
          "straight: straigt interface that moves upwards; "
          "single_powder_particle: a single hanging powder particle above a static straight interface; "
          "powder bed: ");

        prm.add_parameter("straight interface upward speed",
                          speed,
                          "straight interface upward speed");
        prm.add_parameter("straight interface movement end time",
                          end_time,
                          "end time of the straight interface movement");

        prm.add_parameter("powder particle radius",
                          powder_particle_radius,
                          "hanging powder particle radius");
        prm.add_parameter("powder particle offset",
                          powder_particle_offset,
                          "hanging powder particle offset from [dim-1] = 0 plane");
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
        {
#ifdef DEAL_II_WITH_METIS
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (Triangulation<dim>::MeshSmoothing::none),
            true,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
          AssertThrow(
            false,
            ExcMessage(
              "Missing Metis support of the deal.II installation. "
              "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      const Point<dim> bottom_left =
        (dim == 1) ? Point<dim>(domain_y_min) :
        (dim == 2) ? Point<dim>(domain_x_min, domain_y_min) :
                     Point<dim>(domain_x_min, domain_x_min, domain_y_min);
      const Point<dim> top_right =
        (dim == 1) ? Point<dim>(domain_y_max) :
        (dim == 2) ? Point<dim>(domain_x_max, domain_y_max) :
                     Point<dim>(domain_x_max, domain_x_max, domain_y_max);
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          std::vector<unsigned int> subdivisions(
            dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;
          for (int d = 0; d < dim; d++)
            subdivisions[d] *= cell_repetitions[d];

          GridGenerator::subdivided_hyper_rectangle_with_simplices(
            *this->triangulation, subdivisions, bottom_left, top_right, true /*colorize*/);
        }
      else
        {
          GridGenerator::subdivided_hyper_rectangle(
            *this->triangulation, cell_repetitions, bottom_left, top_right, true /*colorize*/);
        }
    }

    void
    set_boundary_conditions() final
    {
      // face numbering according to the deal.II colorize flag
      [[maybe_unused]] const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      if (this->parameters.base.problem_name == ProblemType::radiative_transport)
        this->attach_dirichlet_boundary_condition(
          upper_bc,
          std::make_shared<Heat::GaussProjectionIntensityProfile<dim, double>>(
            power_in, radius_in, center_in, -dealii::Point<dim>::unit_vector(dim - 1)),
          "intensity");
      else
        this->parameters.laser.rte_boundary_id = upper_bc;

      if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
        this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_field_conditions() final
    {
      // pass simulation-specific parameters to the simulation class.
      // Done after json parsing, is relevant for heaviside
      if (interface_case == InterfaceCase::straight)
        interface_case_info = std::pair<double, double>(speed, end_time);
      else if (interface_case == InterfaceCase::single_powder_particle)
        interface_case_info =
          std::pair<double, double>(powder_particle_offset, powder_particle_radius);

      // determine the interface epsilon parameter from minimum mesh size
      double       thickness_scale_factor = 2.5;
      const double epsilon_cell           = GridTools::minimal_cell_diameter(*this->triangulation) /
                                  std::sqrt(dim) * thickness_scale_factor;

      // attach the prescribed heaviside function field
      this->attach_initial_condition(std::make_shared<LevelSetHeaviside<dim>>(interface_case,
                                                                              interface_case_info,
                                                                              epsilon_cell),
                                     "prescribed_heaviside");

      if (this->parameters.base.problem_name == ProblemType::radiative_transport)
        this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(),
                                       "intensity");

      // this is only used to TODO generate the comparison solution with the gaussian laser
      if (this->parameters.base.problem_name == ProblemType::heat_transfer)
        {
          // attach dummy initial conditions for the heat transfer operation
          this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(),
                                         "heat_transfer");
        }
    }

  private:
    double                    domain_x_min = -1.;
    double                    domain_x_max = 1.;
    double                    domain_y_min = -1.;
    double                    domain_y_max = 1.;
    std::vector<unsigned int> cell_repetitions;
    std::pair<double, double> interface_case_info;
    double                    power_in = 0.1;
    Point<dim>                center_in;
    double                    radius_in = domain_x_max / 5.;

    InterfaceCase interface_case = InterfaceCase::straight;

    double speed    = domain_x_max / 2;
    double end_time = 10.0;

    double powder_particle_offset = domain_x_max / 4.;
    double powder_particle_radius = domain_x_max / 6.;
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
