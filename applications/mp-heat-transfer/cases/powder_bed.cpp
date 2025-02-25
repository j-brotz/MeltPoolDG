#include "powder_bed.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/case_registration.hpp>
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>

#include <cmath>


namespace MeltPoolDG::Simulation::PowderBed
{
  template <int dim>
  SimulationPowderBed<dim>::SimulationPowderBed(std::string    parameter_file,
                                                const MPI_Comm mpi_communicator)
    : Heat::HeatTransferCase<dim>(parameter_file, mpi_communicator)
    , cell_repetitions(dim, 1)
  {}


  template <int dim>
  bool
  SimulationPowderBed<dim>::add_simulation_specific_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("simulation specific parameters");
    {
      prm.add_parameter("domain x min", domain_x_min, "minimum x coordinate of simulation domain");
      prm.add_parameter("domain x max", domain_x_max, "maximum x coordinate of simulation domain");
      prm.add_parameter("domain y min", domain_y_min, "minimum y coordinate of simulation domain");
      prm.add_parameter("domain y max", domain_y_max, "maximum y coordinate of simulation domain");
      prm.add_parameter("domain z min", domain_z_min, "minimum z coordinate of simulation domain");
      prm.add_parameter("domain z max", domain_z_max, "maximum z coordinate of simulation domain");
      prm.add_parameter("cell repetitions",
                        cell_repetitions,
                        "cell repetitions per dim applied before global refinement or amr");
      prm.add_parameter("initial temperature", T_initial, "Set the initial temperature.");
      powder_bed_data.add_parameters(prm);
    }
    prm.leave_subsection();

    return this->parameters.base.do_print_parameters;
  }


  template <int dim>
  void
  SimulationPowderBed<dim>::create_spatial_discretization()
  {
    if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
      {
#ifdef DEAL_II_WITH_METIS
        this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
          this->mpi_communicator,
          Triangulation<dim>::none,
          false,
          parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
        AssertThrow(
          false,
          ExcMessage("Missing Metis support of the deal.II installation. "
                     "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
      }
    else
      {
        this->triangulation =
          std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
      }

    const Point<dim> bottom_left = dim == 1 ? Point<dim>(domain_x_min) :
                                   dim == 2 ? Point<dim>(domain_x_min, domain_y_min) :
                                              Point<dim>(domain_x_min, domain_y_min, domain_z_min);
    const Point<dim> top_right   = dim == 1 ? Point<dim>(domain_x_max) :
                                   dim == 2 ? Point<dim>(domain_x_max, domain_y_max) :
                                              Point<dim>(domain_x_max, domain_y_max, domain_z_max);

    if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
      {
        GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                  cell_repetitions,
                                                  bottom_left,
                                                  top_right,
                                                  /* colorize */ true);
      }
    else // do simplex
      {
        std::vector<unsigned int> subdivisions(
          dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
        subdivisions[dim - 1] *= 2;
        for (int d = 0; d < dim; d++)
          subdivisions[d] *= cell_repetitions[d];

        GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                 subdivisions,
                                                                 bottom_left,
                                                                 top_right,
                                                                 /* colorize */ true);
      }
  }


  template <int dim>
  void
  SimulationPowderBed<dim>::set_boundary_conditions()
  {
    // face numbering according to the deal.II colorize flag
    const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
      get_colorized_rectangle_boundary_ids<dim>();

    /*
     * BC for two-phase flow
     * TODO
     */

    /*
     * BC for heat transfer
     */
    this->attach_boundary_condition({lower_bc,
                                     std::make_shared<Functions::ConstantFunction<dim>>(T_initial)},
                                    "dirichlet",
                                    "heat_transfer");

    if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    /*
     * BC for RTE
     */
    if (this->parameters.base.problem_name == "radiative_transport")
      this->attach_boundary_condition(
        {upper_bc,
         std::make_shared<Heat::GaussProjectionIntensityProfile<dim, double>>(
           this->parameters.laser.power,
           this->parameters.laser.radius,
           this->parameters.laser.template get_starting_position<dim>(),
           this->parameters.laser.template get_direction<dim>())},
        "dirichlet",
        "intensity");
    else
      this->parameters.laser.rte_boundary_id = upper_bc;
  }


  template <int dim>
  void
  SimulationPowderBed<dim>::set_field_conditions()
  {
    if (this->parameters.laser.model == Heat::LaserModelType::RTE)
      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(), "intensity");
    if (this->parameters.base.problem_name == "heat_transfer")
      {
        // attach initial temperature
        this->attach_initial_condition(
          std::make_shared<Functions::ConstantFunction<dim>>(T_initial), "heat_transfer");

        // create a temporary reinit data instance since reinit data is not available for the heat
        // problem
        LevelSet::ReinitializationData<double> reinit_data;
        reinit_data.interface_thickness_parameter.type =
          LevelSet::InterfaceThicknessParameterType::proportional_to_cell_size;
        reinit_data.interface_thickness_parameter.value = 1.5;

        // attach prescribed heaviside
        this->attach_initial_condition(std::make_shared<MeltPool::PowderBedLevelSet<dim>>(
                                         powder_bed_data,
                                         MeltPool::LevelSetType::heaviside,
                                         reinit_data.compute_interface_thickness_parameter_epsilon(
                                           GridTools::minimal_cell_diameter(*this->triangulation) /
                                           std::sqrt(dim))),
                                       "prescribed_heaviside");
      }
  }

  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase, SimulationPowderBed, "powder_bed", 1);
  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase, SimulationPowderBed, "powder_bed", 2);
  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase, SimulationPowderBed, "powder_bed", 3);
} // namespace MeltPoolDG::Simulation::PowderBed
