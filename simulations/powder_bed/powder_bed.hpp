#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/melt_pool/powder_bed.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <memory>
#include <string>
#include <vector>

namespace MeltPoolDG::Simulation::PowderBed
{
  template <int dim>
  class SimulationPowderBed : public SimulationBase<dim>
  {
  private:
    double                    domain_x_min = 0;
    double                    domain_x_max = 0;
    double                    domain_y_min = 0;
    double                    domain_y_max = 0;
    double                    domain_z_min = 0;
    double                    domain_z_max = 0;
    std::vector<unsigned int> cell_repetitions;
    double                    T_initial = 500;
    MeltPool::PowderBedData   powder_bed_data;

  public:
    SimulationPowderBed(std::string parameter_file, const MPI_Comm mpi_communicator)
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
        prm.add_parameter("domain x max",
                          domain_x_max,
                          "maximum x coordinate of simulation domain");
        prm.add_parameter("domain y min",
                          domain_y_min,
                          "minimum y coordinate of simulation domain");
        prm.add_parameter("domain y max",
                          domain_y_max,
                          "maximum y coordinate of simulation domain");
        prm.add_parameter("domain z min",
                          domain_z_min,
                          "minimum z coordinate of simulation domain");
        prm.add_parameter("domain z max",
                          domain_z_max,
                          "maximum z coordinate of simulation domain");
        prm.add_parameter("cell repetitions",
                          cell_repetitions,
                          "cell repetitions per dim applied before global refinement or amr");
        prm.add_parameter("initial temperature", T_initial, "Set the initial temperature.");
        powder_bed_data.add_parameters(prm);
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
            (Triangulation<dim>::none),
            false,
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
        (dim == 1) ? Point<dim>(domain_x_min) :
        (dim == 2) ? Point<dim>(domain_x_min, domain_y_min) :
                     Point<dim>(domain_x_min, domain_y_min, domain_z_min);
      const Point<dim> top_right =
        (dim == 1) ? Point<dim>(domain_x_max) :
        (dim == 2) ? Point<dim>(domain_x_max, domain_y_max) :
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

    void
    set_boundary_conditions() override
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
      this->attach_dirichlet_boundary_condition(
        lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_initial), "heat_transfer");

      if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
        this->triangulation->refine_global(this->parameters.base.global_refinements);
      /*
       * BC for RTE
       */
      if (this->parameters.base.problem_name == ProblemType::radiative_transport)
        this->attach_dirichlet_boundary_condition(
          upper_bc,
          std::make_shared<Heat::GaussProjectionIntensityProfile<dim, double>>(
            this->parameters.laser.power,
            this->parameters.laser.radius,
            this->parameters.laser.template get_starting_position<dim>(),
            this->parameters.laser.template get_direction<dim>()),
          "intensity");
      else
        this->parameters.laser.rte_boundary_id = upper_bc;
    }

    void
    set_field_conditions() override
    {
      if (this->parameters.laser.model == Heat::LaserModelType::RTE)
        this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(),
                                       "intensity");
      if (this->parameters.base.problem_name == ProblemType::heat_transfer)
        {
          // attach initial temperature
          this->attach_initial_condition(
            std::make_shared<Functions::ConstantFunction<dim>>(T_initial), "heat_transfer");
          // attach prescribed heaviside
          this->attach_initial_condition(
            std::make_shared<MeltPool::PowderBedLevelSet<dim>>(
              powder_bed_data,
              MeltPool::LevelSetType::heaviside,
              this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
                GridTools::minimal_cell_diameter(*this->triangulation) /
                this->parameters.ls.get_n_subdivisions() / std::sqrt(dim))),
            "prescribed_heaviside");
        }
    }
  };

} // namespace MeltPoolDG::Simulation::PowderBed
