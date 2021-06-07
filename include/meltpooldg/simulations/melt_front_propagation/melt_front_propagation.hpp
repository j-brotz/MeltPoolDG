#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/heat/laser_heat_source_gauss.hpp>
#include <meltpooldg/heat/laser_heat_source_gusarov.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>

/**
 * This simulation represents simple test examples for heat transfer with melt front propagation.
 * The problem is inspired by the Proell et al. [1] single track scan example.
 *
 * The slab has properties of Ti-6Al-4V, is initially below the solidus temperature and is subjected
 * to a Gusarov laser heat source [2] at x = 0.
 *
 * [1] Proell, S. D., Wall, W. A., & Meier, C. (2019). On phase change and latent heat models in
 * metal additive manufacturing process simulation. Advanced Modeling and Simulation in Engineering
 * Sciences, 7(1), 1-32. http://arxiv.org/abs/1906.06238
 *
 * [2] Gusarov, A. V., Yadroitsev, I., Bertrand, P., & Smurov, I. (2009). Model of Radiation and
 * Heat Transfer in Laser-Powder Interaction Zone at Selective Laser Melting. Journal of Heat
 * Transfer, 131(7), 1-10. https://doi.org/10.1115/1.3109245
 */

namespace MeltPoolDG::Simulation::MeltFrontPropagation
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static constexpr double x_max = 0.6e-3;
  static constexpr double y_max = 0.2e-3;
  static constexpr double z_max = 0.2e-3;

  static constexpr double T_0 = 1000.0;

  template <int dim>
  class LaserHeatSourceFunction : public Function<dim>
  {
  public:
    LaserHeatSourceFunction(const LaserData<double> &laser_data)
      : Function<dim>()
      , center(MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(laser_data.center))
      , power(laser_data.power)
    {
      if (laser_data.heat_source_model == "Gusarov")
        laser_model = std::make_unique<Heat::LaserHeatSourceGusarov<dim>>(laser_data.gusarov);
      else if (laser_data.heat_source_model == "Gauss")
        laser_model = std::make_unique<Heat::LaserHeatSourceGauss<dim>>(laser_data.gauss);
      else
        AssertThrow(false, ExcMessage("Unknown laser heat source model! Abort..."));
    }

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      return laser_model->local_compute_volumetric_heat_source(p, center, power);
    }

  private:
    std::unique_ptr<Heat::LaserHeatSourceBase<dim>> laser_model;
    Point<dim>                                      center;
    const double                                    power;
  };

  template <int dim>
  class SimulationMeltFrontPropagation : public SimulationBase<dim>
  {
  public:
    SimulationMeltFrontPropagation(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      this->set_parameters();
    }

    void
    create_spatial_discretization() override
    {
      if constexpr (dim == 1)
        {
          this->triangulation = std::make_shared<parallel::shared::Triangulation<1>>(
            this->mpi_communicator,
            (Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
          // create mesh
          const Point<1> left(0);
          const Point<1> right(x_max);
          GridGenerator::hyper_rectangle(*this->triangulation, left, right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else if constexpr (dim == 2)
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<2>>(this->mpi_communicator);
          std::vector<unsigned> refinements(2, 1);
          refinements[0] = 3;
          // create mesh
          const Point<2> left(0, -z_max);
          const Point<2> right(x_max, 0);
          GridGenerator::subdivided_hyper_rectangle(*this->triangulation, refinements, left, right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else if constexpr (dim == 3)
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<3>>(this->mpi_communicator);
          std::vector<unsigned int> refinements(3, 1);
          refinements[0] = 3;
          refinements[1] = 3;
          // create mesh
          const Point<3> left(0, 0, 0);
          const Point<3> right(x_max, y_max, -z_max);
          GridGenerator::subdivided_hyper_rectangle(*this->triangulation, refinements, left, right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else
        AssertThrow(false, ExcMessage("Impossible dimension! Abort ..."));
    }

    void
    set_boundary_conditions() final
    {
      const types::boundary_id left_bc = 10;

      for (const auto &cell : this->triangulation->cell_iterators())
        {
          for (auto &face : cell->face_iterators())
            if (face->at_boundary())
              {
                if (face->center()[0] == x_max)
                  face->set_boundary_id(left_bc);
              }
        }

      this->attach_dirichlet_boundary_condition(
        left_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_0), "heat_transfer");
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(T_0),
                                     "heat_transfer");
      this->attach_source_field(
        std::make_shared<LaserHeatSourceFunction<dim>>(this->parameters.laser), "heat_transfer");
    }
  };
} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
