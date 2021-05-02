#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>

#include <deal.II/grid/grid_generator.h>

// MeltPoolDG
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

// c++
#include <cmath>
#include <iostream>

/**
 * This example is derived from
 *
 * Meier et al. 2020
 * "A novel smoothed particle hydrodynamics formulation
 * for thermo-capillary phase change problems with focus on metal additive
 * manufacturing melt pool modeling"
 *
 * and
 *
 * C. Ma, D. Bothe 2011, "Direct numerical simulation of thermocapillary flow
 * based on the Volume of Fluid method"
 *
 *                 T2 = fixed
 *                  no slip
 *            +-------------------+
 *            |                   |
 *            |                   |
 *            |       -----       |
 *   sym      |      --   --      |   sym
 * T Neumann  |     --     --     |   T Neumann
 *            |      --   --      |
 *            |       -----       |
 *            |                   |
 *            +-------------------+
 *                   no slip
 *                  T1 = fixed
 *
 * droplet:
 *    rho    = 250 kg/m³
 *    mu     = 0.012 N/m²s
 *    lambda = 1.2e-6 W/m/K
 *    cp     = 5e-5 J/kg/K
 *
 * ambient fluid:
 *    rho    = 500 kg/m³
 *    mu     = 0.024 N/m²s
 *    lambda = 2.4e-6 W/m/K
 *    cp     = 1e-4 J/kg/K
 *
 * droplet radius: a=1.44e-3 m
 * surface tension coefficient: 0.01 N/m
 * temperature-dependent surface tension coefficient: 0.002 N/m/K
 * T1 = 290 K
 * T2 = 290 + 4 * a * 200 K/m = 291.152 K
 */

namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;


  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const double eps)
      : Function<dim>()
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      Point<dim> center = dim == 2 ? Point<dim>(0, 0) : Point<dim>(0, 0, 0);

      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, a), eps);
    }
    const double a = 1.44e-3;

    double eps;
  };


  template <int dim>
  class InitialValuesTemperature : public Function<dim>
  {
  public:
    InitialValuesTemperature()
      : Function<dim>()
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      return 290 + 200 * (p[dim - 1] + 2 * a);
    }
    const double a = 1.44e-3;
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationThermoCapillaryDroplet : public SimulationBase<dim>
  {
  public:
    SimulationThermoCapillaryDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      this->set_parameters();
    }

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.do_simplex)
        {
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (::Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      if constexpr ((dim == 2) || (dim == 3))
        {
          // create mesh
          const Point<dim> bottom_left =
            (dim == 2) ? Point<dim>(x_min, x_min) : Point<dim>(x_min, x_min, x_min);
          const Point<dim> top_right =
            (dim == 2) ? Point<dim>(x_max, x_max) : Point<dim>(x_max, x_max, x_max);

          if (this->parameters.base.do_simplex)
            {
              // create mesh
              std::vector<unsigned int> subdivisions(
                dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
              subdivisions[dim - 1] *= 2;

              GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                       subdivisions,
                                                                       bottom_left,
                                                                       top_right);
            }
          else
            {
              GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const types::boundary_id lower_bc = 1;
      const types::boundary_id upper_bc = 2;
      const types::boundary_id left_bc  = 3;
      const types::boundary_id right_bc = 4;

      this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
      this->attach_no_slip_boundary_condition(upper_bc, "navier_stokes_u");
      this->attach_symmetry_boundary_condition(left_bc, "navier_stokes_u");
      this->attach_symmetry_boundary_condition(right_bc, "navier_stokes_u");

      this->attach_dirichlet_boundary_condition(lower_bc,
                                                std::make_shared<InitialValuesTemperature<dim>>(),
                                                "heat_transfer");
      this->attach_dirichlet_boundary_condition(upper_bc,
                                                std::make_shared<InitialValuesTemperature<dim>>(),
                                                "heat_transfer");

      if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[1] == x_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[1] == x_max)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == x_min)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
                }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    void
    set_field_conditions() final
    {
      double eps = 0.0;
      if (this->parameters.reinit.implementation == "adaflo" ||
          this->parameters.ls.implementation == "adaflo")
        eps = this->parameters.reinit.constant_epsilon > 0.0 ?
                this->parameters.reinit.constant_epsilon :
                GridTools::minimal_cell_diameter(*this->triangulation) /
                  this->parameters.base.degree / std::sqrt(dim);
      else
        eps = this->parameters.reinit.constant_epsilon > 0.0 ?
                this->parameters.reinit.constant_epsilon :
                GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim) *
                  this->parameters.reinit.scale_factor_epsilon;

      AssertThrow(eps > 0, ExcNotImplemented());

      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps), "level_set");
      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(),
                                     "heat_transfer");
    }

  private:
    const double a     = 1.44e-3;
    const double x_min = -2 * a;
    const double x_max = 2 * a;
  };
} // namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
