#pragma once

// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>

namespace MeltPoolDG
{
  namespace Simulation::EvaporatingDropletWithHeat
  {
    using namespace dealii;

    template <int dim>
    class InitialValuesLS : public Function<dim>
    {
    public:
      InitialValuesLS(const double eps)
        : Function<dim>()
        , distance_sphere(dim == 2 ? Point<dim>(0, 0) : Point<dim>(0, 0, 0), 0.02)
        , eps(eps)
      {}

      double
      value(const Point<dim> &p, const unsigned int /*component*/) const
      {
        return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
          -distance_sphere.value(p), eps);
      }

    private:
      const Functions::SignedDistance::Sphere<dim> distance_sphere;
      const double                                 eps;
    };

    /*
     *      This class collects all relevant input data for the level set simulation
     */

    template <int dim>
    class SimulationEvaporatingDropletWithHeat : public SimulationBase<dim>
    {
    public:
      SimulationEvaporatingDropletWithHeat(std::string    parameter_file,
                                           const MPI_Comm mpi_communicator)
        : SimulationBase<dim>(parameter_file, mpi_communicator)
        , lambda(2. * numbers::PI *
                 std::sqrt(3. * this->parameters.surface_tension.surface_tension_coefficient /
                           (9.81 * (this->parameters.material.second.density -
                                    this->parameters.material.first.density))))
      {}

      void
      create_spatial_discretization() override
      {
        if (this->parameters.base.do_simplex)
          {
            this->triangulation =
              std::make_shared<parallel::shared::Triangulation<dim>>(this->mpi_communicator);
          }
        else
          {
            this->triangulation =
              std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
          }

        if constexpr ((dim == 2) || (dim == 3))
          {
            // create mesh
            std::vector<unsigned int> subdivisions(
              dim,
              5 * (this->parameters.base.do_simplex ?
                     Utilities::pow(2, this->parameters.base.global_refinements) :
                     1));

            const Point<dim> top_right   = (dim == 2) ?
                                             Point<dim>(lambda / 2., lambda / 2.) :
                                             Point<dim>(lambda / 2., lambda / 2., lambda / 2.);
            const Point<dim> bottom_left = (dim == 2) ?
                                             Point<dim>(-lambda / 2., -lambda / 2.) :
                                             Point<dim>(-lambda / 2., -lambda / 2., -lambda / 2.);

            if (this->parameters.base.do_simplex)
              {
                GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                         subdivisions,
                                                                         bottom_left,
                                                                         top_right);
              }
            else
              {
                GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                          subdivisions,
                                                          bottom_left,
                                                          top_right);
              }

            if (this->parameters.base.do_simplex == false)
              this->triangulation->refine_global(this->parameters.base.global_refinements);
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }
      }

      void
      set_boundary_conditions() override
      {
        auto dirichlet_temp = std::make_shared<Functions::ConstantFunction<dim>>(10);

        this->attach_dirichlet_boundary_condition(0, dirichlet_temp, "heat_transfer");
        this->attach_open_boundary_condition(0, "navier_stokes_u");
      }

      void
      set_field_conditions() override
      {
        double eps =
          UtilityFunctions::compute_initial_epsilon<dim>(this->parameters, *this->triangulation);

        AssertThrow(eps > 0, ExcNotImplemented());

        this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps), "level_set");
        this->attach_initial_condition(
          std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");
        this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                         new Functions::ConstantFunction<dim>(0.0)),
                                       "heat_transfer");
      }

      const double lambda;
    };

  } // namespace Simulation::EvaporatingDropletWithHeat
} // namespace MeltPoolDG
