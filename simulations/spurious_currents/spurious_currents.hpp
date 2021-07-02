#pragma once

// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>

namespace MeltPoolDG::Simulation::SpuriousCurrents
{
  using namespace dealii;

  template <int dim>
  class InitialLevelSetCircle : public Function<dim>
  {
  public:
    InitialLevelSetCircle(const double eps)
      : Function<dim>()
      , eps(eps)
    {}
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const
    {
      (void)component;

      // set radius of bubble to 0.5, slightly shifted away from the center
      Point<dim> center;
      for (unsigned int d = 0; d < dim; ++d)
        center[d] = 0.02 + 0.01 * d;

      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        DistanceFunctions::spherical_manifold<dim>(p, center, 0.5), eps);

      // Alternative if no signed distance function should be used in the beginning:
      //
      // return UtilityFunctions::CharacteristicFunctions::sgn(
      // DistanceFunctions::spherical_manifold<dim>(p, center, 0.5));
    }

    double eps = 0.0;
  };

  template <int dim>
  class InitialLevelSetEllipse : public Function<dim>
  {
  public:
    InitialLevelSetEllipse(const double eps)
      : Function<dim>()
      , eps(eps)
    {}
    virtual double
    value(const Point<dim> &p, const unsigned int) const
    {
      Point<dim> center;
      for (unsigned int d = 0; d < dim; ++d)
        center[d] = 0.02 + 0.01 * d;

      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        DistanceFunctions::ellipsoidal_manifold<dim>(p, center, 0.75, 0.5), eps);
    }

    double eps = 0.0;
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationSpuriousCurrents : public SimulationBase<dim>
  {
  public:
    SimulationSpuriousCurrents(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("droplet shape",
                          droplet_shape,
                          "Shape of the droplet: circle or ellipse",
                          Patterns::Selection("circle|ellipse"));
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      if constexpr (dim == 2)
        {
          GridGenerator::hyper_cube(*this->triangulation, -2.5, 2.5);
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
      this->attach_dirichlet_boundary_condition(
        0, std::make_shared<Functions::ConstantFunction<dim>>(-1), "level_set");
      this->attach_no_slip_boundary_condition(0, "navier_stokes_u");
      this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");
    }

    void
    set_field_conditions() override
    {
      double eps =
        UtilityFunctions::compute_initial_epsilon<dim>(this->parameters, *this->triangulation);

      AssertThrow(eps > 0, ExcNotImplemented());

      if (droplet_shape == "circle")
        this->attach_initial_condition(std::make_shared<InitialLevelSetCircle<dim>>(eps),
                                       "level_set");
      else if (droplet_shape == "ellipse")
        this->attach_initial_condition(std::make_shared<InitialLevelSetEllipse<dim>>(eps),
                                       "level_set");
      else
        AssertThrow(false,
                    ExcMessage("Unknown droptlet shape: \"" + droplet_shape + "\"! Abort..."));
      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
    }

  private:
    std::string droplet_shape = "circle";
  };
} // namespace MeltPoolDG::Simulation::SpuriousCurrents
