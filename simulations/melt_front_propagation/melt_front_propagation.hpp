#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>


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

  template <int dim>
  class InitialLevelSet : public Function<dim>
  {
  public:
    InitialLevelSet(const double z_level, const double eps)
      : Function<dim>()
      , z_level(z_level)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      const auto z = p[dim - 1];
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(z_level - z,
                                                                                     eps);
    }

  private:
    const double z_level;
    const double eps;
  };

  template <int dim>
  class InitialLevelSetHeaviside : public Function<dim>
  {
  public:
    InitialLevelSetHeaviside(const double z_level, const double eps)
      : Function<dim>()
      , z_level(z_level)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      const auto z = p[dim - 1];
      return UtilityFunctions::CharacteristicFunctions::heaviside(z_level - z, eps);
    }

  private:
    const double z_level;
    const double eps;
  };

  template <int dim>
  class SimulationMeltFrontPropagation : public SimulationBase<dim>
  {
  private:
    double x_min        = 0.0;
    double x_max        = 0.0;
    double y_min        = 0.0;
    double y_max        = 0.0;
    double z_min        = 0.0;
    double z_max        = 0.0;
    double T_0          = 0.0;
    bool   do_two_phase = false;

  public:
    SimulationMeltFrontPropagation(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("domain x min", x_min, "minimum x coordinate of simulation domain");
        prm.add_parameter("domain x max", x_max, "maximum x coordinate of simulation domain");

        prm.add_parameter("domain z min", z_min, "minimum z coordinate of simulation domain");
        prm.add_parameter("domain z max", z_max, "maximum z coordinate of simulation domain");
        if constexpr (dim == 3)
          {
            prm.add_parameter("domain y min", y_min, "minimum y coordinate of simulation domain");
            prm.add_parameter("domain y max", y_max, "maximum y coordinate of simulation domain");
          }
        prm.add_parameter("initial temperature", T_0);
        prm.add_parameter("do two phase", do_two_phase);
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      AssertThrow(
        dim == 1 || x_min < x_max,
        ExcMessage("The upper bound of the domain must be greater than the lower bound! Abort..."));
      AssertThrow(
        z_min < z_max,
        ExcMessage("The upper bound of the domain must be greater than the lower bound! Abort..."));
      if constexpr (dim == 3)
        AssertThrow(
          y_min < y_max,
          ExcMessage(
            "The upper bound of the domain must be greater than the lower bound! Abort..."));

      if constexpr (dim == 1)
        {
#ifdef DEAL_II_WITH_METIS
          this->triangulation = std::make_shared<parallel::shared::Triangulation<1>>(
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
          // create mesh
          const Point<1> left(z_min);
          const Point<1> right(z_max);
          GridGenerator::hyper_rectangle(*this->triangulation, left, right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else if constexpr (dim == 2)
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<2>>(this->mpi_communicator);

          if (this->parameters.base.problem_name == ProblemType::heat_transfer && !do_two_phase)
            {
              std::vector<unsigned> refinements(2, 1);
              refinements[0] = 3;
              // create mesh
              const Point<2> left(0, -z_max);
              const Point<2> right(x_max, 0);
              GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        refinements,
                                                        left,
                                                        right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
          else
            {
              std::vector<unsigned> refinements(2, 1);
              refinements[0] = 3;
              refinements[1] = 2;
              // create mesh
              const Point<2> left(0, -z_max);
              const Point<2> right(x_max, z_max);
              GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        refinements,
                                                        left,
                                                        right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
        }
      else if constexpr (dim == 3)
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<3>>(this->mpi_communicator);

          if (this->parameters.base.problem_name == ProblemType::heat_transfer && !do_two_phase)
            {
              std::vector<unsigned int> refinements(3, 1);
              refinements[0] = 3;
              refinements[1] = 3;
              // create mesh
              const Point<3> left(0, -y_max, -z_max);
              const Point<3> right(x_max, y_max, z_max);
              GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        refinements,
                                                        left,
                                                        right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
          else
            {
              std::vector<unsigned int> refinements(3, 1);
              refinements[0] = 3;
              refinements[1] = 3;
              refinements[2] = 2;
              // create mesh
              const Point<3> left(0, 0, z_max);
              const Point<3> right(x_max, y_max, -z_max);
              GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        refinements,
                                                        left,
                                                        right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
        }
      else
        AssertThrow(false, ExcMessage("Impossible dimension! Abort ..."));
    }

    void
    set_boundary_conditions() final
    {
      if (this->parameters.base.problem_name == ProblemType::heat_transfer)
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
      else if (this->parameters.base.problem_name == ProblemType::melt_pool)
        {
          const types::boundary_id                  lower_bc = 1;
          const types::boundary_id                  upper_bc = 2;
          const types::boundary_id                  left_bc  = 3;
          const types::boundary_id                  right_bc = 4;
          [[maybe_unused]] const types::boundary_id front_bc = 5;
          [[maybe_unused]] const types::boundary_id back_bc  = 6;
          for (const auto &cell : this->triangulation->cell_iterators())
            for (auto &face : cell->face_iterators())
              if (face->at_boundary())
                {
                  if constexpr (dim == 1)
                    {
                      if (face->center()[0] == z_max)
                        face->set_boundary_id(lower_bc);
                      else if (face->center()[0] == z_min)
                        face->set_boundary_id(upper_bc);
                    }
                  else
                    {
                      if (face->center()[0] == x_max)
                        face->set_boundary_id(right_bc);
                      else if (face->center()[0] == x_min)
                        face->set_boundary_id(left_bc);

                      if constexpr (dim == 2)
                        {
                          if (face->center()[1] == -z_max)
                            face->set_boundary_id(lower_bc);
                          if (face->center()[1] == z_max)
                            face->set_boundary_id(upper_bc);
                        }
                      if constexpr (dim == 3)
                        {
                          if (face->center()[1] == -y_max)
                            face->set_boundary_id(lower_bc);
                          if (face->center()[1] == y_max)
                            face->set_boundary_id(upper_bc);
                          if (face->center()[2] == -z_max)
                            face->set_boundary_id(front_bc);
                          if (face->center()[2] == z_max)
                            face->set_boundary_id(back_bc);
                        }
                    }
                }
          this->attach_no_slip_boundary_condition(left_bc, "navier_stokes_u");
          this->attach_no_slip_boundary_condition(right_bc, "navier_stokes_u");
          this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
          this->attach_symmetry_boundary_condition(upper_bc, "navier_stokes_u");
          if constexpr (dim == 3)
            {
              this->attach_no_slip_boundary_condition(front_bc, "navier_stokes_u");
              this->attach_no_slip_boundary_condition(back_bc, "navier_stokes_u");
            }
          this->attach_fix_pressure_constant_condition(lower_bc, "navier_stokes_p");
          this->attach_dirichlet_boundary_condition(
            lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_0), "heat_transfer");
        }
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(T_0),
                                     "heat_transfer");
      if (this->parameters.base.problem_name == ProblemType::heat_transfer && do_two_phase)
        this->attach_initial_condition(std::make_shared<InitialLevelSetHeaviside<dim>>(0.0,
                                                                                       z_max / 5),
                                       "prescribed_heaviside");
      if (this->parameters.base.problem_name == ProblemType::melt_pool)
        {
          const double eps =
            this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
              GridTools::minimal_cell_diameter(*this->triangulation) /
              this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));
          this->attach_initial_condition(std::make_shared<InitialLevelSet<dim>>(0.0, eps),
                                         "level_set");
          this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                           std::make_shared<Functions::ZeroFunction<dim>>(dim)),
                                         "navier_stokes_u");
        }
    }
  };
} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
