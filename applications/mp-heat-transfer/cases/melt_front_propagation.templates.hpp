#include "melt_front_propagation.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/case_registration.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <cmath>
#include <vector>

#include "../../mp-melt-pool/melt_pool_case.hpp"


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

  BETTER_ENUM(LevelSetType, char, level_set, heaviside, signed_distance)

  template <int dim, typename number>
  class InitialLevelSet : public Function<dim, number>
  {
  public:
    InitialLevelSet(const number z_level, const LevelSetType level_set_type, const number eps)
      : Function<dim, number>()
      , z_level(z_level)
      , level_set_type(level_set_type)
      , eps(eps)
    {}

    number
    value(const Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const auto signed_distance = z_level - p[dim - 1];

      switch (level_set_type)
        {
          case LevelSetType::level_set:
            return CharacteristicFunctions::tanh_characteristic_function(signed_distance, eps);
          case LevelSetType::heaviside:
            return CharacteristicFunctions::smoothed_heaviside(signed_distance, eps);
          case LevelSetType::signed_distance:
            return signed_distance;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      // unreachable dummy return
      return 0.0;
    }

  private:
    const number       z_level;
    const LevelSetType level_set_type;
    const number       eps;
  };


  template <int dim, typename number, typename CaseClass>
  SimulationMeltFrontPropagation<dim, number, CaseClass>::SimulationMeltFrontPropagation(
    std::string    parameter_file,
    const MPI_Comm mpi_communicator)
    : CaseClass(parameter_file, mpi_communicator)
  {}


  template <int dim, typename number, typename CaseClass>
  bool
  SimulationMeltFrontPropagation<dim, number, CaseClass>::add_simulation_specific_parameters(
    dealii::ParameterHandler &prm)
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

    return this->parameters.base.do_print_parameters;
  }


  template <int dim, typename number, typename CaseClass>
  void
  SimulationMeltFrontPropagation<dim, number, CaseClass>::create_spatial_discretization()
  {
    AssertThrow(dim == 1 || x_min < x_max,
                ExcMessage(
                  "The upper bound of the domain must be greater than the lower bound! Abort..."));
    AssertThrow(z_min < z_max,
                ExcMessage(
                  "The upper bound of the domain must be greater than the lower bound! Abort..."));
    if constexpr (dim == 3)
      AssertThrow(
        y_min < y_max,
        ExcMessage("The upper bound of the domain must be greater than the lower bound! Abort..."));

    if constexpr (dim == 1)
      {
#ifdef DEAL_II_WITH_METIS
        this->triangulation = std::make_shared<parallel::shared::Triangulation<1>>(
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

        if constexpr (std::is_same_v<CaseClass, Heat::HeatTransferCase<dim, number>>)
          {
            if (not do_two_phase)
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

        if constexpr (std::is_same_v<CaseClass, Heat::HeatTransferCase<dim, number>>)
          {
            if (not do_two_phase)
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


  template <int dim, typename number, typename CaseClass>
  void
  SimulationMeltFrontPropagation<dim, number, CaseClass>::set_boundary_conditions()
  {
    if constexpr (std::is_same_v<CaseClass, Heat::HeatTransferCase<dim, number>>)
      {
        const types::boundary_id right_bc = 10;
        for (const auto &cell : this->triangulation->cell_iterators())
          {
            for (auto &face : cell->face_iterators())
              if (face->at_boundary())
                {
                  if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
                }
          }
        this->attach_boundary_condition({right_bc,
                                         std::make_shared<Functions::ConstantFunction<dim>>(T_0)},
                                        "dirichlet",
                                        "heat_transfer");
      }
    else if constexpr (std::is_same_v<CaseClass, MeltPoolDG::MeltPoolCase<dim, number>>)
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
        this->attach_boundary_condition(left_bc, "no_slip", "navier_stokes_u");
        this->attach_boundary_condition(right_bc, "no_slip", "navier_stokes_u");
        this->attach_boundary_condition(lower_bc, "no_slip", "navier_stokes_u");
        this->attach_boundary_condition(upper_bc, "symmetry", "navier_stokes_u");
        if constexpr (dim == 3)
          {
            this->attach_boundary_condition(front_bc, "no_slip", "navier_stokes_u");
            this->attach_boundary_condition(back_bc, "no_slip", "navier_stokes_u");
          }
        this->attach_boundary_condition(lower_bc, "fix_pressure_constant", "navier_stokes_p");
        this->attach_boundary_condition({lower_bc,
                                         std::make_shared<Functions::ConstantFunction<dim>>(T_0)},
                                        "dirichlet",
                                        "heat_transfer");
      }
  }


  template <int dim, typename number, typename CaseClass>
  void
  SimulationMeltFrontPropagation<dim, number, CaseClass>::set_field_conditions()
  {
    this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(T_0),
                                   "heat_transfer");

    if constexpr (std::is_same_v<CaseClass, Heat::HeatTransferCase<dim, number>>)
      if (do_two_phase)
        {
          if (this->parameters.heat.operator_type != Heat::TwoPhaseOperatorType::cut)
            {
              this->attach_initial_condition(std::make_shared<InitialLevelSet<dim, number>>(
                                               0.0, LevelSetType::heaviside, z_max / 5),
                                             "prescribed_heaviside");
            }
          else
            {
              const number offset = 6e-6;
              this->attach_initial_condition(std::make_shared<InitialLevelSet<dim, number>>(
                                               offset, LevelSetType::signed_distance, 0.0),
                                             "prescribed_signed_distance");
              if (this->parameters.amr.do_amr and
                  this->parameters.application_specific_parameters.amr_strategy ==
                    Heat::AMRStrategy::generic)
                this->attach_initial_condition(std::make_shared<InitialLevelSet<dim, number>>(
                                                 offset, LevelSetType::heaviside, z_max / 4),
                                               "prescribed_heaviside");
            }
        }

    if constexpr (std::is_same_v<CaseClass, MeltPoolDG::MeltPoolCase<dim, number>>)
      {
        const number eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
          GridTools::minimal_cell_diameter(*this->triangulation) /
          this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));
        this->attach_initial_condition(
          std::make_shared<InitialLevelSet<dim, number>>(0.0, LevelSetType::level_set, eps),
          "level_set");
        this->attach_initial_condition(std::shared_ptr<Function<dim, number>>(
                                         std::make_shared<Functions::ZeroFunction<dim, number>>(
                                           dim)),
                                       "navier_stokes_u");
      }
  }
} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
