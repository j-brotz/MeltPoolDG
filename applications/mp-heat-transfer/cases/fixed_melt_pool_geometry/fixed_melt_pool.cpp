#include "fixed_melt_pool.hpp"
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include "meltpooldg/level_set/level_set_tools.hpp"
#include <meltpooldg/core/case_registration.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <algorithm>
#include <memory>


namespace MeltPoolDG::Simulation::FixedMeltPool
{
  template <int dim, typename number>
  FixedMeltPoolGeometry<dim, number>::FixedMeltPoolGeometry(
    const LevelSet::LevelSetType level_set_type,
    const number                 eps)
    : dealii::Function<dim>()
    , level_set_type(level_set_type)
    , eps(eps)
  {
    AssertThrow(dim == 2, dealii::ExcNotImplemented());
  }

  template <int dim, typename number>
  number
  FixedMeltPoolGeometry<dim, number>::value(const dealii::Point<dim> &p,
                                            const unsigned int /*component*/) const
  {
    number     signed_distance;
    const auto inside_melt_pool = is_inside_melt_pool(p);
    if (p[1] >= centre[1]) // above centre line
      {
        if (inside_melt_pool)
          signed_distance = signed_distance_beads(p);
        else // not inside melt pool
          signed_distance = signed_distance_level(p);
      }
    else // below center line
      {
        if (inside_melt_pool)
          signed_distance = signed_distance_pool(p);
        else // not inside melt pool
          signed_distance = std::min(signed_distance_pool(p), signed_distance_level(p));
      }

    switch (level_set_type)
      {
        case LevelSet::LevelSetType::tanh:
          return CharacteristicFunctions::tanh_characteristic_function(signed_distance, eps);
        case LevelSet::LevelSetType::smoothed_heaviside:
          return CharacteristicFunctions::smoothed_heaviside(signed_distance, eps);
        case LevelSet::LevelSetType::signed_distance:
          return signed_distance;
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
    // unreachable dummy return
    return 0.0;
  }

  template <int dim, typename number>
  number
  FixedMeltPoolGeometry<dim, number>::signed_distance_level(const dealii::Point<dim> &p) const
  {
    Assert(not is_inside_melt_pool(p), dealii::ExcMessage("can only apply outside melt pool"));
    return -p[1] + y_level;
  }

  template <int dim, typename number>
  number
  FixedMeltPoolGeometry<dim, number>::signed_distance_pool(const dealii::Point<dim> &p) const
  {
    Assert(p[1] <= centre[1], dealii::ExcMessage("can only apply below center line"));
    return p.distance(centre) - centre_radius;
  }

  template <int dim, typename number>
  number
  FixedMeltPoolGeometry<dim, number>::signed_distance_beads(const dealii::Point<dim> &p) const
  {
    Assert(is_inside_melt_pool(p), dealii::ExcMessage("can only apply inside melt pool"));
    Assert(p[1] >= centre[1], dealii::ExcMessage("can only apply above center line"));
    const dealii::Point<dim> p_proj(std::abs(p[0] - centre[0]), p[1]);
    const dealii::Point<dim> bead_center_proj(bead_center_x - centre[0], centre[1]);
    return bead_radius - p_proj.distance(bead_center_proj);
  }

  template <int dim, typename number>
  bool
  FixedMeltPoolGeometry<dim, number>::is_inside_melt_pool(const dealii::Point<dim> &p) const
  {
    return std::abs(p[0] - centre[0]) <= bead_center_x;
  }

  template <int dim, typename number>
  SimulationFixedMeltPool<dim, number>::SimulationFixedMeltPool(std::string    parameter_file,
                                                                const MPI_Comm mpi_communicator)
    : Heat::HeatTransferCase<dim, number>(parameter_file, mpi_communicator)
  {}

  template <int dim, typename number>
  bool
  SimulationFixedMeltPool<dim, number>::add_simulation_specific_parameters(
    dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("case specific");
    {
      prm.add_parameter("grid filename", grid_file, "optional grid input file");
      prm.add_parameter("interface thickness",
                        interface_thickness,
                        "thickness of the diffuse interface transition region");
    }
    prm.leave_subsection();

    return this->parameters.base.do_print_parameters;
  }

  template <int dim, typename number>
  void
  SimulationFixedMeltPool<dim, number>::create_spatial_discretization()
  {
    this->triangulation =
      std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

    if (not grid_file.empty())
      dealii::GridIn<dim>(*this->triangulation).read(grid_file);
    else
      {
        AssertThrow(dim == 2, dealii::ExcNotImplemented());

        const dealii::Point<dim> bottom_left = dealii::Point<dim>(-a, -a);
        const dealii::Point<dim> top_right   = dealii::Point<dim>(a, a);

        dealii::GridGenerator::hyper_rectangle(*this->triangulation,
                                               bottom_left,
                                               top_right,
                                               true /* colorize */);
      }

    this->triangulation->refine_global(this->parameters.base.global_refinements);
  }

  template <int dim, typename number>
  void
  SimulationFixedMeltPool<dim, number>::set_boundary_conditions()
  {
    [[maybe_unused]] const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
      get_colorized_rectangle_boundary_ids<dim>();

    const auto T_bc = std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(T_hat);
    this->attach_boundary_condition({upper_bc, T_bc}, "dirichlet", "heat_transfer");
    this->attach_boundary_condition({lower_bc, T_bc}, "dirichlet", "heat_transfer");
  }

  template <int dim, typename number>
  void
  SimulationFixedMeltPool<dim, number>::set_field_conditions()
  {
    // by default, set the interface thickness to 32 element side lengths
    if (interface_thickness == numbers::invalid_double)
      {
        const auto h =
          2 * a /
          std::pow(2,
                   this->parameters.amr.do_amr ? this->parameters.amr.max_grid_refinement_level :
                                                 this->parameters.base.global_refinements);
        interface_thickness = 32 * h;
      }

    this->attach_initial_condition(std::make_shared<FixedMeltPoolGeometry<dim, number>>(
                                     LevelSet::LevelSetType::smoothed_heaviside,
                                     interface_thickness / 6.),
                                   "prescribed_heaviside");
    this->attach_initial_condition(std::make_shared<FixedMeltPoolGeometry<dim, number>>(
                                     LevelSet::LevelSetType::signed_distance),
                                   "prescribed_signed_distance");

    this->attach_initial_condition(
      std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(T_hat), "heat_transfer");
  }

  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase,
                           SimulationFixedMeltPool,
                           "fixed_melt_pool",
                           2,
                           double);
} // namespace MeltPoolDG::Simulation::FixedMeltPool
