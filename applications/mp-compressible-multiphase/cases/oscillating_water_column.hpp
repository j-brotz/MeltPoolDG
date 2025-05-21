/**
 * @brief Simulation of an oscillating water column inside two compressible air columns.
 *
 * This test case is taken from:
 * Duret et al., 2018: "A pressure based method for vaporizing compressible two-phase flows with
 * interface capturing approach."
 *
 * Adiabatic no-slip wall boundary conditions are applied.
 *
 *       ___________________________________
 *      |    air   |   water    |    air   |
 *     x=0         <----------->          x=1
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>

#include "../compressible_multiphase_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  template <int dim, typename number>
  class NegatedFunction : public dealii::Function<dim, number>
  {
  public:
    explicit NegatedFunction(std::shared_ptr<const dealii::Function<dim, number>> func)
      : dealii::Function<dim, number>(func->n_components)
      , func(func)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int component = 0) const final
    {
      return -func->value(p, component);
    }

  private:
    std::shared_ptr<const dealii::Function<dim, number>> func;
  };

  template <int dim, typename number>
  class InitialFieldOscillatingWaterColumn : public dealii::Function<dim, number>
  {
  public:
    explicit InitialFieldOscillatingWaterColumn(const auto &liquid_material_data,
                                                const auto &gas_material_data,
                                                const bool  gas_phase_is_first = true)
      : dealii::Function<dim, number>(2 * (dim + 2))
      , liquid_material_data(liquid_material_data)
      , gas_material_data(gas_material_data)
      , gas_phase_is_first(gas_phase_is_first)
    {
      // Currently, only 1D simulations are possible
      Assert(dim == 1, dealii::ExcNotImplemented());
    }

    number
    value(const dealii::Point<dim, number> &p, const unsigned int component) const final
    {
      std::array<unsigned int, dim + 2> gas_components;
      std::array<unsigned int, dim + 2> liquid_components;

      for (unsigned int i = 0; i < dim + 2; ++i)
        {
          if (gas_phase_is_first)
            {
              gas_components[i]    = i;
              liquid_components[i] = i + dim + 2;
            }
          else
            {
              gas_components[i]    = i + dim + 2;
              liquid_components[i] = i;
            }
        }

      // initial phase interface positions
      constexpr number R_1 = 0.10001;
      constexpr number R_2 = 0.80001;

      // an initial pressure gradient in the liquid phase triggers the movement of the water column
      constexpr number p_r = 0.5e5;
      constexpr number p_l = 1.e5;
      constexpr number T   = 293.15;

      // gas phases

      if (component == gas_components[0])
        {
          if (p[0] <= 0.5)
            return p_l / (gas_material_data.specific_gas_constant * T);
          else
            return p_r / (gas_material_data.specific_gas_constant * T);
        }
      else if (component == gas_components[1])
        return 0.;
      else if (component == gas_components[2])
        if (p[0] <= 0.5)
          return p_l / (gas_material_data.gamma - 1.0);
        else
          return p_r / (gas_material_data.gamma - 1.0);

      // liquid phase

      const number pressure =
        (p_r - p_l) / (R_2 - R_1) * p[0] + (p_l * R_2 - p_r * R_1) / (R_2 - R_1);

      if (component == liquid_components[0])
        {
          return (pressure + liquid_material_data.eos_data.p_inf) /
                 ((liquid_material_data.gamma - 1.) * liquid_material_data.specific_isobaric_heat /
                  liquid_material_data.gamma * T);
        }
      else if (component == liquid_components[1])
        return 0.;
      else if (component == liquid_components[2])
        {
          return (pressure + liquid_material_data.gamma * liquid_material_data.eos_data.p_inf) /
                 (liquid_material_data.gamma - 1.);
        }
      else
        return 0.;
    }

  private:
    Flow::CompressibleFluidMaterialPhaseData<number> liquid_material_data;
    Flow::CompressibleFluidMaterialPhaseData<number> gas_material_data;
    bool                                             gas_phase_is_first;
  };

  template <int dim, typename number>
  class SimulationOscillatingWaterColumn final
    : public Multiphase::CompressibleMultiphaseCase<dim, number>
  {
  public:
    SimulationOscillatingWaterColumn(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Multiphase::CompressibleMultiphaseCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      // TODO: no distributed triangulation possible for dim=1
      this->triangulation =
        std::make_shared<dealii::parallel::shared::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim, number> lower_left;
      lower_left[0] = 0.;
      for (unsigned int d = 1; d < dim; ++d)
        lower_left[d] = 0.;

      dealii::Point<dim, number> upper_right;
      upper_right[0] = 1.;
      for (unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 0.;

      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 20;
      for (unsigned int d = 1; d < dim; ++d)
        subdivisions[d] = 1;

      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation, subdivisions, lower_left, upper_right, true);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      auto inflow_outflow_solution =
        std::make_shared<InitialFieldOscillatingWaterColumn<dim, number>>(
          this->parameters.flow.material.liquid, this->parameters.flow.material.gas);

      // face numbering according to the deal.II colorize flag
      const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      // left boundary
      this->attach_boundary_condition({lower_bc, inflow_outflow_solution},
                                      "no_slip_wall",
                                      "compressible_multiphase_flow");
      // right boundary
      this->attach_boundary_condition({upper_bc, inflow_outflow_solution},
                                      "no_slip_wall",
                                      "compressible_multiphase_flow");
    }

    void
    set_field_conditions() override
    {
      // The solution vector is ordered, such that the liquid phase is the first phase and the gas
      // phase is the second phase.
      auto initial_condition = std::make_shared<InitialFieldOscillatingWaterColumn<dim, number>>(
        this->parameters.flow.material.liquid,
        this->parameters.flow.material.gas,
        false /*gas_phase_is_first*/);
      this->attach_initial_condition(initial_condition, "compressible_multiphase_flow");

      // set level-set function
      dealii::Point<dim> p;
      // avoid phase interface colliding with element face (bug in dealii has to be fixed)
      p[0] = 0.45001;

      constexpr number radius = 0.35;

      const auto inverse_level_set =
        std::make_shared<dealii::Functions::SignedDistance::Sphere<dim>>(p, radius);
      // level-set must be inverted for correct orientation
      const auto level_set = std::make_shared<NegatedFunction<dim, number>>(inverse_level_set);
      this->attach_field_function(level_set, "level_set", "compressible_multiphase_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      InitialFieldOscillatingWaterColumn<dim, number> reference_values(
        this->parameters.flow.material.liquid, this->parameters.flow.material.gas);
      this->print_relative_norm(generic_data_out, reference_values, "norm");
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
