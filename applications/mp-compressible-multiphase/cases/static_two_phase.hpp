/**
 * @brief Simulation of (currently) one-dimensional two-phase flows.
 * The initial conditions are used from the user input file.
 * There is an inflow boundary for the liquid phase and an outflow boundary with fixed total energy
 * for the gas phase.
 *       _____________________________________
 *    ->|      liquid      |        gas      | ->
 *     x=-5              x=0.13             x=5
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include "../compressible_multiphase_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  template <int dim>
  class FlowField : public Function<dim>
  {
  public:
    explicit FlowField(std::string ic_gas_phase,
                       std::string ic_liquid_phase,
                       const bool  gas_phase_is_first = true)
      : Function<dim>(2 * (dim + 2))
      , gas_phase_is_first(gas_phase_is_first)
    {
      parsing_function_gas->initialize(dim == 3 ?
                                         std::string("x,y,z") :
                                         (dim == 2 ? std::string("x,y") : std::string("x")),
                                       ic_gas_phase,
                                       constants,
                                       false);
      parsing_function_liquid->initialize(dim == 3 ?
                                            std::string("x,y,z") :
                                            (dim == 2 ? std::string("x,y") : std::string("x")),
                                          ic_liquid_phase,
                                          constants,
                                          false);

      // Currently, only 1D simulations are possible
      Assert(dim == 1, ExcNotImplemented());
    }

    double
    value(const Point<dim> &p, const unsigned int component) const final
    {
      unsigned int gas_components[dim + 2];
      unsigned int liquid_components[dim + 2];

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

      // gas phase
      if (component == gas_components[0])
        return parsing_function_gas->value(p, 0);
      else if (component == gas_components[1])
        return parsing_function_gas->value(p, 1);
      else if (component == gas_components[2])
        return parsing_function_gas->value(p, 2);

      // liquid phase
      else if (component == liquid_components[0])
        return parsing_function_liquid->value(p, 0);
      else if (component == liquid_components[1])
        return parsing_function_liquid->value(p, 1);
      else if (component == liquid_components[2])
        return parsing_function_liquid->value(p, 2);
      else
        return 0.;
    }

  private:
    bool                                         gas_phase_is_first;
    std::map<std::string, double>                constants;
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_gas =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_liquid =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);
  };

  template <int dim>
  class SimulationStaticTwoPhase final : public Multiphase::CompressibleMultiphaseCase<dim>
  {
  public:
    SimulationStaticTwoPhase(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Multiphase::CompressibleMultiphaseCase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      // TODO: no distributed triangulation possible for dim=1
      this->triangulation =
        std::make_shared<parallel::shared::Triangulation<dim>>(this->mpi_communicator);

      Point<dim> lower_left;
      lower_left[0] = -5.;
      for (unsigned int d = 1; d < dim; ++d)
        lower_left[d] = 0.;

      Point<dim> upper_right;
      upper_right[0] = 5.;
      for (unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 0.;

      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 20;
      for (unsigned int d = 1; d < dim; ++d)
        subdivisions[d] = 1;

      GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                subdivisions,
                                                lower_left,
                                                upper_right);

      set_fitted_boundary_id(*this->triangulation);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      auto inflow_outflow_solution =
        std::make_shared<FlowField<dim>>(ic_gas_phase, ic_liquid_phase);
      auto dummy_solution = std::make_shared<FlowField<dim>>(ic_gas_phase, ic_liquid_phase);
      this->attach_boundary_condition({BoundaryID::inflow, inflow_outflow_solution},
                                      "inflow",
                                      "compressible_multiphase");
      this->attach_boundary_condition({BoundaryID::subsonic_outflow_with_fixed_energy,
                                       inflow_outflow_solution},
                                      "outflow_fixed_energy",
                                      "compressible_multiphase");
    }

    void
    set_field_conditions() override
    {
      // The solution vector is ordered, such that the liquid phase is the first phase and the gas
      // phase is the second phase.
      auto initial_condition = std::make_shared<FlowField<dim>>(ic_gas_phase,
                                                                ic_liquid_phase,
                                                                false /*gas_phase_is_first*/);
      this->attach_initial_condition(initial_condition, "compressible_multiphase");

      // set level-set function
      Point<dim> p;
      // avoid phase interface colliding with element face (bug in dealii has to be fixed)
      p[0] = 0.13;

      dealii::Tensor<1, dim> normal;
      normal[0] = -1.;

      const auto level_set = std::make_shared<Functions::SignedDistance::Plane<dim>>(p, normal);
      this->attach_field_function(level_set, "level_set", "compressible_multiphase");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const override
    {
      FlowField<dim> reference_values(ic_gas_phase, ic_liquid_phase);
      this->print_relative_norm(generic_data_out, reference_values, "Norm");
    }

  private:
    // initial conditions functions
    std::string ic_gas_phase    = this->parameters.flow.initial_conditions_gas_phase;
    std::string ic_liquid_phase = this->parameters.flow.initial_conditions_liquid_phase;

    /**
     * Enumeration for the fitted boundary id's.
     */
    enum BoundaryID
    {
      inflow,
      subsonic_outflow_with_fixed_energy
    };

    /**
     * Set boundary id's for fitted boundaries
     */
    void
    set_fitted_boundary_id(const auto &triangulation) const
    {
      for (const auto &cell : triangulation.cell_iterators())
        if (cell->at_boundary())
          {
            for (const auto &face : cell->face_iterators())
              {
                const auto center = face->center();
                // outflow boundary with fixed energy
                if (center(0) > 5. - 1e-12)
                  face->set_boundary_id(BoundaryID::subsonic_outflow_with_fixed_energy);
                if (center(0) < -5. + 1e-12)
                  face->set_boundary_id(BoundaryID::inflow);
              }
          }
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
