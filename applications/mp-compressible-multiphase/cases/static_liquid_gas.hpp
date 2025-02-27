#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/utility_functions.hpp>

#include "../compressible_multiphase_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  template <int dim>
  class FlowField : public Function<dim>
  {
  public:
    explicit FlowField()
      : Function<dim>(dim + 2)
    {}

    double
    value(const Point<dim> &, const unsigned int component) const final
    {
      if (component == 0)
        return 1.;
      else if (component == 3)
        return 1.;
      else if (component == 4)
        return 0.;
      else if (component == 1)
        return 0.;
      else if (component == 2)
        return 1. * 1004.85 / 1.4 * 293.15;
      else if (component == 5)
        return 1. * 1004.85 / 1.4 * 293.15;
      else
        return 0.;
    }
  };

  template <int dim>
  class SimulationStaticLiquidGas final : public Multiphase::CompressibleMultiphaseCase<dim>
  {
  public:
    SimulationStaticLiquidGas(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Multiphase::CompressibleMultiphaseCase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      //TODO: no distributed triangulation possible for d=1
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
      subdivisions[0] = 100;
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
      auto inflow_outflow_solution = std::make_shared<FlowField<dim>>();
      auto dummy_solution          = std::make_shared<FlowField<dim>>();
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
      auto initial_condition = std::make_shared<FlowField<dim>>();
      this->attach_initial_condition(initial_condition, "compressible_multiphase");

      // set level-set function
      Point<dim> p;
      // avoid phase interface colliding with element face (bug in dealii has to be fixed)
      p[0] = 0.1548467;

      dealii::Tensor<1, dim> normal;
      normal[0] = -1.;

      const auto level_set =
        std::make_shared<Functions::/*ConstantFunction<dim,double>>(1.);*/SignedDistance::Plane<dim>>(p, normal);
      this->attach_field_function(level_set, "level_set", "compressible_multiphase");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const override
    {
      FlowField<dim> reference_values;
      this->print_relative_norm(generic_data_out, reference_values, "Norm");
    }

  private:
    // for self-registration
    static SimulationCaseRegistrar<Multiphase::CompressibleMultiphaseCase<dim>> registrar;

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
                if (center(0) < -5.+1e-12)
                  face->set_boundary_id(BoundaryID::inflow);
              }
          }
    }
  };

  // for self-registration
  template <int dim>
  SimulationCaseRegistrar<Multiphase::CompressibleMultiphaseCase<dim>>
    SimulationStaticLiquidGas<dim>::registrar(
      "static_liquid_gas",
      [](const std::string &parameter_file, const MPI_Comm mpi_communicator) {
        return std::make_unique<SimulationStaticLiquidGas<dim>>(parameter_file,
                                                                    mpi_communicator);
      });
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
