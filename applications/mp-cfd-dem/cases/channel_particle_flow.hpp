#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include "../cfd_dem_case.hpp"

namespace MeltPoolDG::Simulation::CfdDem
{
  template <int dim, typename number>
  class InflowFlowField : public dealii::Function<dim, number>
  {
  public:
    explicit InflowFlowField(const number time)
      : dealii::Function<dim, number>(dim + 2, time)
    {}

    number
    value(const dealii::Point<dim, number> &loc, const unsigned int component) const final
    {
      if (component == 0)
        return 0.4;
      else if (component == 1)
        return loc[0] == 0. ?
                 (this->get_time() < 1e-8 ? 0.4 * 150 / 1e-8 * this->get_time() : 0.4 * 150) :
                 0.;
      else if (component == dim + 1)
        return 0.4 * (loc[0] == 0. ? 750450 + 0.5 * std::pow(this->get_time() < 1e-8 ?
                                                               0.4 * 150 / 1e-8 * this->get_time() :
                                                               0.4 * 150,
                                                             2) :
                                     739200);
      else
        return 0.;
    }
  };

  template <int dim, typename number>
  class SimulationChannelParticleFlow final : public CfdDemCase<dim, number>
  {
  public:
    SimulationChannelParticleFlow(std::string parameter_file, const MPI_Comm mpi_communicator)
      : CfdDemCase<dim, number>(parameter_file, mpi_communicator)
    {
      AssertThrow(
        dim > 1,
        dealii::ExcMessage(
          "The particles in channel flow case requires dim > 1 but the dimension is set to dim = " +
          std::to_string(dim) + "."));
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      // TODO: Put this to the input file
      std::vector<unsigned int>  repetitions;
      dealii::Point<dim, number> p1;
      dealii::Point<dim, number> p2;
      if constexpr (dim == 2)
        {
          repetitions = {12, 6};
          p1          = dealii::Point<2, number>(0, 0);
          p2          = dealii::Point<2, number>(80e-6, 40e-6);
        }
      else if constexpr (dim == 3)
        {
          repetitions = {12, 6, 6};
          p1          = dealii::Point<3, number>(0, 0, 0);
          p2          = dealii::Point<3, number>(80e-6, 40e-6, 40e-6);
        }

      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation, repetitions, p1, p2, true);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      auto inflow_outflow_solution =
        std::make_shared<InflowFlowField<dim, number>>(this->parameters.time_stepping.start_time);
      auto dummy_solution =
        std::make_shared<InflowFlowField<dim, number>>(this->parameters.time_stepping.start_time);
      this->attach_boundary_condition({0, inflow_outflow_solution}, "inflow", "cfd_dem");
      this->attach_boundary_condition({1, inflow_outflow_solution},
                                      "outflow_fixed_energy",
                                      "cfd_dem");
      // channel boundary
      this->attach_boundary_condition({2, dummy_solution}, "slip_wall", "cfd_dem");
      this->attach_boundary_condition({3, dummy_solution}, "slip_wall", "cfd_dem");
      this->attach_boundary_condition({4, dummy_solution}, "slip_wall", "cfd_dem");
      this->attach_boundary_condition({5, dummy_solution}, "slip_wall", "cfd_dem");
    }

    void
    set_field_conditions() override
    {
      auto initial_condition =
        std::make_shared<InflowFlowField<dim, number>>(this->parameters.time_stepping.start_time);
      this->attach_initial_condition(initial_condition, "cfd_dem");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      InflowFlowField<dim, number> reference_values(generic_data_out.get_time());
      this->print_relative_norm(generic_data_out, reference_values, "norm");
    }
  };
} // namespace MeltPoolDG::Simulation::CfdDem
