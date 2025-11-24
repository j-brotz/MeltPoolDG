#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <cmath>
#include <memory>
#include <vector>

#include "../cfd_dem_case.hpp"

namespace MeltPoolDG::Simulation::CfdDem
{
  BETTER_ENUM(FreeJetProfile, char, cosine, constant)

  template <int dim, typename number>
  class Inflow : public dealii::Function<dim, number>
  {
  public:
    explicit Inflow(const number                     time,
                    const number                     ambient_density,
                    const number                     ambient_temperature,
                    const number                     peak_inflow_velocity,
                    const FreeJetProfile             free_jet_profile,
                    const dealii::Point<dim, number> free_jet_center,
                    const number                     free_jet_diameter,
                    const number                     specific_gas_constant,
                    const number                     heat_capacity_ratio)
      : dealii::Function<dim, number>(dim + 2, time)
      , free_jet_profile(free_jet_profile)
      , free_jet_center(free_jet_center)
      , free_jet_radius(free_jet_diameter * 0.5)
      , ambient_density(ambient_density)
      , ambient_temperature(ambient_temperature)
      , peak_inflow_velocity(peak_inflow_velocity)
      , const_vol_specific_heat(specific_gas_constant / (heat_capacity_ratio - 1.))
    {}

    number
    value(const dealii::Point<dim, number> &location, const unsigned int component) const final
    {
      // density component
      if (component == 0)
        return ambient_density;

      // not in free jet area
      if (free_jet_center.distance_square(location) > free_jet_radius * free_jet_radius)
        {
          if (component == dim + 1)
            return ambient_density * const_vol_specific_heat * ambient_temperature;
          else
            return 0.;
        }

      // in free jet area
      number velocity;
      switch (free_jet_profile)
        {
            case FreeJetProfile::cosine: {
              velocity = peak_inflow_velocity * std::cos(free_jet_center.distance(location) /
                                                         free_jet_radius * 0.5 * M_PI);
              break;
            }
            case FreeJetProfile::constant: {
              velocity = peak_inflow_velocity;
              break;
            }
            default: {
              AssertThrow(false, dealii::ExcMessage("Unknown free jet profile!"));
            }
        }

      if (component == dim + 1)
        return ambient_density *
               (const_vol_specific_heat * ambient_temperature + velocity * velocity * 0.5);
      else if (component == dim)
        return ambient_density * velocity;
      else
        return 0.;
    }

  private:
    const FreeJetProfile             free_jet_profile;
    const dealii::Point<dim, number> free_jet_center;
    const number                     free_jet_radius;
    const number                     ambient_density;
    const number                     ambient_temperature;
    const number                     peak_inflow_velocity;
    const number                     const_vol_specific_heat;
  };

  template <int dim, typename number>
  class SimulationAMPowderBed final : public CfdDemCase<dim, number>
  {
  public:
    SimulationAMPowderBed(std::string parameter_file, const MPI_Comm mpi_communicator)
      : CfdDemCase<dim, number>(parameter_file, mpi_communicator)
    {
      AssertThrow(
        dim > 1,
        dealii::ExcMessage(
          "The particles in additive manufacturing powder bed case requires dim > 1 but the dimension is set to dim = " +
          std::to_string(dim) + "."));
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim, number> p1;
      dealii::Point<dim, number> p2;
      for (int i = 0; i < dim; ++i)
        p2[i] = domain_size[i];

      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation, domain_discretization, p1, p2, true);
    }

    void
    set_boundary_conditions() override
    {
      // inflow boundary
      auto inflow_boundary_conditon = std::make_shared<Inflow<dim, number>>(
        this->parameters.time_stepping.start_time,
        ambient_density,
        ambient_temperature,
        jet_peak_velocity,
        free_jet_profile,
        jet_hole_center,
        jet_hole_diameter,
        this->parameters.compressible_material.specific_gas_constant,
        this->parameters.compressible_material.gamma);
      this->attach_boundary_condition({4, inflow_boundary_conditon}, "inflow", "cfd_dem");

      // outflow boundaries (all except the floor)
      auto outflow_energy = std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(
        ambient_density * this->parameters.compressible_material.specific_gas_constant /
        (this->parameters.compressible_material.gamma - 1.) * ambient_temperature);
      for (unsigned int i : {0, 1, 2, 3, 5})
        this->attach_boundary_condition({i, outflow_energy}, "outflow_fixed_energy", "cfd_dem");
    }

    void
    set_field_conditions() override
    {
      std::vector<number> initial_condition;
      switch (this->parameters.application.flow_solver_type)
        {
            case FlowSolverType::compressible: {
              initial_condition.reserve(dim + 2);
              initial_condition.emplace_back(ambient_density);
              for (int i = 0; i < dim; ++i)
                initial_condition.emplace_back(0.);
              initial_condition.emplace_back(
                ambient_density * this->parameters.compressible_material.specific_gas_constant /
                (this->parameters.compressible_material.gamma - 1.) * ambient_temperature);
              break;
            }
            case FlowSolverType::incompressible: {
              initial_condition.reserve(dim);
              for (int i = 0; i < dim; ++i)
                initial_condition.emplace_back(0.);
              break;
            }
            default: {
              AssertThrow(false, dealii::ExcMessage("Unknown flow solver type!"));
            }
        }


      auto initial_condition_function =
        std::make_shared<dealii::Functions::ConstantFunction<dim, number>>(initial_condition);
      this->attach_initial_condition(initial_condition_function, "cfd_dem");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      dealii::Functions::ConstantFunction<dim, number> all_zero(0., dim + 2);
      this->print_relative_norm(generic_data_out, all_zero, "norm");
    }

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("am simulation specific parameters");
      {
        prm.add_parameter("ambient temperature",
                          ambient_temperature,
                          "Ambient temperature used at the inflow boundary.");
        prm.add_parameter("ambient density",
                          ambient_density,
                          "Ambient density used at the inflow boundary.");
        prm.add_parameter("domain size", domain_size, "size of the simulation domain.");
        prm.add_parameter("domain discretization",
                          domain_discretization,
                          "Number of cells in the respective spatial direction.");
        prm.add_parameter("jet hole diameter", jet_hole_diameter, "The diameter of the free jet.");
        prm.add_parameter("jet hole center", jet_hole_center, "Jet hole center coordinates.");
        prm.add_parameter(
          "jet flow angle",
          jet_flow_angle,
          "Angle of the free jet. Zero degrees corresponds to a vertical jet flow.");
        prm.add_parameter("jet peak velocity",
                          jet_peak_velocity,
                          "peak velocity of the free jet, measured at the jet hole center.");
        prm.add_parameter(
          "free jet profile",
          free_jet_profile,
          "The profile of the free jet velocity. There are two options: A 'constant' profile and a 'cosine' profile.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Ambient temperature used for the initial and inflow boundary condition
    number ambient_temperature;

    /// Ambient density used for the initial and inflow boundary condition
    number ambient_density;

    /// Domain size
    std::vector<number> domain_size;

    /// Initial domain discretization (number of cells in the corresponding direction)
    std::vector<unsigned int> domain_discretization;

    /// Jet-hole diamater
    number jet_hole_diameter;
    /// Jet-hole center coordinates
    dealii::Point<dim, number> jet_hole_center;
    /// Angle of the inflow jet flow
    number jet_flow_angle;
    /// Flow jet peak velocity
    number jet_peak_velocity;
    /// The velocity profile of the free jet
    FreeJetProfile free_jet_profile{FreeJetProfile::constant};
  };
} // namespace MeltPoolDG::Simulation::CfdDem
