#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <limits>
#include <string>
#include <vector>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  BETTER_ENUM(
    LaserModelType,
    char,
    // must be specified by user
    not_initialized,
    // analytical laser model, see MeltPoolDG::Heat::LaserAnalyticalTemperatureField
    analytical_temperature,
    // volumetric heat source
    volumetric,
    // interfacial heat source; use continuum surface force modeling within the interface region
    interface_projection_regularized,
    // interfacial heat source; evaluate integral as surface integral over the sharp interface,
    // determined by the margin cube algorithm
    interface_projection_sharp,
    // ONLY FOR MESH CONFORMING INTERFACES: evaluate integral as surface integral over the element
    // faces that represent the interface
    interface_projection_sharp_conforming,
    // use radiative transport equation, see MeltPoolDG::RadiativeTransportOperation
    RTE)

  BETTER_ENUM(LaserIntensityProfileType,
              char,
              // uniform laser model, see MeltPoolDG::Heat::LaserHeatSourceUniform
              uniform,
              // Gauss heat source distribution, see MeltPoolDG::Heat::LaserHeatSourceGauss
              Gauss,
              // Gusarov laser model, see MeltPoolDG::Heat::LaserHeatSourceGusarov
              Gusarov)

  template <typename number = double>
  struct LaserData
  {
    LaserModelType            model             = LaserModelType::not_initialized;
    LaserIntensityProfileType intensity_profile = LaserIntensityProfileType::Gauss;

    number      power            = 0.0;
    std::string power_over_time  = "constant";
    number      power_start_time = 0.0;
    number      power_end_time   = std::numeric_limits<number>::max();

    // TODO these can be unified
    number absorptivity_liquid = 1.0;
    number absorptivity_gas    = 1.0;

    bool   do_move    = false;
    number scan_speed = 0.0;

    template <int dim>
    Point<dim, number>
    get_starting_position() const;

    // return unit direction vector
    template <int dim>
    Tensor<1, dim, number>
    get_direction() const;

    number radius = 0.0;

    // if model = RTE, this value must be set by the simulation
    types::boundary_id rte_boundary_id = numbers::invalid_boundary_id;

    LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;

    struct GusarovData
    {
      number reflectivity           = 0.0; // rho
      number extinction_coefficient = 0.0; // beta
      number layer_thickness        = 0.0; // L
    } gusarov;

    struct AnalyticalData
    {
      number temperature_x_to_y_ratio = 1.0;
      number max_temperature          = 0.0;
      number ambient_temperature      = 0.0;
    } analytical;

    void
    add_parameters(ParameterHandler &prm);

    void
    post(const unsigned int dim,
         const bool         heat_use_volume_specific_thermal_capacity_for_phase_interpolation,
         const MaterialData<number> &material);

    void
    check_input_parameters() const;

  private:
    std::vector<number> starting_position; // default value will be set in post()
    std::vector<number> direction;         // default value will be set in post()
  };
} // namespace MeltPoolDG::Heat
