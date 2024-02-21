#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <string>
#include <vector>

namespace MeltPoolDG
{
  using namespace dealii;
  BETTER_ENUM(
    LaserHeatSourceModel,
    char,
    not_initialized, // must be specified by user
    Gauss,           // Gauss heat source distribution, see MeltPoolDG::Heat::LaserHeatSourceGauss
    Gusarov,         // Gusarov laser model, see MeltPoolDG::Heat::LaserHeatSourceGusarov
    Analytical, // analytical laser model, see MeltPoolDG::Heat::LaserAnalyticalTemperatureField
    uniform,    // uniform laser model, see MeltPoolDG::Heat::LaserHeatSourceUniform
    RTE         // use radiative transport equation, see MeltPoolDG::RadiativeTransportOperation
  )
  BETTER_ENUM(
    LaserImpactType,
    char,
    // volumetric heat source
    volumetric,
    // interfacial heat source; use continuum surface force modeling within the interface region
    interface,
    // interfacial heat source; evaluate integral as surface integral over the sharp interface,
    // determined by the margin cube algorithm
    interface_sharp,
    // ONLY FOR MESH CONFORMING INTERFACES: evaluate integral as surface integral over the element
    // faces that represent the interface
    interface_sharp_conforming)

  template <typename number = double>
  struct LaserData
  {
    LaserHeatSourceModel heat_source_model = LaserHeatSourceModel::not_initialized;

    number      power            = 0.0;
    std::string power_over_time  = "constant";
    number      power_start_time = 0.0;
    number      power_end_time   = 1.e12;

    bool   do_move    = false;
    number scan_speed = 0.0;

    template <int dim>
    Point<dim, number>
    get_starting_position() const;

    // return unit direction vector
    template <int dim>
    Tensor<1, dim, number>
    get_direction() const;

    LaserImpactType impact_type = LaserImpactType::volumetric;

    LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;

    struct GaussData
    {
      number laser_beam_radius   = 0.0;
      number absorptivity_liquid = 0.0;
      number absorptivity_gas    = 0.0;
    } gauss;

    struct GusarovData
    {
      number laser_beam_radius      = 0.0; // R
      number reflectivity           = 0.0; // rho
      number extinction_coefficient = 0.0; // beta
      number layer_thickness        = 0.0; // L
    } gusarov;

    struct AnalyticalData
    {
      number temperature_x_to_y_ratio = 1.0;
      number absorptivity_liquid      = 0.0;
      number absorptivity_gas         = 0.0;
      number max_temperature          = 0.0;
      number ambient_temperature      = 0.0;
    } analytical;

    void
    add_parameters(ParameterHandler &prm);

    void
    post(const unsigned int dim,
         const bool         heat_use_volume_specific_thermal_capacity_for_phase_interpolation,
         const MaterialData<number> &material);

  private:
    std::vector<number> starting_position; // default value will be set in post()
    std::vector<number> direction;
  };
} // namespace MeltPoolDG
