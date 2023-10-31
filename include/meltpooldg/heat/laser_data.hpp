#pragma once
#include <meltpooldg/level_set/delta_approximation_phase_weighted_parameters.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

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
    number              power            = 0.0;
    std::string         power_over_time  = "constant";
    number              power_start_time = 0.0;
    number              power_end_time   = 1.e12;
    std::vector<double> center; // default value will be set after parameters are read
    bool                do_move     = false;
    number              scan_speed  = 0.0;
    LaserImpactType     impact_type = LaserImpactType::volumetric;
    DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;
    LaserHeatSourceModel heat_source_model = LaserHeatSourceModel::not_initialized;
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
    add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("laser");
      {
        prm.add_parameter("laser power", power, "Intensity of the laser");
        prm.add_parameter("laser power over time",
                          power_over_time,
                          "Temporal distribution of the laser power",
                          Patterns::Selection("constant|ramp"));
        prm.add_parameter("laser power start time",
                          power_start_time,
                          "In case of time-dependent laser power: activation time of ");
        prm.add_parameter("laser power end time",
                          power_end_time,
                          "In case of time-dependent laser power: end time of ");
        prm.add_parameter("laser center",
                          center,
                          "Center coordinates of the laser beam on the interface melt/gas.");
        prm.add_parameter("laser scan speed", scan_speed, "Scan speed of the ");
        prm.add_parameter(
          "laser do move",
          do_move,
          "Set this parameter to true to move the laser in x-direction with the given parameter scan speed.");
        prm.add_parameter(
          "laser impact type",
          impact_type,
          "Laser impact model. "
          "volumetric: volumetric heat source; "
          "interface: surface heat source at the two-phase interface modelled as a "
          "continuum surface force within the interface region; "
          "interface sharp: impact of surface heat source at the two-phase interface modelled as a "
          "a sharp surface force (evaluation of a surface integral);");
        delta_approximation_phase_weighted.add_parameters(prm);
        prm.add_parameter("laser heat source model", heat_source_model, "Laser heat source model.");
        /*
         *   Gusarov
         */
        prm.add_parameter("laser gusarov laser beam radius",
                          gusarov.laser_beam_radius,
                          "Laser beam radius.");
        prm.add_parameter("laser gusarov reflectivity",
                          gusarov.reflectivity,
                          "Reflectivity of the material.");
        prm.add_parameter("laser gusarov extinction coefficient",
                          gusarov.extinction_coefficient,
                          "Extinction coefficient in [1/m].");
        prm.add_parameter("laser gusarov layer thickness",
                          gusarov.layer_thickness,
                          "Layer thickness");
        /*
         *   Gauss
         */
        prm.add_parameter("laser gauss laser beam radius",
                          gauss.laser_beam_radius,
                          "Laser beam radius.");
        prm.add_parameter("laser gauss absorptivity gas",
                          gauss.absorptivity_gas,
                          "Laser energy absorptivity of the gaseous part of the domain.");
        prm.add_parameter("laser gauss absorptivity liquid",
                          gauss.absorptivity_liquid,
                          "Laser energy absorptivity of the liquid part of the domain.");
        prm.enter_subsection("analytical");
        {
          /*
           *   Analytical temperature field
           */
          prm.add_parameter("absorptivity liquid",
                            analytical.absorptivity_liquid,
                            "Absorptivity of the liquid part of domain");
          prm.add_parameter("absorptivity gas",
                            analytical.absorptivity_gas,
                            "Absorptivity of the gaseous part of domain");
          prm.add_parameter("ambient temperature",
                            analytical.ambient_temperature,
                            "Ambient temperature in the inert gas.");
          prm.add_parameter(
            "max temperature",
            analytical.max_temperature,
            "Maximum temperature arising in the melt pool. If this temperature is lower than the boiling"
            " temperature, this value is corrected to correspond to the boiling temperature + 500 K.");
          prm.add_parameter(
            "temperature x to y ratio",
            analytical.temperature_x_to_y_ratio,
            "This factor scales the analytical temperature field to be anisotropic.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
