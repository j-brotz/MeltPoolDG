#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/interface/finite_element_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  BETTER_ENUM(ProblemType,
              char,
              not_initialized,
              advection_diffusion,
              reinitialization,
              level_set,
              melt_pool,
              level_set_with_evaporation,
              heat_transfer,
              radiative_transport)


  BETTER_ENUM(ApplicationName,
              char,
              not_initialized,
              reinit_circle,
              advection_diffusion,
              advection_diffusion_user_output,
              rotating_bubble,
              flow_past_cylinder,
              spurious_currents,
              rising_bubble,
              zalesak_disk,
              recoil_pressure,
              vortex_bubble,
              stefans_problem,
              stefans_problem_with_flow,
              stefans_problem1_with_flow_and_heat,
              stefans_problem2_with_flow_and_heat,
              evaporating_droplet,
              evaporating_shell,
              evaporating_droplet_with_heat,
              unidirectional_heat_transfer,
              thermo_capillary_droplet,
              thermo_capillary_two_droplets,
              solidification_slab,
              film_boiling,
              melt_front_propagation,
              oscillating_droplet,
              moving_droplet,
              radiative_transport,
              powder_bed)

  struct BaseData
  {
    ApplicationName   application_name    = ApplicationName::not_initialized;
    ProblemType       problem_name        = ProblemType::not_initialized;
    unsigned int      dimension           = 2;
    unsigned int      global_refinements  = 1;
    bool              do_print_parameters = true;
    unsigned int      verbosity_level     = 0;
    FiniteElementData fe;

    BaseData();

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const unsigned int ls_n_subdivisions) const;
  };
} // namespace MeltPoolDG
