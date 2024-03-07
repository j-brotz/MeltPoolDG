#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/flow/surface_tension_data.hpp>
#include <meltpooldg/material/material_data.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  struct FlowData
  {
    number gravity = 0.0;

    SurfaceTensionData<number> surface_tension;
    DarcyDampingData<number>   darcy_damping;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const MaterialData<number> &material);

    void
    check_input_parameters(const bool curv_enable) const;
  };
} // namespace MeltPoolDG::Flow