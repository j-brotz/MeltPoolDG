/**
 * @brief Collection of cut-related parameters required by the cut single-phase and multiphase
 * compressible Navier-Stokes operators.
 */
#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/cut/cut_data.hpp>

#include <string>

namespace MeltPoolDG::Flow
{
  template <typename number>
  struct CompressibleFlowCutData
  {
    // flow boundary condition at the immersed boundary
    // (only relevant for single phase flow problem)
    // (choose between: "no_slip_wall", "inflow")
    std::string unfitted_flow_boundary_condition = "no_slip_wall";

    // cut-related stabilization parameters
    CutStabilizationData<number> stabilization;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("cut");
      {
        prm.add_parameter("unfitted flow boundary condition",
                          unfitted_flow_boundary_condition,
                          "Flow boundary condition type at the unfitted boundary. "
                          "Choose between 'no_slip_wall' and 'inflow'.");
        stabilization.add_parameters(prm);
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG::Flow
