#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  template <typename number = double>
  struct GhostPenaltyData
  {
    // mass matrix stabilization

    // ghost-penalty parameter for penalization of jumps in the values on the ghost-faces
    double gamma_M_degree_0 = 1.;
    // ghost-penalty parameter for penalization of jumps in the normal gradients on the ghost-faces
    double gamma_M_degree_1 = 1.;
    // ghost-penalty parameter for penalization of jumps in the normal hessians on the ghost-faces
    // (only relevant for polynomial degree = 2)
    double gamma_M_degree_2 = 1.;

    // stiffness matrix stabilization

    // ghost-penalty parameter for penalization of jumps in the values on the ghost-faces
    double gamma_A_degree_0 = 1.;
    // ghost-penalty parameter for penalization of jumps in the normal gradients on the ghost-faces
    double gamma_A_degree_1 = 1.;
    // ghost-penalty parameter for penalization of jumps in the normal hessians on the ghost-faces
    // (only relevant for polynomial degree = 2)
    double gamma_A_degree_2 = 1.;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  template <typename number = double>
  struct CutStabilizationData
  {
    // Nitsche stabilization parameter
    double nitsche_parameter = 1.;

    // parameters for ghost-penalty stabilization
    GhostPenaltyData<number> ghost_penalty;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG
