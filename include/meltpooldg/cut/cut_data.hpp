#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  /**
   * @brief Collection of parameters for the ghost-penalty stabilization of cutFEM and cutDG
   * applications.
   */
  template <typename number>
  struct GhostPenaltyData
  {
    // mass matrix stabilization

    /// Ghost-penalty parameter for penalization of jumps in the values on the ghost-faces
    number gamma_M_degree_0 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal gradients on the ghost-faces
    number gamma_M_degree_1 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal hessians on the ghost-faces
    /// (only relevant for polynomial degree = 2)
    number gamma_M_degree_2 = 1.;

    // stiffness matrix stabilization

    /// Ghost-penalty parameter for penalization of jumps in the values on the ghost-faces
    number gamma_A_degree_0 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal gradients on the ghost-faces
    number gamma_A_degree_1 = 1.;
    /// Ghost-penalty parameter for penalization of jumps in the normal hessians on the ghost-faces
    /// (only relevant for polynomial degree = 2)
    number gamma_A_degree_2 = 1.;

    /**
     * @brief Add ghost-penalty parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  /**
   * @brief Collection of parameters for the stabilization of cutFEM and cutDG applications.
   */
  template <typename number>
  struct CutStabilizationData
  {
    /// Nitsche stabilization parameter
    number nitsche_parameter = 1.;

    /// Parameters for ghost-penalty stabilization
    GhostPenaltyData<number> ghost_penalty;

    /**
     * @brief Add cut-related stabilization parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG
