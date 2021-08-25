#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  /**
   * This class computes the Darcy damping force (or Darcy source term).
   *
   *         /              \
   *  f_d =  | N_a, K N_b u |
   *         \              /
   *                         Ω
   *
   * with the isotropic permeability of the mushy zone K given by the Kozeny-Carman equation:
   *
   *                 fs²
   *  K = - C ----------------
   *           ( 1- fs )³ + b
   *
   * with the solid fraction fs \in [0, 1], the morphology of the mushy zone C and the parameter b
   * to avoid division by zero.
   *
   * @note The constant b must be greater than zero.
   *
   * Voller, V. R., & Prakash, C. (1987). A fixed grid numerical modelling methodology for
   * convection-diffusion mushy region phase-change problems. International Journal of Heat and Mass
   * Transfer, 30(8), 1709–1719. https://doi.org/10.1016/0017-9310(87)90317-6
   */
  template <int dim>
  class DarcyDampingOperation
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const double            mushy_zone_morphology;
    const double            avoid_div_zero_constant;
    const ScratchData<dim> &scratch_data;
    const unsigned int      flow_vel_hanging_nodes_dof_idx;
    const unsigned int      flow_quad_idx;
    const unsigned int      solid_dof_idx;

  public:
    DarcyDampingOperation(const DarcyDampingData<double> &data_in,
                          const ScratchData<dim> &        scratch_data,
                          const unsigned int              flow_vel_hanging_nodes_dof_idx,
                          const unsigned int              flow_quad_idx,
                          const unsigned int              solid_dof_idx);

    void
    compute_darcy_damping(VectorType &      force_rhs,
                          const VectorType &velocity_vec,
                          const VectorType &solid_fraction_vec,
                          const bool        zero_out = true);
  };
} // namespace MeltPoolDG::Flow
