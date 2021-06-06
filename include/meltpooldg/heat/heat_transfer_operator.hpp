/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

// MeltPoolDG
#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /*
   * This operator computes the residual and its consistent tangent of the discretized heat
   * equation with temperature dependent material properties:
   * ρ^(n) = ρ(T^(n)), c_p^(n) = c_p(T^(n)), k^(n) = k(T^(n))
   *
   *                  1  /                                                     \
   *  R(T_b^(n+1)) = --- | N_a, ρ^(n+1) c_p^(n+1) N_b ( T_b^(n+1) - T_b^(n)) ) |
   *                 dt  \                                                     /
   *                                                                            Ω
   *                 /                               \
   *               + | ∇N_a, k^(n+1) ∇N_b T_b^(n+1)) |
   *                 \                               /
   *                                                  Ω
   *                 /                                           \
   *               + | N_a, ρ^(n+1) c_p^(n+1) ∇N_b T_b^(n+1) · u |
   *                 \                                           /
   *                                                              Ω
   *                 /                                          \
   *               + | N_a, ρ^(n+1) c_p^(n+1) N_b T_b^(n+1) ∇·u |    (this term is not yet considered @todo)
   *                 \                                          /
   *                                                             Ω
   *                 /      _   \     /      _  \
   *               - | N_a, q_s |  -  | N_a, q  | = 0
   *                 \          /     \         /
   *                             Ω               Γ
   *                                              N
   *
   *
   *  dR(T^(n+1))    1  /                            \
   *  ----------- = --- | N_a, ρ^(n+1) c_p^(n+1) N_b |
   *  dT_b^(n+1)     dt \                            /
   *                                                  Ω
   *                1  /       d ρ c_p |                              \
   *              + -- | N_a, ---------| N_b ( T_b^(n+1) - T_b^(n)) ) |
   *                dt \         d T   |                              /
   *                                   |(n+1)                          Ω
   *                /                       d k |                \
   *              + | ∇N_a, k^(n+1) ∇N_b + -----| ∇N_b T_b^(n+1) |
   *                \                       d T |                /
   *                                            |(n+1)            Ω
   *                /                                             \
   *              + | N_a, ρ^(n+1) c_p^(n+1) ( ∇N_b u + N_b ∇·u ) |
   *                \                                             /
   *                                                               Ω
   *                /       d ρ c_p |                                            \
   *              + | N_a, ---------| ( ∇N_b T_b^(n+1) · u + N_b T_b^(n+1) ∇·u ) |
   *                \         d T   |                                            /
   *                                |(n+1)                                        Ω
   *                            _                      _
   *                /        d q_s     \    /        d q       \
   *              - | N_a, ---------   |  - | N_a, ---------   |
   *                \       dT_b^(n+1) /    \       dT_b^(n+1) /
   *                                    Ω                       Γ
   *                                                             N
   *
   * with shape functions N_a and N_b, nodal temperature values T_b^(n+1), the density ρ, the
   * specific heat capacity c_p and the conductivity k, source/sink terms q_s and prescribed
   * fluxes q along Neumann boundaries. The heat flux may result from radiative losses
   *
   *  _
   *  q = σ ϵ (T^4-T∞^4)
   *
   * with the Stefan-Boltzmann constant σ, the emissivity ϵ and the temperature of the surroundings
   * T∞ as well as convective losses
   *
   *  _
   *  q = α (T-T∞)
   *
   * with the convection coefficient denoted as α.
   *
   * We assume that the density and the specific heat capacity do not dependent on the temperature.
   *
   */

  template <int dim, typename number = double>
  class HeatTransferOperator : public OperatorBase<dim,
                                                   number,
                                                   LinearAlgebra::distributed::Vector<number>,
                                                   LinearAlgebra::distributed::Vector<number>>
  {
  private:
    using VectorType       = LinearAlgebra::distributed::Vector<number>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    const ScratchData<dim> &    scratch_data;
    const HeatData<number> &    data;
    const MaterialData<number> &material;
    const unsigned int          temp_dof_idx;
    const unsigned int          temp_quad_idx;

    const VectorType &temperature;
    const VectorType &temperature_old;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
                                    neumann_bc; //@todo find a nice way to provide BC
    std::vector<types::boundary_id> bc_radiation_indices;
    std::vector<types::boundary_id> bc_convection_indices;

    const VectorType &heat_source;


    // optional: flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType * velocity;

    // optional: level-set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    const VectorType * level_set_as_heaviside;

    // optional: two phase flow with evaporation
    const EvaporationData<double> *evapor_data;
    double                         evaporation_heat_transfer_coefficient = 0;
    VectorType *                   evaporative_mass_flux                 = nullptr;

    /*
     * contribution to heat source due to evaporation;
     * just for output purposes
     */
    mutable VectorType                                        evapor_heat_source;
    mutable std::vector<std::vector<VectorizedArray<double>>> q_vapor;


    const double inv_mushy_interval;

  public:
    HeatTransferOperator(const std::shared_ptr<BoundaryConditions<dim>> &bc,
                         const ScratchData<dim> &                        scratch_data_in,
                         const HeatData<number> &                        data_in,
                         const MaterialData<number> &                    material_data_in,
                         unsigned int                                    temp_dof_idx_in,
                         unsigned int                                    temp_quad_idx_in,
                         const VectorType &                              temperature_in,
                         const VectorType &                              temperature_old_in,
                         const VectorType &                              heat_source_in,
                         unsigned int                                    vel_dof_idx_in = 0,
                         const VectorType *                              velocity_in    = nullptr,
                         unsigned int                                    ls_dof_idx_in  = 0,
                         const VectorType *             level_set_as_heaviside_in       = nullptr,
                         const EvaporationData<double> *evapor_data_in                  = nullptr);

    void
    register_evaporative_mass_flux(VectorType *evaporative_mass_flux_in);

    void
    assemble_matrixbased(const VectorType &advected_field_old,
                         SparseMatrixType &matrix,
                         VectorType &      rhs) const override;

    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src /*solution_update*/) const override;

    void
    tangent_cell_loop(const MatrixFree<dim, number> &       matrix_free,
                      VectorType &                          dst,
                      const VectorType &                    src,
                      std::pair<unsigned int, unsigned int> cell_range) const;

    /*
     * compute the tangent of Robin-type boundary conditions for convection and radiation
     */
    void
    tangent_boundary_loop(const MatrixFree<dim, number> &       matrix_free,
                          VectorType &                          dst,
                          const VectorType &                    src,
                          std::pair<unsigned int, unsigned int> face_range) const;

    void
    compute_inverse_diagonal(VectorType &diagonal) const;

    void
    compute_system_matrix(TrilinosWrappers::SparseMatrix &system_matrix,
                          bool                            include_boundary_terms = false) const;

    void
    rhs_cell_loop(const MatrixFree<dim, number> &       matrix_free,
                  VectorType &                          dst,
                  const VectorType &                    src, /* temperature_old*/
                  std::pair<unsigned int, unsigned int> cell_range) const;

    /*
     * compute the RHS due to Neumann and Robin-type boundary conditions for convection and
     * radiation
     *
     * @todo: add equations
     */
    void
    rhs_boundary_loop(const MatrixFree<dim, number> &       matrix_free,
                      VectorType &                          dst,
                      [[maybe_unused]] const VectorType &   src,
                      std::pair<unsigned int, unsigned int> face_range) const;

    /**
     * -R(T)
     */
    void
    create_rhs(VectorType &dst, const VectorType &src /*temperature_old*/) const override;

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

  private:
    /**
     * This function executes the local cell operation for computing the tangent.
     *
     * @note The function assumes that @p temp_vals has been already initialized
     *   and the dof-values of @p temp_vals are already set a priori. Afterwards, the
     *   dof-values held by @p temp_vals can be written back to the global vector via
     *   temp_vals.distribute_local_to_global(dst).
     */
    void
    tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &  temp_vals,
                                 FECellIntegrator<dim, 1, number> &  temp_lin_vals,
                                 FECellIntegrator<dim, 1, number> &  temp_old_vals,
                                 FECellIntegrator<dim, dim, number> &velocity_vals,
                                 FECellIntegrator<dim, 1, number> &  ls_vals,
                                 bool                                do_reinit_cells) const;

    /**
     * This function executes the local boundary operation for computing the tangent.
     *
     * @note The function assumes that @p temp_vals has been already initialized
     *   and the dof-values of @p temp_vals are already set a priori. Afterwards, the
     *   dof-values held by @p temp_vals can be written back to the global vector via
     *   temp_vals.distribute_local_to_global(dst).
     */
    void
    tangent_local_boundary_operation(FEFaceIntegrator<dim, 1, number> &dQ_dT,
                                     FEFaceIntegrator<dim, 1, number> &temp_vals,
                                     bool                              do_reinit_face) const;

    /**
     * Determine the material parameters. This function takes two-phase flow and solidification
     * effects into account.
     *
     * The values of \p rho_cp and \p conductivity must be set to the material.first values
     * initially. This function only modifies their values if necessary. I.e. in case of no
     * two-phase flow and no solidification this function does nothing.
     */
    void
    get_material_parameters(VectorizedArray<number> &               rho_cp,
                            VectorizedArray<number> &               conductivity,
                            bool                                    with_solidification,
                            bool                                    with_two_phase,
                            const FECellIntegrator<dim, 1, number> &temp_lin_val,
                            const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
                            unsigned int                            q_index) const;

    /**
     * Determine the material parameters and their temperature derivatives. This function takes
     * two-phase flow and solidification effects into account.
     *
     * The values of \p rho_cp, \p conductivity, \p d_rho_cp_dT and \p d_conductivity_dT must be set
     * to the material.first values initially. This function only modifies their values if
     * necessary. I.e. in case of no two-phase flow and no solidification this function does
     * nothing.
     */
    void
    get_material_parameters_and_derivatives(
      VectorizedArray<number> &               rho_cp,
      VectorizedArray<number> &               conductivity,
      VectorizedArray<number> &               d_rho_cp_dT,
      VectorizedArray<number> &               d_conductivity_dT,
      bool                                    with_solidification,
      bool                                    with_two_phase,
      const FECellIntegrator<dim, 1, number> &temp_lin_val,
      const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
      unsigned int                            q_index) const;

    /*
     * Determine material parameters (\p capacity, \p conductivity and \p density) for
     * solidification/melting. Input the liquid's material parameters via \p liq_capacity, \p
     * liq_conductivity and \p liq_density. In the mushy zone (where the solid fraction os between 0
     * and 1) the material parameters will be interpolated with smooth cubic function, see
     * UtilityFunctions::interpolate_cubic().
     *
     * In case of two-phase flow use get_material_parameters_with_two_phase_flow() to determine the
     * liquid's parameters first.
     */
    void
    get_material_parameters_with_solidification(
      VectorizedArray<number> &      capacity,
      VectorizedArray<number> &      conductivity,
      VectorizedArray<number> &      density,
      const VectorizedArray<number> &liq_capacity,
      const VectorizedArray<number> &liq_conductivity,
      const VectorizedArray<number> &liq_density,
      const VectorizedArray<number> &solid_fraction) const;

    /*
     * Determine derivatives of the material parameters (\p d_capacity_dT, \p d_conductivity_dT and
     * \p d_density_dT) with respect to the temperature for solidification/melting. Input the
     * liquid's material parameters via \p liq_capacity, \p liq_conductivity and \p liq_density.
     * This function will return the temperature derivatives of the values determined by
     * get_material_parameters_with_solidification().
     *
     * In case of two-phase flow use get_material_parameters_with_two_phase_flow() to determine the
     * liquid's parameters first.
     */
    void
    get_material_parameter_derivatives_with_solidification(
      VectorizedArray<number> &      d_capacity_dT,
      VectorizedArray<number> &      d_conductivity_dT,
      VectorizedArray<number> &      d_density_dT,
      const VectorizedArray<number> &liq_capacity,
      const VectorizedArray<number> &liq_conductivity,
      const VectorizedArray<number> &liq_density,
      const VectorizedArray<number> &solid_fraction) const;

    /*
     * Determine the material parameters (\p capacity, \p conductivity and \p density) for two phase
     * flow. If \p ls_heaviside_val = 0 this function returns the first materials parameters and if
     * \p ls_heaviside_val = 1 it returns the second materials parameters. At the interface the
     * parameters are smeared if "heat variable properties over interface" is set to true, else the
     * parameters will jump.
     *
     * This does not account for solidification effect. In case of solidification this function must
     * be used to determine the liquid's material parameters for
     * get_material_parameters_with_solidification() and
     * get_material_parameter_derivatives_with_solidification().
     */
    void
    get_material_parameters_with_two_phase_flow(
      VectorizedArray<number> &      capacity,
      VectorizedArray<number> &      conductivity,
      VectorizedArray<number> &      density,
      const VectorizedArray<number> &ls_heaviside_val) const;

    /*
     * Compute the solid fraction for a temperature between the liquidus and the solidus
     * temperature. If the temperature is equal to the liquidus temperature, then the solid
     * fraction is zero. If the temperature is equal to the solidus temperature, then the solid
     * fraction is one. In between there is a linear interpolation.
     */
    VectorizedArray<number>
    calculate_solid_fraction(const VectorizedArray<number> &temperature) const;
  };
} // namespace MeltPoolDG::Heat
