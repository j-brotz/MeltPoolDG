/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <tuple>

// MeltPoolDG
#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
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
  class HeatTransferOperator : public OperatorBase<dim, number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using OperatorBase<dim, number>::vmult;
    using OperatorBase<dim, number>::assemble_matrixbased;
    using OperatorBase<dim, number>::create_rhs;
    using OperatorBase<dim, number>::compute_inverse_diagonal_from_matrixfree;

  private:
    using VectorType       = LinearAlgebra::distributed::Vector<number>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    const ScratchData<dim> &scratch_data;
    const HeatData<number> &data;
    const Material<number> &material;
    const unsigned int      temp_dof_idx;
    const unsigned int      temp_quad_idx;
    const unsigned int      temp_hanging_nodes_dof_idx;

    const VectorType &temperature;
    const VectorType &temperature_old;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
                                    neumann_bc; //@todo find a nice way to provide BC
    std::vector<types::boundary_id> bc_radiation_indices;
    std::vector<types::boundary_id> bc_convection_indices;

    const VectorType &heat_source;


    // optional: flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType  *velocity;

    // optional: level set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    const VectorType  *level_set_as_heaviside;

    // optional: two phase flow with evaporation
    VectorType  *evaporative_mass_flux      = nullptr;
    unsigned int evapor_mass_flux_dof_idx   = 0;
    double       latent_heat_of_evaporation = 0;
    bool         do_phenomenological_recoil_pressure;

    // optional: melting/solidification effects
    const bool do_solidification;

    // interpolation of the level set space to the temperature space
    FullMatrix<double> ls_to_temp_grad_interpolation_matrix;
    bool               do_level_set_temperature_gradient_interpolation = false;

    /*
     * contribution to heat source due to evaporation;
     * just for output purposes
     */
    mutable VectorType                             evapor_heat_source;
    mutable VectorType                             evapor_heat_source_projected;
    mutable AlignedVector<VectorizedArray<double>> q_vapor;

    // phase weighted delta functin, only used for evaporative heat loss
    std::unique_ptr<const LevelSet::DeltaApproximationBase<double>> delta_phase_weighted;

    mutable AlignedVector<VectorizedArray<double>> conductivity_at_q;
    mutable VectorType                             conductivity_vec;

    mutable AlignedVector<VectorizedArray<double>> rho_cp_at_q;
    mutable VectorType                             rho_cp_vec;

    // for efficient ghost value update
    mutable std::vector<bool> do_update_ghosts;

  public:
    HeatTransferOperator(const std::shared_ptr<BoundaryConditions<dim>> &bc,
                         const ScratchData<dim>                         &scratch_data_in,
                         const HeatData<number>                         &data_in,
                         const Material<number>                         &material,
                         const unsigned int                              temp_dof_idx_in,
                         const unsigned int                              temp_quad_idx_in,
                         const unsigned int                              temp_hanging_nodes_dof_idx,
                         const VectorType                               &temperature_in,
                         const VectorType                               &temperature_old_in,
                         const VectorType                               &heat_source_in,
                         const unsigned int                              vel_dof_idx_in = 0,
                         const VectorType                               *velocity_in    = nullptr,
                         const unsigned int                              ls_dof_idx_in  = 0,
                         const VectorType *level_set_as_heaviside_in                    = nullptr,
                         const bool        do_solidifiaction_in                         = false);

    void
    register_evaporative_mass_flux(
      VectorType        *evaporative_mass_flux_in,
      const unsigned int evapor_mass_flux_dof_idx_in,
      const double       latent_heat_of_evaporation,
      const typename Evaporation::EvaporationData<number>::EvaporativeCooling &evapor_cooling_data);

    void
    register_surface_mesh(
      const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                   std::vector<Point<dim>> /*quad_points*/,
                                   std::vector<double> /*weights*/
                                   >> &surface_mesh_info);

    void
    assemble_matrixbased(const VectorType &advected_field_old,
                         SparseMatrixType &matrix,
                         VectorType       &rhs) const final;

    void
    update_ghost_values() const;

    void
    zero_out_ghost_values() const;
    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src /*solution_update*/) const final;

    void
    tangent_cell_loop(const MatrixFree<dim, number>        &matrix_free,
                      VectorType                           &dst,
                      const VectorType                     &src,
                      std::pair<unsigned int, unsigned int> cell_range) const;

    /*
     * compute the tangent of Robin-type boundary conditions for convection and radiation
     */
    void
    tangent_boundary_loop(const MatrixFree<dim, number>        &matrix_free,
                          VectorType                           &dst,
                          const VectorType                     &src,
                          std::pair<unsigned int, unsigned int> face_range) const;

    void
    compute_inverse_diagonal_from_matrixfree(VectorType &diagonal) const final;

    void
    compute_system_matrix_from_matrixfree_reduced(
      TrilinosWrappers::SparseMatrix &system_matrix) const;

    void
    compute_system_matrix_from_matrixfree(
      TrilinosWrappers::SparseMatrix &system_matrix) const final;

    void
    rhs_cell_loop(const MatrixFree<dim, number>        &matrix_free,
                  VectorType                           &dst,
                  const VectorType                     &src, /* temperature_old*/
                  std::pair<unsigned int, unsigned int> cell_range) const;

    /*
     * compute the RHS due to Neumann and Robin-type boundary conditions for convection and
     * radiation
     *
     * @todo: add equations
     */
    void
    rhs_boundary_loop(const MatrixFree<dim, number>        &matrix_free,
                      VectorType                           &dst,
                      [[maybe_unused]] const VectorType    &src,
                      std::pair<unsigned int, unsigned int> face_range) const;

    void
    rhs_cut_cell_loop(VectorType &dst) const;

    /**
     * -R(T)
     */
    void
    create_rhs(VectorType &dst, const VectorType &src /*temperature_old*/) const final;

    /**
     * attach vector for solution transfer for AMR
     */
    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

    void
    distribute_constraints();

    void
    reinit() final;

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
    tangent_local_cell_operation(
      FECellIntegrator<dim, 1, number>                        &temp_vals,
      FECellIntegrator<dim, 1, number>                        &temp_lin_vals,
      FECellIntegrator<dim, 1, number>                        &temp_old_vals,
      FECellIntegrator<dim, dim, number>                      &velocity_vals,
      FECellIntegrator<dim, 1, number>                        &ls_vals,
      FECellIntegrator<dim, 1, number>                        &ls_interpolated_vals,
      const std::unique_ptr<FECellIntegrator<dim, 1, number>> &evapor_vals,
      bool                                                     do_reinit_cells) const;

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
     * The values of @p rho_cp and @p conductivity must be set to the material.gas values
     * initially. This function only modifies their values if necessary. I.e. in case of no
     * two-phase flow and no solidification this function does nothing.
     */
    std::tuple<VectorizedArray<number>, VectorizedArray<number>>
    get_material_parameters(const FECellIntegrator<dim, 1, number> &temp_lin_val,
                            const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
                            unsigned int                            q_index) const;

    /**
     * Determine the material parameters and their temperature derivatives. This function takes
     * two-phase flow and solidification effects into account.
     *
     * The values of @p rho_cp, @p conductivity, @p d_rho_cp_d_T and @p d_conductivity_dT must be set
     * to the material.gas values initially (0 for the derivatives). This function only modifies
     * their values if necessary. I.e. in case of no two-phase flow and no solidification this
     * function does nothing.
     *
     * @note The derivatives @p d_rho_cp_d_T and @p d_conductivity_dT are only non-zero in the case of solidification
     * and if the temperature is between the solidus- and liquidus temperature.
     */
    std::tuple<VectorizedArray<number>,
               VectorizedArray<number>,
               VectorizedArray<number>,
               VectorizedArray<number>>
    get_material_parameters_and_derivatives(
      const FECellIntegrator<dim, 1, number> &temp_lin_val,
      const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
      unsigned int                            q_index) const;

    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<double> /*weights*/
                                 >> *surface_mesh_info = nullptr;

    Evaporation::EvaporCoolingInterfaceFluxType evapor_flux_type =
      Evaporation::EvaporCoolingInterfaceFluxType::none;
  };
} // namespace MeltPoolDG::Heat
