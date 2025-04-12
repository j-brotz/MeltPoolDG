#pragma once

#include <meltpooldg/level_set/reinitialization_DG_diffusion_operator.hpp>
#include <meltpooldg/level_set/reinitialization_DG_grad_operator.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

/**
 * For implementation details see
 *
 * A. Karakus, N. Chalmers, and T. Warburton. A local discontinuous galerkin
 * level set reinitialization with subcell stabilization on unstructured meshes. Com-
 * puters & Mathematics with Applications, 123:160–170, 2022.
 *
 * and
 *
 * A. Ritthaler. “High-Performance Matrix-Free High-Order Discontinuous Galerkin Level-Set Ad-
 * vection and Reinitialization”. Technical University of Munich, 2023.
 */


namespace MeltPoolDG::LevelSet
{

  template <int dim, typename number>
  class ReinitilizationDGOperator
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    ReinitilizationDGOperator(const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
                              const ReinitializationData<number>              &reinit_data_in,
                              const unsigned int                               reinit_dof_idx_in,
                              const unsigned int                               reinit_quad_idx_in,
                              const VectorType                                &curvature_in,
                              const BlockVectorType                           &normal_vector_in);


    /**
     * Sets the smoothed signum field and the Godunov gradient
     * @param solution is the distorted level set field before the reinilization.
     */
    void
    prepare_operator(const VectorType &solution);

    /**
     * Applies the DG reinilization operator to the src vector and stores the result in the dst
     * vector. The time needs to be passed for a consistent interface of the time integration scheme
     * @param time current time
     * @param dst result of the operator applied to @param src
     * @param src source vector for the operator
     */
    void
    apply_operator(number const                                           time,
                   VectorType                                            &dst,
                   VectorType const                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func) const;

    /**
     * Function is needed for a conistent time integration interface. In this case the function is
     * empty.
     * @param time current time
     * @param dst result of the operator applied to @param src
     * @param src source vector for the operator
     */
    void
    apply_dirichlet_boundary_operator([[maybe_unused]] number const      time,
                                      [[maybe_unused]] VectorType       &dst,
                                      [[maybe_unused]] VectorType const &src) const {};
    /**
     * Resizes member vectors to the right size of the underlying DOF handler
     */
    void
    reinit();

    /**
     * Applies the diffusion term with an implicit time integration in order to keep the time step
     * size of the integration scheme not limited by the diffusion term.
     *
     * @param time current time
     * @param time_step size of the time
     * @param solution_history keeps different time instances of the level set field
     */
    void
    apply_diffusion_implicit(number const time,
                             number const time_step,
                             [[maybe_unused]] TimeIntegration::SolutionHistory<VectorType, number>
                               &solution_history) const;

    /**
     * flag for the time integration scheme if field functions should be updated in every step.
     */
    bool update_field_functions = true;

    /**
     * Sets field functions. Is empty in this case.
     * @param time current time
     */
    void
    set_field_functions([[maybe_unused]] number const time) const {};

    number
    get_max_diffusitivity() const;

    VectorType &
    get_signum_smoothed() const
    {
      return signum_smoothed;
    }

    VectorType &
    get_sign_indicator_function()
    {
      return sign_indicator_function;
    }

    void
    set_artificial_diffusitivity();

  private:
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data;

    const unsigned int reinit_dof_idx;
    const unsigned int reinit_quad_idx;

    const ReinitializationData<number> &reinit_data;

    mutable VectorType num_Hamiltonian;
    mutable VectorType signum_smoothed;
    mutable VectorType God_grad;
    mutable VectorType sign_indicator_function;

    // auxiliary vectors for Godunov's scheme
    mutable VectorType grad_x_l;
    mutable VectorType grad_x_r;
    mutable VectorType grad_y_l;
    mutable VectorType grad_y_r;
    mutable VectorType grad_z_l;
    mutable VectorType grad_z_r;

    /**
     *operators
     */
    ReinitializationDGDiffusionOperator<dim, number> RI_DG_diffusion_operator;
    mutable RIGradOperator<dim, number>              RI_grad_operator;


    /**
     * Time integration scheme for the IMEX integration of the diffusive term.
     */
    std::shared_ptr<TimeIntegratorBase<number>> IMEX_integration;


    /**
     * Calculates the numerical Hamiltonian of the Hamilton-Jacobi equation
     * @param solution level set field
     */
    void
    compute_godunov_hamiltonian(const VectorType &solution) const;

    /**
     * Calculates the Godunov gradient
     * @param solution level set field
     */
    void
    compute_godunov_gradient(const VectorType &solution) const;

    /**
     * Calculates the smoothed signum function and stores the result in the member @p signum_smoothed
     * @param solution the distorted level set field before reinitialization
     * @param min_vertex_distance smallest vertex distance of the mesh
     */
    void
    compute_smoothed_signum(const VectorType &solution, const number min_vertex_distance) const;

    /**
     * Adds a penalty term to reinit equation when the interface moves a lot
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    interface_movement_penalty(const dealii::MatrixFree<dim, number>       &data,
                               VectorType                                  &dst,
                               const VectorType                            &src,
                               const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Computes the weak form of the Hamiltonian
     */
    void
    local_apply_domain_num_Hamiltonian(
      const dealii::MatrixFree<dim, number>       &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;
  };
} // namespace MeltPoolDG::LevelSet
