#pragma once

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

/**
 * Class for the diffusive stabilization of the level set reinitialization
 */
namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class ReinitializationDGDiffusionOperator
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    ReinitializationDGDiffusionOperator(
      const MeltPoolDG::ScratchData<dim, dim, number> &scratch_datain,
      const ReinitializationData<number>              &reinit_data_in,
      const unsigned int                               reinit_dof_idx_in,
      const unsigned int                               reinit_quad_idx_in,
      const VectorType                                &curvature_in,
      const BlockVectorType                           &normal_vector_in,
      const VectorType                                &smooth_signum_in);

    /**
     * Computes the necessary amount of diffusion
     */
    void
    compute_diffusitivity_value();

    /**
     * Computes the necessary penalty parameter
     */
    void
    compute_penalty_parameter();

    /**
     * If an analytical function for a field is provided and an analytical update is
     * enabled, this function sets the field according to the analytical function. This function
     * implementation is needed for a time integrator. This function is currently unused
     * @param time
     */
    void
    set_field_functions([[maybe_unused]] const number time) const {};

    /**
     * Applies the DG diffusion operator to the src vector and stores the result in the dst vector.
     * The dst vector is zeroed out before the operation.
     * @param time
     * @param dst destination vecor where the diffusive part is stored
     * @param src source vector for the operator
     */
    void
    apply_operator([[maybe_unused]] const number                          time,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func = {}) const;

    void
    local_apply_inverse_mass_matrix(const dealii::MatrixFree<dim, number>                    &data,
                                    dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                                    const dealii::LinearAlgebra::distributed::Vector<number> &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the dirichlet contribution of the DG diffusion operator to the src vector and stores
     * the result in the dst vector. The dst vector is NOT zeroed out before the operation.
     * Is not implemented since for a levelset reinitialization, dirichlet boundaries don't make
     * much sense.
     */
    void
    apply_dirichlet_boundary_operator([[maybe_unused]] const number      time,
                                      [[maybe_unused]] VectorType       &dst,
                                      [[maybe_unused]] const VectorType &src) const {};

    number
    get_max_diffusitivity() const;

    /**
     *Variable is needed for time integration
     */
    bool update_field_functions = false;

  private:
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data;
    const ReinitializationData<number>              &reinit_data;

    const unsigned int reinit_dof_idx;
    const unsigned int reinit_quad_idx;

    mutable VectorType                                     diffusitivity;
    mutable AlignedVector<dealii::VectorizedArray<number>> array_penalty_parameter;

    const VectorType      &curvature;
    const BlockVectorType &normal_vector;
    const VectorType      &smooth_signum;

    /**
     * Applies the domain integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_domain(const dealii::MatrixFree<dim, number>       &data,
                       VectorType                                  &dst,
                       const VectorType                            &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the inner face integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_inner_face(const dealii::MatrixFree<dim, number>       &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the boundary face integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number>       &data,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &cell_range) const;
  };
} // namespace MeltPoolDG::LevelSet
