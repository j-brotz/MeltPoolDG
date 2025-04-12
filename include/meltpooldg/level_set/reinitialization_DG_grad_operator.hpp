#pragma once

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

/**
 * Compute the gradient of the level set function using Godunov's scheme
 */
namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class RIGradOperator
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;


    RIGradOperator(const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
                   const unsigned int                               reinit_dof_idx_in,
                   const unsigned int                               reinit_quad_idx_in);


    /**
     * Applies the gradient operator to the level set field
     * @param src source vector of the operator
     * @param dst destination vector where the result is stored
     * @tparam is_right flag if left or right upwind gradient should be calculated
     * @tparam component component of gradient
     */
    template <bool is_right, uint component>
    void
    apply(const VectorType &src, VectorType &dst);

  private:
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data;

    const unsigned int reinit_dof_idx;
    const unsigned int reinit_quad_idx;

    /**
     * Applies the domain integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    template <uint component>
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
    template <bool is_right, uint component>
    void
    local_apply_inner_face(const dealii::MatrixFree<dim, number>       &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Applies the boundary face integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    template <bool is_right, uint component>
    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number>       &data,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const;
  };
} // namespace MeltPoolDG::LevelSet
