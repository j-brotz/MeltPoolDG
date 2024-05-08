#pragma once

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

/**
 * Compute the Levelset gradient using Godunovs scheme
 */

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename Number = double>
  class RIGradOperator
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;


    RIGradOperator(const MeltPoolDG::ScratchData<dim> &scratch_data_in,
                   const unsigned int                  reinit_dof_idx_in,
                   const unsigned int                  reinit_quad_idx_in);


    /**
     * Applies the gradient operator to the levelset field
     * @param src source vector of the operator
     * @param dst destination vector where the result is stored
     * @tparam is_right flag if left or right upwind gradient should be calculated
     * @tparam component component of gradient
     */
    template <bool is_right, uint component>
    void
    apply(const VectorType &src, VectorType &dst);

    /**
     * Applies the local inverse of the mass matrix
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_inverse_mass_matrix(const MatrixFree<dim, Number>               &data,
                                    VectorType                                  &dst,
                                    const VectorType                            &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const;

  private:
    const MeltPoolDG::ScratchData<dim> &scratch_data;

    const unsigned int reinit_dof_idx;
    const unsigned int reinit_quad_idx;

    /**
     * Applies the domain integral
     * * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    template <uint component>
    void
    local_apply_domain(const MatrixFree<dim, Number>               &data,
                       VectorType                                  &dst,
                       const VectorType                            &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the inner face integral
     * * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    template <bool is_right, uint component>
    void
    local_apply_inner_face(const MatrixFree<dim, Number>               &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Applies the boundary face integral
     * * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    template <bool is_right, uint component>
    void
    local_apply_boundary_face(const MatrixFree<dim, Number>               &data,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const;
  };
} // namespace MeltPoolDG::LevelSet
