#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/helmholtz_DG_operator.hpp>
#include <meltpooldg/level_set/normal_vector_data.hpp>
#include <meltpooldg/level_set/normal_vector_operation_base.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>
#include <utility>

namespace MeltPoolDG::LevelSet
{
  /**
   *    !!!!
   *          the normal vector field is not normalized to length one,
   *          it actually represents the gradient of the level set
   *          function
   *    !!!!
   */


  template <int dim, typename number>
  class NormalVectorDGOperation : public NormalVectorOperationBase<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    NormalVectorDGOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                            const unsigned int                   normal_dof_idx_in,
                            const unsigned int                   normal_quad_idx_in,
                            const VectorType                    &solution_level_set_in,
                            const NormalVectorData<number>      &normal_vector_data);

    void
    reinit() override;

    void
    solve() override;

    const BlockVectorType &
    get_solution_normal_vector() const override;

    BlockVectorType &
    get_solution_normal_vector() override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

  private:
    const ScratchData<dim, dim, number> &scratch_data;
    const VectorType                    &solution_level_set;
    const NormalVectorData<number>       normal_vector_data;

    TimeIntegration::SolutionHistory<BlockVectorType> solution_history;

    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int normal_dof_idx;
    const unsigned int normal_quad_idx;

    const HelmholtzDGOperator<dim, number> helmholtz_operator;

    /**
     * Applies the domain integral of the right hand side
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    template <uint direction>
    void
    right_hand_side_domain(const dealii::MatrixFree<dim, number>       &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;
  };
} // namespace MeltPoolDG::LevelSet
