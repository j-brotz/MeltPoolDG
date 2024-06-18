/* ---------------------------------------------------------------------
 *
 * Author: Johannes Resch, TUM, May 2024
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/normal_vector/helmholtz_operator.hpp>
#include <meltpooldg/normal_vector/normal_vector_data.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation_base.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    /**
     *    !!!!
     *          the normal vector field is not normalized to length one,
     *          it actually represents the gradient of the level set
     *          function
     *    !!!!
     */

    using namespace dealii;
    template <int dim, typename Number = double>
    class NormalVectorDGOperation : public NormalVectorOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      NormalVectorDGOperation(const ScratchData<dim>         &scratch_data_in,
                              const unsigned int              normal_dof_idx_in,
                              const unsigned int              normal_quad_idx_in,
                              const VectorType               &solution_level_set_in,
                              const NormalVectorData<double> &normal_vector_data);

      void
      reinit() override;

      void
      solve() override;

      const BlockVectorType &
      get_solution_normal_vector() const override;

      BlockVectorType &
      get_solution_normal_vector() override;

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    private:
      const ScratchData<dim>        &scratch_data;
      const VectorType              &solution_level_set;
      const NormalVectorData<double> normal_vector_data;

      TimeIntegration::SolutionHistory<BlockVectorType> solution_history;

      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      const unsigned int normal_dof_idx;
      const unsigned int normal_quad_idx;

      const HelmholtzOperator<dim, Number> helmholtz_operator;

      /**
       * Applies the domain integral of the right hand side
       * @param data the matrix free object
       * @param dst destination where the result is stored
       * @param src source vector
       * @param cell_range
       */
      template <uint direction>
      void
      right_hand_side_domain(const MatrixFree<dim, Number>               &data,
                             VectorType                                  &dst,
                             const VectorType                            &src,
                             const std::pair<unsigned int, unsigned int> &cell_range) const;

      /**
       * Applies the face integral of the right hand side
       * @param data the matrix free object
       * @param dst destination where the result is stored
       * @param src source vector
       * @param face_range
       */
      template <uint direction>
      void
      right_hand_side_inner_face(const MatrixFree<dim, Number>               &data,
                                 VectorType                                  &dst,
                                 const VectorType                            &src,
                                 const std::pair<unsigned int, unsigned int> &face_range) const;

      /**
       * Applies the boundary integral of the right hand side
       * @param data the matrix free object
       * @param dst destination where the result is stored
       * @param src source vector
       * @param face_range
       */
      template <uint direction>
      void
      right_hand_side_boundary_face(const MatrixFree<dim, Number>               &data,
                                    VectorType                                  &dst,
                                    const VectorType                            &src,
                                    const std::pair<unsigned int, unsigned int> &face_range) const;
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
