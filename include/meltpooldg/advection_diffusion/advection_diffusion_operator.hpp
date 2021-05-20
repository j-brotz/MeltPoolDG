/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// MeltPoolDG
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::AdvectionDiffusion
{
  static std::map<std::string, double> get_generalized_theta = {
    {"explicit_euler", 0.0},
    {"implicit_euler", 1.0},
    {"crank_nicolson", 0.5},
  };

  using namespace dealii;

  template <int dim, typename number = double>
  class AdvectionDiffusionOperator : public OperatorBase<dim,
                                                         number,
                                                         LinearAlgebra::distributed::Vector<number>,
                                                         LinearAlgebra::distributed::Vector<number>>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using VectorizedArrayType = VectorizedArray<number>;
    using vector              = Tensor<1, dim, VectorizedArray<number>>;
    using scalar              = VectorizedArray<number>;

  public:
    AdvectionDiffusionOperator(const ScratchData<dim> &              scratch_data_in,
                               const VectorType &                    advection_velocity_in,
                               const AdvectionDiffusionData<number> &data_in,
                               unsigned int                          dof_idx_in,
                               unsigned int                          quad_idx_in,
                               unsigned int                          velocity_dof_idx_in);

    /*
     *    this is the matrix-based implementation of the rhs and the matrix
     *    @todo: this could be improved by using the WorkStream functionality of deal.II
     */

    void
    assemble_matrixbased(const VectorType &advected_field_old,
                         SparseMatrixType &matrix,
                         VectorType &      rhs) const override;

    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src) const override;

    void
    create_rhs(VectorType &dst, const VectorType &src) const override;

  private:
    const ScratchData<dim> &              scratch_data;
    const VectorType &                    advection_velocity;
    const AdvectionDiffusionData<number> &data;
    const unsigned int                    velocity_dof_idx;
    double                                theta;
  };
} // namespace MeltPoolDG::AdvectionDiffusion
