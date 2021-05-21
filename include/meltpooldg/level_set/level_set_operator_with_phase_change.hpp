/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// MeltPoolDG
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>

#include <map>
#include <string>

namespace MeltPoolDG::LevelSet
{
  static std::map<std::string, double> get_generalized_theta = {
    {"explicit_euler", 0.0},
    {"implicit_euler", 1.0},
    {"crank_nicolson", 0.5},
  };

  using namespace dealii;

  template <int dim, typename number = double>
  class LevelSetOperatorWithPhaseChange
    : public OperatorBase<dim,
                          number,
                          LinearAlgebra::distributed::Vector<number>,
                          LinearAlgebra::distributed::Vector<number>>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = VectorizedArray<number>;
    using vector              = Tensor<1, dim, VectorizedArray<number>>;
    using scalar              = VectorizedArray<number>;
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;

    const ScratchData<dim> &    scratch_data;
    const VectorType &          advection_velocity;
    const VectorType &          evapor_velocity;
    const LevelSetData<number> &data;
    const unsigned int          ls_dof_idx;
    const unsigned int          ls_hanging_nodes_dof_idx;
    const unsigned int          flow_vel_dof_idx;
    const unsigned int          evapor_vel_dof_idx;
    double                      theta;

  public:
    LevelSetOperatorWithPhaseChange(const ScratchData<dim> &    scratch_data_in,
                                    const VectorType &          advection_velocity_in,
                                    const VectorType &          evapor_velocity_in,
                                    const LevelSetData<number> &data_in,
                                    const unsigned int          ls_dof_idx_in,
                                    const unsigned int          ls_hanging_nodes_dof_idx_in,
                                    const unsigned int          ls_quad_idx_in,
                                    const unsigned int          flow_vel_dof_idx_in,
                                    const unsigned int          evapor_vel_dof_idx_in);
    /*
     *    this is the matrix-based implementation of the rhs and the matrix
     *    @todo: this could be improved by using the WorkStream functionality of dealii
     */

    void
    assemble_matrixbased([[maybe_unused]] const VectorType &advected_field_old,
                         [[maybe_unused]] SparseMatrixType &matrix,
                         [[maybe_unused]] VectorType &      rhs) const override;

    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src) const override;

    void
    create_rhs(VectorType &dst, const VectorType &src) const override;

    void
    solve(const double dt, VectorType &advected_field);
  };
} // namespace MeltPoolDG::LevelSet
