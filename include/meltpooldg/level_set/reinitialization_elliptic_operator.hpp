#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG::LevelSet
{
  /**
   * @brief Operator for the matrix-free evaluation of a compressible single-phase flow cutDG
   * formulation for explicit time integration.
   *
   * @tparam dim Dimension of the considered simulation case.
   * @tparam number Floating point format type.
   */
  template <int dim, typename number>
  class ReinitializationEllipticOperator
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    /**
     * @brief Constructor.
     *
     * Operator for the matrix-free solution of the nonlinear elliptic reinitialization.
     *
     * @param scratch_data_in Reference to the used ScratchData object.
     * @param reinit_data_in Reference to the object for reinitialization-specific data.
     * @param reinit_dof_idx_in Index of the used dof-handler object in @p scratch_data_in.
     * @param reinit_quad_idx_in Index of the used quadrature object in @p scratch_data_in.
     * @param normal_vector_in Reference to the collection of normal vectors for the individual
     * space dimensions.
     * @param is_dg_in Boolean indicator whether DG-FEM or CG-FEM is used.
     */
    ReinitializationEllipticOperator(
      const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
      const ReinitializationData<number>              &reinit_data_in,
      const unsigned int                               reinit_dof_idx_in,
      const unsigned int                               reinit_quad_idx_in,
      const BlockVectorType                           &normal_vector_in,
      const bool                                       is_dg_in);

  private:
    /// Boolean indicator whether DG-FEM or CG-FEM is used. This indicator is used to evaluate
    /// the correct terms in the cell and face loops.
    bool is_dg;
  };
} // namespace MeltPoolDG::LevelSet
