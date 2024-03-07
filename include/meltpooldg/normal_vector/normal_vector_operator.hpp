/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

using namespace dealii;

namespace MeltPoolDG
{
  namespace LevelSet
  {
    template <int dim, typename number = double>
    class NormalVectorOperator : public OperatorBase<dim, number>
    {
      //@todo: to avoid compiler warnings regarding hidden overriden functions
      using OperatorBase<dim, number>::vmult;
      using OperatorBase<dim, number>::assemble_matrixbased;
      using OperatorBase<dim, number>::create_rhs;
      using OperatorBase<dim, number>::compute_inverse_diagonal_from_matrixfree;

    public:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using VectorizedArrayType = VectorizedArray<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;

      NormalVectorOperator(const ScratchData<dim>         &scratch_data_in,
                           const NormalVectorData<double> &normal_vector_data_in,
                           const unsigned int              normal_dof_idx_in,
                           const unsigned int              normal_quad_idx_in,
                           const unsigned int              ls_dof_idx_in,
                           const VectorType               *solution_level_set_in = nullptr);

      void
      assemble_matrixbased(const VectorType &level_set_in,
                           SparseMatrixType &matrix,
                           BlockVectorType  &rhs) const final;

      /*
       *  matrix-free utility
       */

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const final;

      void
      create_rhs(BlockVectorType &dst, const VectorType &src) const final;

      void
      compute_system_matrix_from_matrixfree(
        TrilinosWrappers::SparseMatrix &system_matrix) const final;

      void
      compute_inverse_diagonal_from_matrixfree(BlockVectorType &diagonal) const final;

      static void
      get_unit_normals_at_quadrature(const FEValues<dim>         &fe_values,
                                     const BlockVectorType       &normal_vector_field_in,
                                     std::vector<Tensor<1, dim>> &unit_normal_at_quadrature,
                                     const double                 zero = 1e-16);

    private:
      void
      tangent_local_cell_operation(FECellIntegrator<dim, 1, number> &normal_vals,
                                   FECellIntegrator<dim, 1, number> &level_set_vals,
                                   const bool                        do_reinit_cells) const;

    private:
      void
      reinit() final;

    private:
      const ScratchData<dim>         &scratch_data;
      const NormalVectorData<double> &normal_vector_data;

      const unsigned int normal_dof_idx;
      const unsigned int normal_quad_idx;
      const unsigned int ls_dof_idx;

      // optional parameters for narrow band
      const VectorType *solution_level_set;

      AlignedVector<VectorizedArray<double>> damping;
    };

    /**
     * Matrix-free
     *
     * For a given @param cell, compute the cell_size dependent filter parameter
     *
     *    scale_factor * h^2
     *
     * using a given @param scale_factor.
     *
     * @todo: move to normal_vector_utils.hpp
     */
    template <int dim>
    inline VectorizedArray<double>
    compute_cell_size_dependent_filter_parameter_mf(const ScratchData<dim> &scratch_data,
                                                    const unsigned int      dof_idx,
                                                    const unsigned int      cell_idx,
                                                    const double            scale_factor)
    {
      const double n_subdivisions =
        scratch_data.is_FE_Q_iso_Q_1(dof_idx) ? scratch_data.get_degree(dof_idx) : 1;
      return Utilities::fixed_power<2>(
               std::max(VectorizedArray<double>(scratch_data.get_min_cell_size()),
                        scratch_data.get_cell_sizes()[cell_idx] / (double)n_subdivisions)) *
             scale_factor;
    }

    /**
     * For a given @param cell, compute the cell_size dependent filter parameter
     *
     *    scale_factor * h^2
     *
     * using a given @param scale_factor.
     *
     * @todo: move to normal_vector_utils.hpp
     */
    template <int dim, typename cell_type>
    double
    compute_cell_size_dependent_filter_parameter(const ScratchData<dim> &scratch_data,
                                                 const unsigned int      dof_idx,
                                                 const cell_type         cell,
                                                 const double            scale_factor)
    {
      const double n_subdivisions =
        scratch_data.is_FE_Q_iso_Q_1(dof_idx) ? scratch_data.get_degree(dof_idx) : 1;

      return Utilities::fixed_power<2>(
               std::max(scratch_data.get_min_cell_size(),
                        cell->diameter() / (std::sqrt(dim) * n_subdivisions))) *
             scale_factor;
    }
  } // namespace LevelSet
} // namespace MeltPoolDG
