/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

// for parallelization
#  include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#  include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#  include <meltpooldg/interface/operator_base.hpp>
#  include <meltpooldg/interface/parameters.hpp>
#  include <meltpooldg/normal_vector/normal_vector_operation_base.hpp>
#  include <meltpooldg/utilities/utility_functions.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/block_matrix_extension.h>
#  include <adaflo/level_set_okz_compute_normal.h>
#  include <adaflo/level_set_okz_preconditioner.h>
#  include <adaflo/util.h>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    template <int dim>
    class NormalVectorOperationAdaflo : public NormalVectorOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      /**
       * Constructor.
       */
      NormalVectorOperationAdaflo(const ScratchData<dim>         &scratch_data,
                                  const int                       advec_diff_dof_idx,
                                  const int                       normal_vec_dof_idx,
                                  const int                       normal_vec_quad_idx,
                                  const VectorType               &advected_field_in,
                                  const NormalVectorData<double> &data_in,
                                  const double                    reinit_scale_factor_epsilon);

      void
      reinit() override;

      /**
       * Solver time step
       */
      void
      solve() override;

      /**
       *  getter
       */
      const LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() const override;

      LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() override;

      LevelSetOKZSolverComputeNormal<dim> &
      get_adaflo_obj();

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    private:
      void
      set_adaflo_parameters(const NormalVectorData<double> &normal_vec_data,
                            const double                    epsilon,
                            const int                       advec_diff_dof_idx,
                            const int                       normal_vec_dof_idx,
                            const int                       normal_vec_quad_idx);

      void
      initialize_vectors();

    private:
      void
      create_operator();

      const ScratchData<dim> &scratch_data;

      const VectorType &advected_field;

      /**
       *  Vectors for computing the normals
       */
      BlockVectorType normal_vector_field;
      BlockVectorType rhs;
      /**
       * Adaflo parameters for the level set problem
       */
      LevelSetOKZSolverComputeNormalParameter normal_vec_adaflo_params;
      /**
       * Reference to the actual advection diffusion solver from adaflo
       */
      std::shared_ptr<LevelSetOKZSolverComputeNormal<dim>> normal_vec_operation;

      /**
       *  Diagonal preconditioner @todo
       */
      DiagonalPreconditioner<double>         preconditioner;
      std::shared_ptr<BlockMatrixExtension>  projection_matrix;
      std::shared_ptr<BlockILUExtension>     ilu_projection_matrix;
      AlignedVector<VectorizedArray<double>> cell_diameters;
      double                                 cell_diameter_min;
      double                                 cell_diameter_max;
      double                                 epsilon_used;
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
#endif
