#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>

// MeltPoolDG
#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/level_set/normal_vector_operation_base.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/predictor.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  /**
   *  This function calculates the normal vector of the current level set function being
   *  the solution of an intermediate projection step
   *
   *              (w, n_ϕ)  + η_n (∇w, ∇n_ϕ)  = (w,∇ϕ)
   *                      Ω                 Ω            Ω
   *
   *  with test function w, the normal vector n_ϕ, damping parameter η_n and the
   *  level set function ϕ.
   *
   *    !!!!
   *          the normal vector field is not normalized to length one,
   *          it actually represents the gradient of the level set
   *          function
   *    !!!!
   */

  template <int dim, typename number>
  class NormalVectorOperation : public NormalVectorOperationBase<dim, number>
  {
  private:
    using VectorType       = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType  = dealii::LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;

  public:
    NormalVectorOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                          const NormalVectorData<number>      &normal_vector_data,
                          const VectorType                    &solution_level_set,
                          const std::array<unsigned int, dim> &normal_dof_indices_per_block_in,
                          const unsigned int                   normal_no_bc_dof_idx_in,
                          const unsigned int                   normal_quad_idx_in,
                          const unsigned int                   ls_dof_idx_in);

    void
    reinit() override;

    void
    solve() override;

    /**
     * @brief Create map containing local DoF indices of DoFs affected by the wetting boundary
     * condition and the values of each component of the interface normal vector.
     */
    void
    create_wetting_constraints();

    /**
     * @brief TODO AA
     */
    void
    create_contact_angle_constraints();

    const BlockVectorType &
    get_solution_normal_vector() const override;

    BlockVectorType &
    get_solution_normal_vector() override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    /**
     * @brief Set wetting boundary IDs and functions map.
     *
     * @param p_wetting_bc_map Map containing functions corresponding to different boundaries that
     * express the value of the normal vector at a given point of a boundary for Dirichlet boundary
     * conditions.
     */
    void
    set_wetting_bc_map(const std::map<dealii::types::boundary_id,
                                      std::shared_ptr<dealii::Function<dim>>> &p_wetting_bc_map);

    /**
     * @brief Set wetting boundary IDs and functions map.
     *
     * @param p_contact_angle_bc_map Map containing functions corresponding to different boundaries that
     * express the value of the normal vector at a given point of a boundary for Dirichlet boundary
     * conditions.
     */
    void
    set_contact_angle_bc_map(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &p_contact_angle_bc_map);

  private:
    /**
     * This function creates the normal vector operator for assembling the system operator (either
     * matrix based or matrix free) and the right-hand side.
     */
    void
    create_operator();

  private:
    const ScratchData<dim, dim, number> &scratch_data;
    const NormalVectorData<number>       normal_vector_data;
    const VectorType                    &solution_level_set;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim,dim,number> object is selected. This is important when
     * ScratchData<dim,dim,number> holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int                  normal_no_bc_dof_idx;
    const std::array<unsigned int, dim> normal_dof_indices_per_block;
    const unsigned int                  normal_quad_idx;
    const unsigned int                  ls_dof_idx;

    TimeIntegration::SolutionHistory<BlockVectorType> solution_history;

    std::unique_ptr<Predictor<BlockVectorType, number>> predictor;
    /*
     *    This is the primary solution variable of this module, which will be also publicly
     *    accessible for output_results.
     */
    BlockVectorType solution_normal_vector_predictor;
    BlockVectorType rhs;
    /*
     * This pointer will point to your user-defined normal vector operator.
     */
    std::unique_ptr<NormalVectorOperator<dim, number>> normal_vector_operator;
    /*
     * Preconditioner for the curvature operator
     */
    Preconditioner<dim, BlockVectorType, number> preconditioner;

    /*
     * Wetting Boundary conditions
     */
    /** Map containing boundary IDs where a wetting boundary condition is applied and the functions
       giving the normal vector values.*/
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> wetting_bc_map;
    std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      contact_angle_bc_map;
    /** Pair regrouping local DoF indices and values of the normal vector for wetting boundary
       conditions. */
    std::pair<std::vector<unsigned int>, std::vector<std::vector<number>>>
      wetting_constraints_indices_and_values;
    std::pair<std::vector<unsigned int>, std::vector<std::vector<number>>>
      contact_angle_constraints_indices_and_values;
  };
} // namespace MeltPoolDG::LevelSet
