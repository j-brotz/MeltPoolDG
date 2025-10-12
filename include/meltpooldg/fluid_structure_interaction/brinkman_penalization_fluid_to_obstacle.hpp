#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  struct BrinkmanObstacleForce
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor. Stores all relevant data internally.
     *
     * @param solution Reference to the solution of the flow field.
     * @param mf Matrix free object used by the compressible flow solver.
     * @param dof_idx DoF index within the matrix-free object.
     * @param quad_idx Quadrature index within the matrix-free object.
     * @param scratch_data Object for caching relevant data for the penalty term computation.
     */
    BrinkmanObstacleForce(
      const VectorType                                                &solution,
      const dealii::MatrixFree<dim, number>                           &mf,
      const unsigned                                                   dof_idx,
      const unsigned                                                   quad_idx,
      BrinkmanPenalizationCellScratchData<dim, number, ObstacleType> &&scratch_data);

    /**
     * Compute the force from the fluid on all obstacles in the given obstacle field @param obstacle_field.
     * This is done by evaluating the Brinkman penalization terms at all (fluid) quadrature points
     * lying in the obstacle volume, multiplying by the corresponding quadrature weigth and
     * computing the sum of the individual contributions. The final result is then added to the
     * force property of the corresponding obstacle.
     */
    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const;

  private:
    /// Cached cell data for computing Brinkman penalty term.
    mutable BrinkmanPenalizationCellScratchData<dim, number, ObstacleType> brinkman_scratch_data;

    /// Matrix free object used by the compressible flow solver.
    const dealii::MatrixFree<dim, number> &matrix_free;

    /// DoF index within the matrix-free object.
    unsigned dof_idx;

    /// Quadrature index within the matrix-free object.
    unsigned quad_idx;

    /// Solution of the flow field.
    const VectorType &solution;
  };
} // namespace MeltPoolDG