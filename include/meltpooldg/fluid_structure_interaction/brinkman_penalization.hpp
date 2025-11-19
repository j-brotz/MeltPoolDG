#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>


namespace MeltPoolDG
{

  template <int dim, typename number, typename ObstacleType>
  struct BrinkmanObstacleForce
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor. Stores all relevant data internally.
     *
     * @param solution Reference to the solution of the flow field.
     * @param mf Matrix free object used by the compressible flow solver.
     * @param dof_idx DoF index within the matrix-free object.
     * @param quad_idx Quadrature index within the matrix-free object.
     * @param scratch_data Object for caching relevant data for the penalty term computation.
     */
    BrinkmanObstacleForce(const ObstacleField<dim, number, ObstacleType> &obstacle_field,
                          const VectorType                               &solution,
                          const dealii::MatrixFree<dim, number>          &mf,
                          const unsigned                                  dof_idx,
                          const unsigned                                  quad_idx,
                          const BrinkmanPenalizationData<number>    &data);

    /**
     * Compute the force from the fluid on all obstacles in the given obstacle field @param obstacle_field.
     * This is done by evaluating the Brinkman penalization terms at all (fluid) quadrature points
     * lying in the obstacle volume, multiplying by the corresponding quadrature weight and
     * computing the sum of the individual contributions. The final result is then added to the
     * force property of the corresponding obstacle.
     */
    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const;

  private:
    /// Brinkman penalization data
    const BrinkmanPenalizationData<number> brinkman_penalization_data;

    /// Cached cell data for computing Brinkman penalty term.
    mutable CellObstacleCache<dim, number, ObstacleType> cell_obstacle_cache;

    /// Matrix free object used by the compressible flow solver.
    const dealii::MatrixFree<dim, number> &matrix_free;

    /// DoF index within the matrix-free object.
    unsigned dof_idx;

    /// Quadrature index within the matrix-free object.
    unsigned quad_idx;

    /// Solution of the flow field.
    const VectorType &solution;
  };


  /**
   * Implementation of the Brinkman penalization force for compressible flows
   * advanced in time using explicit time integration.
   *
   * The penalization force is computed as
   * \f[
   * f^{vp}_\rho &= 0,\\
   * f^{vp}_\bm{m} &= \frac{\xi}{\eta} \left(\rho_f \mathbf{u}_s - \mathbf{m}_f \right), \\
   * f^{vp}_e &= \frac{\xi}{\eta} \left(\rho_f \mathbf{u}_s - \mathbf{m}_f \right)\frac{m_j}{\rho},
   * \\
   * \f]
   * where:
   * - \f(\xi\f) is the mask function,
   * - \f(\eta\f) is the permeability,
   * - \f(\rho_f\f) is the local fluid density,
   * - \f(\mathbf{u}_s\f) is the local obstacle velocity,
   * - \f(\mathbf{m}_f\f) is the current fluid momentum.
   */
  template <int dim, typename number, typename ObstacleType>
  struct BrinkmanPenalizationResidualContribution final
    : public Flow::AdditionalCellAndQuadOperation<dim, number>
  {
    using ConservedVariablesType = Flow::CompressibleFlowTypes::ConservedVariablesType<dim, number>;

  public:
    /**
     * Constructor that stores relevant data internally and uses the default mask function
     * for the penalty term computation. The default mask function returns 1 for points inside
     * the obstacle volume and 0 outside.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param brinkman_penalization_data Data required for computing the Brinkman penalization term.
     * @param mask_function User-defined mask function used in the penalty term evaluation.
     */
    BrinkmanPenalizationResidualContribution(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const BrinkmanPenalizationData<number>    &brinkman_penalization_data);

    /**
     * Identifies and stores all relevant particles for the given cell batch, to be used
     * during the quadrature operation.
     *
     * This function is responsible for locating particles that are relevant to the current cell
     * batch and caching them internally. The cached particles are then accessed during the later
     * quadrature computation.
     *
     * @param matrix_free The MatrixFree object relevant for the cell batch.
     * @param cell_batch_id The index of the cell batch to process.
     * @param dof_idx Index of the relevant dof handler in the matrix-free object.
     */
    void
    cell_operation(const dealii::MatrixFree<dim, number> &matrix_free,
                   const unsigned int                     cell_batch_id,
                   const unsigned int                     dof_idx) override;

    /**
     * Computes the Brinkman penalty term at the specified (vectorized) points, typically
     * quadrature points.
     *
     * The computation considers only those obstacles that have been cached internally, as
     * determined by the most recent call to @p cell_operation().
     *
     * @param time_step_size Current time step size.
     * @param q_point Coordinates at which the penalty term is to be evaluated.
     * @param w_q Conserved variables evaluated at the given coordinates.
     * @return The computed Brinkman penalty term at the specified points.
     */
    ConservedVariablesType
    quad_operation(number                                                     time_step_size,
                   const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                   const ConservedVariablesType                              &w_q) override;

  private:
    /// Brinkman penalization data
    const BrinkmanPenalizationData<number> brinkman_penalization_data;

    /// Cached cell data for computing Brinkman penalty term.
    mutable CellObstacleCache<dim, number, ObstacleType> cell_obstacle_cache;
  };


  /**
   * Implementation of the Brinkman penalization force Jacobian for compressible flows.
   *
   * The penalization force is computed as
   * \f[
   * f^{vp}_\rho &= 0,\\
   * f^{vp}_\bm{m} &= \frac{\xi}{\eta} \left(\rho_f \mathbf{u}_s - \mathbf{m}_f \right), \\
   * f^{vp}_e &= \frac{\xi}{\eta} \left(\rho_f \mathbf{u}_s - \mathbf{m}_f \right)\frac{m_j}{\rho},
   * \\
   * \f]
   * where:
   * - \f(\xi\f) is the mask function,
   * - \f(\eta\f) is the permeability,
   * - \f(\rho_f\f) is the local fluid density,
   * - \f(\mathbf{u}_s\f) is the local obstacle velocity,
   * - \f(\mathbf{m}_f\f) is the current fluid momentum.
   */
  template <int dim, typename number, typename ObstacleType>
  struct BrinkmanPenalizationJacobianContribution final
    : public Flow::AdditionalCellAndQuadOperationJacobian<dim, number>
  {
    using ConservedVariablesType = Flow::CompressibleFlowTypes::ConservedVariablesType<dim, number>;

  public:
    /**
     * Constructor that stores relevant data internally and uses the default mask function
     * for the penalty term computation. The default mask function returns 1 for points inside
     * the obstacle volume and 0 outside.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param brinkman_penalization_data Data required for computing the Brinkman penalization term.
     */
    BrinkmanPenalizationJacobianContribution(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const BrinkmanPenalizationData<number>    &brinkman_penalization_data);

    /**
     * Identifies and stores all relevant particles for the given cell batch, to be used
     * during the quadrature operation.
     *
     * This function is responsible for locating particles that are relevant to the current cell
     * batch and caching them internally. The cached particles are then accessed during the later
     * quadrature computation.
     *
     * @param matrix_free The MatrixFree object relevant for the cell batch.
     * @param cell_batch_id The index of the cell batch to process.
     */
    void
    cell_operation(const dealii::MatrixFree<dim, number> &matrix_free,
                   const unsigned int                     cell_batch_id) override;

    /**
     * Computes the Jacobian * delta_w of the Brinkman penalty term at the specified
     * (vectorized) points, typically quadrature points.
     *
     * The computation considers only those obstacles that have been cached internally, as
     * determined by the most recent call to @p cell_operation().
     *
     * @param time_step_size Current time step size.
     * @param q_point Coordinates at which the penalty term is to be evaluated.
     * @param w_q Conserved variables evaluated at the given coordinates.
     * @param delta_w_q Change in conserved variables at the given coordinates.
     * @return The computed Brinkman penalty term at the specified points.
     */
    ConservedVariablesType
    quad_operation(number                                                     time_step_size,
                   const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                   const ConservedVariablesType                              &w_q,
                   const ConservedVariablesType                              &delta_w_q) override;

  private:
    /// Brinkman penalization data
    const BrinkmanPenalizationData<number> brinkman_penalization_data;

    /// Cached cell data for computing Brinkman penalty term.
    mutable CellObstacleCache<dim, number, ObstacleType> cell_obstacle_cache;
  };
} // namespace MeltPoolDG
