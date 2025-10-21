#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>

namespace MeltPoolDG
{
  /**
   * @brief Implementation of the Brinkman penalization force for compressible flows
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
     * @brief Constructor that stores relevant data internally and uses the default mask function
     * for the penalty term computation. The default mask function returns 1 for points inside
     * the obstacle volume and 0 outside.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param brinkman_penalization_data Data required for computing the Brinkman penalization term.
     * @param mask_function User-defined mask function used in the penalty term evaluation.
     */
    BrinkmanPenalizationResidualContribution(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const FluidStructureInteractionData<number>    &brinkman_penalization_data,
      typename BrinkmanPenalizationCellScratchData<dim, number, ObstacleType>::MaskFunctionType
        mask_function =
          static_cast<typename BrinkmanPenalizationCellScratchData<dim, number, ObstacleType>::
                        MaskFunctionType>(
            discontinuous_mask_function<dim, dealii::VectorizedArray<number>, ObstacleType>));

    /**
     * @brief Identifies and stores all relevant particles for the given cell batch, to be used
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
     * @brief Computes the Brinkman penalty term at the specified (vectorized) points, typically
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
    BrinkmanPenalizationCellScratchData<dim, number, ObstacleType> brinkman_cell_scratch_data;
  };


  /**
   * @brief Implementation of the Brinkman penalization force Jacobian for compressible flows.
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
     * @brief Constructor that stores relevant data internally and uses the default mask function
     * for the penalty term computation. The default mask function returns 1 for points inside
     * the obstacle volume and 0 outside.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param brinkman_penalization_data Data required for computing the Brinkman penalization term.
     * @param mask_function User-defined mask function used in the penalty term evaluation.
     */
    BrinkmanPenalizationJacobianContribution(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const FluidStructureInteractionData<number>    &brinkman_penalization_data,
      typename BrinkmanPenalizationCellScratchData<dim, number, ObstacleType>::MaskFunctionType
        mask_function =
          static_cast<typename BrinkmanPenalizationCellScratchData<dim, number, ObstacleType>::
                        MaskFunctionType>(
            discontinuous_mask_function<dim, dealii::VectorizedArray<number>, ObstacleType>));

    /**
     * @brief Identifies and stores all relevant particles for the given cell batch, to be used
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
     * @brief Computes the Jacobian * delta_w of the Brinkman penalty term at the specified
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
    BrinkmanPenalizationCellScratchData<dim, number, ObstacleType> brinkman_cell_scratch_data;
  };
} // namespace MeltPoolDG
