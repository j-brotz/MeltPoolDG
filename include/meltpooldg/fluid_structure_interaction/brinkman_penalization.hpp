#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/utils.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_util.hpp>
#include <meltpooldg/particles/matrix_free_particle_cache.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <memory>


namespace MeltPoolDG
{

  template <int dim, typename number, typename ObstacleType>
  struct BrinkmanObstacleForce
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor. Stores all relevant data internally.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param solution Reference to the solution of the flow field.
     * @param matrix_free MatrixFree object and corresponding relevant indices.
     * @param data Object for caching relevant data for the penalty term computation.
     */
    BrinkmanObstacleForce(
      const ObstacleField<dim, number, ObstacleType>                           &obstacle_handler,
      const VectorType                                                         &solution,
      const MatrixFreeContext<dim, number>                                     &matrix_free,
      const BrinkmanPenalizationData<number>                                   &data,
      std::shared_ptr<Particles::MatrixFreeCellBatchParticleCache<dim, number>> particle_cache);

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

    /// Matrix free object and corresponding relevant indices used by the compressible flow solver.
    const MatrixFreeContext<dim, number> matrix_free;

    /// Solution of the flow field.
    const VectorType &solution;

    std::shared_ptr<Particles::MatrixFreeCellBatchParticleCache<dim, number>> particle_cache;
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
    : public CompressibleFlow::ExternalFlowForce<dim, number>
  {
    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;

  public:
    /**
     * Constructor that stores relevant data internally and uses the default mask function
     * for the penalty term computation. The default mask function returns 1 for points inside
     * the obstacle volume and 0 outside.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param brinkman_penalization_data Data required for computing the Brinkman penalization term.
     */
    BrinkmanPenalizationResidualContribution(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data,
      std::shared_ptr<Particles::MatrixFreeCellBatchParticleCache<dim, number>> particle_cache);

    /**
     * This function evaluates the Brinkman penalty term at a set of vectorized points, typically
     * quadrature points. Since successive calls often operate on the same set of cells, the
     * function internally caches the obstacles relevant to the provided cells. The cache is updated
     * automatically if the cell set differs from that used in the previous call.
     *
     * @param time_step_size Current time step size.
     * @param cell_iterators Container holding an iterator to the cells associated with the provided
     * points.
     * @param q_point Coordinates at which the penalty term is to be evaluated.
     * @param w_q Conserved variables evaluated at the given coordinates.
     * @return The computed Brinkman penalty term at the specified points.
     */
    ConservedVariablesType
    value(
      number                                                                         time_step_size,
      const unsigned int                                                             cell_batch_id,
      const dealii::Point<dim, dealii::VectorizedArray<number>>                     &q_point,
      const ConservedVariablesType                                                  &w_q) override;

  private:
    /// Brinkman penalization data
    const BrinkmanPenalizationData<number> brinkman_penalization_data;

    /// Cached cell data for computing Brinkman penalty term.
    std::shared_ptr<Particles::MatrixFreeCellBatchParticleCache<dim, number>> particle_cache;

    number inverse_permeability;
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
    : public CompressibleFlow::ExternalFlowForceJacobian<dim, number>
  {
    using ConservedVariablesType = CompressibleFlow::ConservedVariablesType<dim, number>;

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
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data,
      std::shared_ptr<Particles::MatrixFreeCellBatchParticleCache<dim, number>> particle_cache);

    /**
     * This function evaluates the Jacobian of the Brinkman penalty term at a set of vectorized
     * points, typically quadrature points. Since successive calls often operate on the same set of
     * cells, the function internally caches the obstacles relevant to the provided cells. The cache
     * is updated automatically if the cell set differs from that used in the previous call.
     *
     * @param time_step_size Current time step size.
     * @param cell_iterators Container holding an iterator to the cells associated with the provided
     * points.
     * @param q_point Coordinates at which the penalty term is to be evaluated.
     * @param w_q Conserved variables evaluated at the given coordinates.
     * @param delta_w_q Change in conserved variables at the given coordinates.
     * @return The computed Brinkman penalty term at the specified points.
     */
    ConservedVariablesType
    value(
      number                                                                         time_step_size,
      const unsigned int                                                             cell_batch_id,
      const dealii::Point<dim, dealii::VectorizedArray<number>>                     &q_point,
      const ConservedVariablesType                                                  &w_q,
      const ConservedVariablesType &delta_w_q) override;

  private:
    /// Brinkman penalization data
    const BrinkmanPenalizationData<number> brinkman_penalization_data;

    std::shared_ptr<Particles::MatrixFreeCellBatchParticleCache<dim, number>> particle_cache;
  };
} // namespace MeltPoolDG
