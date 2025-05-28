#pragma once

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>

#include <functional>


namespace MeltPoolDG
{
  /**
   * @brief Implementation of the Brinkman penalization force for compressible flows
   * advanced in time using explicit time integration.
   *
   * The penalization force is computed as
   * \f[
   * \frac{\xi}{\eta} \left(\rho_f \mathbf{u}_s - \mathbf{m}_f \right)
   * \f],
   * where:
   * - \f(\xi\f) is the mask function,
   * - \f(\eta\f) is the permeability,
   * - \f(\rho_f\f) is the local fluid density,
   * - \f(\mathbf{u}_s\f) is the local obstacle velocity,
   * - \f(\mathbf{m}_f\f) is the current fluid momentum.
   */

  template <int dim, typename number, typename ObstacleType>
  struct BrinkmanPenalizationFluidForceRightHandSideContribution final
    : public Flow::ExternalFluidForcesRightHandSideContribution<dim, number>
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
     */
    BrinkmanPenalizationFluidForceRightHandSideContribution(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data)
      : cell_relevant_obstacles(ObstacleType::n_obstacle_properties)
      , obstacle_handler(obstacle_handler)
      , brinkman_penalization_data(brinkman_penalization_data)
    {}

    /**
     * @brief Constructor providing the flexibility to specify a custom mask function used in the
     * computation of the Brinkman penalization term.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param brinkman_penalization_data Data required for computing the Brinkman penalization term.
     * @param mask_function User-defined mask function used in the penalty term evaluation.
     */
    BrinkmanPenalizationFluidForceRightHandSideContribution(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data,
      std::function<
        dealii::VectorizedArray<number>(dealii::Point<dim, dealii::VectorizedArray<number>> &,
                                        dealii::Particles::ParticleAccessor<dim> &)> mask_function)
      : obstacle_handler(obstacle_handler)
      , brinkman_penalization_data(brinkman_penalization_data)
      , mask_function(mask_function)
    {}

    /**
     * @brief Identifies and stores all relevant particles for the given cell batch, to be used
     * during the quadrature operation.
     *
     * This function is responsible for locating particles that are relevant to the current cell
     * batch and caching them internally. The cached particles are then accessed during the
     * subsequent quadrature computation.
     *
     * @param matrix_free The MatrixFree object relevant for the cell batch.
     * @param cell_batch_id The index of the cell batch to process.
     * @param n_lanes Number of lanes (i.e., cells) in the cell batch.
     */
    void
    cell_operation(const dealii::MatrixFree<dim, number> &matrix_free,
                   const unsigned int                     cell_batch_id,
                   const unsigned int n_lanes = dealii::VectorizedArray<number>::size) override;

    /**
     * @brief Computes the Brinkman penalty term at the specified (vectorized) points, typically
     * quadrature points.
     *
     * The computation considers only those obstacles that have been cached internally, as
     * determined by the most recent call to @p cell_operation().
     *
     * @param q_point Coordinates at which the penalty term is to be evaluated.
     * @param w_q Conserved variables evaluated at the given coordinates.
     * @return The computed Brinkman penalty term at the specified points.
     */
    ConservedVariablesType
    quad_operation(const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                   const ConservedVariablesType                              &w_q) override;

  private:
    /// Property pool used to cache obstacle properties relevant for the current cell batch.
    dealii::Particles::PropertyPool<dim> cell_relevant_obstacles;

    /// Reference to the obstacle handler managing all obstacles in the domain.
    const ObstacleField<dim, number, ObstacleType> &obstacle_handler;

    /// Data structure containing parameters required for computing the Brinkman penalization term.
    const BrinkmanPenalizationData<number> &brinkman_penalization_data;

    /// Mask function used in the computation of the Brinkman penalization term. The default
    /// implementation returns 1.0 if the point lies inside the obstacle's radius, and 0.0
    /// otherwise.
    std::function<
      dealii::VectorizedArray<number>(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                                      dealii::Particles::PropertyPool<dim> &,
                                      const typename dealii::Particles::PropertyPool<dim>::Handle)>
      mask_function = [](const dealii::Point<dim, dealii::VectorizedArray<number>>  &location,
                         dealii::Particles::PropertyPool<dim>                       &property_pool,
                         const typename dealii::Particles::PropertyPool<dim>::Handle handle)
      -> dealii::VectorizedArray<number> {
      // Default mask function: returns 1 if the point lies within the obstacle's radius,
      // 0 otherwise.
      dealii::Point<dim, dealii::VectorizedArray<number>> vectorized_obstacle_location;
      for (auto i = 0; i < dim; ++i)
        vectorized_obstacle_location[i] =
          dealii::VectorizedArray<number>(property_pool.get_location(handle)[i]);
      const dealii::VectorizedArray<number> distance =
        location.distance(vectorized_obstacle_location);
      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
        distance,
        dealii::VectorizedArray<number>(
          ObstacleType::get_property(property_pool, handle, ObstacleType::Properties::radius)),
        dealii::VectorizedArray<number>(1.),
        dealii::VectorizedArray<number>(0.));
    };
  };
} // namespace MeltPoolDG


/*************************************************************************************************
 * Inlined functions
 *************************************************************************************************/
template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::BrinkmanPenalizationFluidForceRightHandSideContribution<dim, number, ObstacleType>::
  cell_operation(const dealii::MatrixFree<dim, number> &matrix_free,
                 const unsigned int                     cell_batch_id,
                 const unsigned int                     n_lanes)
{
  obstacle_handler.get_obstacles_in_cell_batch(cell_relevant_obstacles,
                                               matrix_free,
                                               cell_batch_id,
                                               n_lanes);
}

template <int dim, typename number, typename ObstacleType>
auto
MeltPoolDG::BrinkmanPenalizationFluidForceRightHandSideContribution<dim, number, ObstacleType>::
  quad_operation(const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                 const ConservedVariablesType &w_q) -> ConservedVariablesType
{
  // In order to loop over all obstacles in the property pool we need to ensure that there are no
  // 'dead' slots inside the property pool which could provide wrong data. In other words we do
  // not want empty slots resulting from deregistering particles.
  // TODO: What happens if we removed one handle and added one with a new handle, then the for
  // loop would fail but the assert wouldn't be triggered!
  Assert(cell_relevant_obstacles.n_slots() == cell_relevant_obstacles.n_registered_slots(),
         dealii::ExcInternalError());

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> fluid_velocity;
  for (int d = 0; d < dim; ++d)
    fluid_velocity[d] = w_q[d + 1];

  // TODO Penalty term for mass balance equation?

  dealii::Tensor<1, dim, dealii::VectorizedArray<number>> momentum_fluid_force;
  for (typename dealii::Particles::PropertyPool<dim>::Handle obstacle_handle = 0;
       obstacle_handle < cell_relevant_obstacles.n_slots();
       ++obstacle_handle)
    {
      // TODO: Consider obstacle velocity
      momentum_fluid_force = -mask_function(q_point, cell_relevant_obstacles, obstacle_handle) /
                             brinkman_penalization_data.permeability * fluid_velocity;
    }

  ConservedVariablesType fluid_force;
  for (int d = 0; d < dim; ++d)
    fluid_force[d + 1] = momentum_fluid_force[d];
  return fluid_force;
}
