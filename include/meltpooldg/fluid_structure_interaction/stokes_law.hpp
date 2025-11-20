#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  struct StokesLawSphericalParticleForce
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * @brief Constructor. Stores all relevant data internally.
     *
     * @param solution Reference to the solution of the flow field.
     * @param matrix_free MatrixFree object and corresponding relevant indices.
     * @param scratch_data Object for caching relevant data for the penalty term computation.
     */
    StokesLawSphericalParticleForce(const VectorType                     &solution,
                                    const MatrixFreeContext<dim, number> &matrix_free,
                                    const number                          dynamic_viscosity);

    /**
     * Computes the hydrodynamic force exerted by the fluid on all obstacles in the provided
     * obstacle field. The force on each obstacle is evaluated using Stokes' law (valid for Re <<
     * 1!), based on the translational velocity of the particle at its center and the fluid velocity
     * at the same location:
     * @f[
     *   \mathbf{F} = 6 \pi \mu R \left( \mathbf{u}_f - \mathbf{u}_p \right).
     * @f]
     *
     * @param obstacle_field Reference to the obstacle field containing the obstacles for which
     * the force will be computed.
     */

    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const;

  private:
    /// Matrix free object and corresponding relevant indices used by the compressible flow solver.
    const MatrixFreeContext<dim, number> matrix_free;

    /// Solution of the flow field.
    const VectorType &solution;

    /// Dynamic viscosity of the fluid.
    const number dynamic_viscosity;
  };

  /**
   * This class applies a penalty force to the fluid equations to account for the presence of
   * particles. The penalty term is derived from Stokes' law and effectively introduces the Stokes
   * drag as a forcing term within the fluid momentum equations.
   *
   * The approach follows the method proposed by Chen et al.
   * (https://doi.org/10.1016/j.actamat.2020.06.033) in the context of vapor–particle interactions
   * during metal additive manufacturing.
   */
  template <int dim, typename number, typename ObstacleType>
  struct StokesLawFluidForce final : public Flow::AdditionalCellAndQuadOperation<dim, number>
  {
    using ConservedVariablesType = Flow::CompressibleFlowTypes::ConservedVariablesType<dim, number>;
    using VectorType             = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    /**
     * Constructor that stores relevant data internally.
     *
     * @param solution Reference to the solution of the flow field.
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param dynamic_viscosity Dynamic viscosity of the fluid.
     */
    StokesLawFluidForce(const VectorType                         &solution,
                        ObstacleField<dim, number, ObstacleType> &obstacle_handler,
                        const number                              dynamic_viscosity);

    /**
     * This function computes the penalty term for the cells in the given cell batch. The method
     * applied here restricts the penalty force to the cell that contains the corresponding particle
     * center. Using Stokes' law, the hydrodynamic force on the particle and, by Newton’s third law,
     * the equal and opposite force exerted on the fluid, is evaluated and incorporated as a penalty
     * term in the fluid momentum equations.
     *
     * The force is given by Stokes' law:
     * @f[
     *   \mathbf{F} = 6 \pi \mu R \left( \mathbf{u}_f - \mathbf{u}_p \right).
     * @f]
     *
     * The resulting momentum penalty force is stored internally and used in the subsequent call to
     * quad_operation().
     *
     * @param matrix_free MatrixFree object and corresponding relevant indices.
     * @param cell_batch_id The index of the cell batch to process.
     */
    void
    cell_operation(const MatrixFreeContext<dim, number> &matrix_free,
                   unsigned int                          cell_batch_id) override;

    /**
     * Computes the penalty term at the given points. This function effectively returns the penalty
     * contribution to the momentum balance that was previously computed during the call to
     * cell_operation(). The corresponding penalty contributions to the mass and energy balance
     * equations are zero.
     *
     * @param time_step_size The current time-step size.
     * @param q_point Coordinates at which the penalty term is evaluated.
     * @param w_q Conserved variables evaluated at the given coordinates.
     * @return The computed penalty term at the specified points.
     */
    ConservedVariablesType
    quad_operation(number                                                     time_step_size,
                   const dealii::Point<dim, dealii::VectorizedArray<number>> &q_point,
                   const ConservedVariablesType                              &w_q) override;

  private:
    /// Solution of the flow field.
    const VectorType &solution;

    /// Dynamic viscosity of the fluid.
    const number dynamic_viscosity;

    /// Momentum penalty force for the current cell batch.
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> cell_penalty_force;

    /// Reference to the obstacle handler managing obstacles in the domain.
    ObstacleField<dim, number, ObstacleType> &obstacle_handler;
  };

} // namespace MeltPoolDG