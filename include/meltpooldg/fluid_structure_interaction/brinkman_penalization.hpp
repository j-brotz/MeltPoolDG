#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/particles/property_pool.h>

#include <meltpooldg/flow/flow_utils.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>


namespace MeltPoolDG
{

  template <int dim, typename number, typename ObstacleType>
  struct BrinkmanObstacleForce
  {
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;

    /**
     * Constructor. Stores all relevant data internally.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param solution Reference to the solution of the flow field.
     * @param matrix_free MatrixFree object and corresponding relevant indices.
     * @param scratch_data Object for caching relevant data for the penalty term computation.
     */
    BrinkmanObstacleForce(const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
                          const VectorType                               &solution,
                          const MatrixFreeContext<dim, number>           &matrix_free,
                          const BrinkmanPenalizationData<number>         &data,
                          FlowSolverType                                  flow_solver_type,
                          number constant_density = dealii::numbers::signaling_nan<number>());

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

    /// Matrix free object and corresponding relevant indices used by the compressible flow solver.
    const MatrixFreeContext<dim, number> matrix_free;

    /// Solution of the flow field.
    const VectorType &solution;

    ///
    const FlowSolverType flow_solver_type;

    const number constant_density;

    static constexpr unsigned torque_size = ObstacleType::size_angular_velocity;

    static constexpr int n_components_compressible = dim + 2;

    static constexpr int n_components_incompressible = dim;

    template <int n_components>
    void
    local_apply_cell(const dealii::MatrixFree<dim, number> &,
                     dealii::Tensor<1, dim, number> &,
                     const VectorType                     &solution,
                     const std::pair<unsigned, unsigned>  &cell_range,
                     dealii::Particles::PropertyPool<dim> &global_particle_properties) const
    {
      FECellIntegrator<dim, n_components, number> phi(matrix_free.mf,
                                                      matrix_free.dof_idx,
                                                      matrix_free.quad_idx);

      for (unsigned cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(solution, dealii::EvaluationFlags::values);

          for (const unsigned int q : phi.quadrature_point_indices())
            {
              dealii::Tensor<1, n_components, VectorizedArrayType> w;

              // This needs to be done as for dim==1 the solution vector of the incompressible flow
              // only has a single component. For this case the FeCellIntegrator returns a scalar
              // type and not a tensor.
              if constexpr (dim == 1 and n_components == 1)
                w[0] = phi.get_value(q);
              else
                w = phi.get_value(q);

              for (unsigned int src_handle = 0;
                   src_handle < global_particle_properties.n_registered_slots();
                   ++src_handle)
                {
                  dealii::Tensor<1, dim, VectorizedArrayType>         force;
                  dealii::Tensor<1, torque_size, VectorizedArrayType> torque;

                  const auto q_point = phi.quadrature_point(q);

                  const auto mask = mask_function<dim, VectorizedArrayType, ObstacleType>(
                    brinkman_penalization_data.mask_function_type,
                    q_point,
                    global_particle_properties,
                    src_handle);

                  force = compute_penalty_force(w,
                                                ObstacleType::get_velocity(
                                                  global_particle_properties, src_handle, q_point),
                                                mask) *
                          phi.JxW(q);

                  torque = compute_torque(force,
                                          ObstacleType::vector_to_center_of_gravity(
                                            q_point, global_particle_properties, src_handle));

                  dealii::Tensor<1, dim, number>         summed_force;
                  dealii::Tensor<1, torque_size, number> summed_torque;

                  for (int i = 0; i < dim; ++i)
                    summed_force[i] = force[i].sum();
                  for (unsigned i = 0; i < torque_size; ++i)
                    summed_torque[i] = torque[i].sum();

                  ObstacleType::accumulate_force(summed_force,
                                                 global_particle_properties,
                                                 src_handle);
                  ObstacleType::accumulate_torque(summed_torque,
                                                  global_particle_properties,
                                                  src_handle);
                }
            }
        }
    }

    // compressible
    dealii::Tensor<1, dim, VectorizedArrayType>
    compute_penalty_force(const dealii::Tensor<1, dim + 2, VectorizedArrayType> &w,
                          const dealii::Tensor<1, dim, VectorizedArrayType>     &obstacle_velocity,
                          const VectorizedArrayType                             &mask) const;

    // incompressible
    dealii::Tensor<1, dim, VectorizedArrayType>
    compute_penalty_force(const dealii::Tensor<1, dim, VectorizedArrayType> &w,
                          const dealii::Tensor<1, dim, VectorizedArrayType> &obstacle_velocity,
                          const VectorizedArrayType                         &mask) const;

    dealii::Tensor<1, torque_size, VectorizedArrayType>
    compute_torque(
      const dealii::Tensor<1, dim, VectorizedArrayType> &force,
      const dealii::Tensor<1, dim, VectorizedArrayType> &distance_to_center_of_gravity) const;
  };

  template <int dim, typename number, int n_components, typename ObstacleType>
  class IncompressibleBrinkmanPenalizationFluidForce
    : public Flow::IncompressibleExternalFluidForce<dim, number, n_components>
  {
    using VectorizedArrayType = dealii::VectorizedArray<number>;

  public:
    /**
     * Constructor that stores relevant data internally and uses the default mask function
     * for the penalty term computation. The default mask function returns 1 for points inside
     * the obstacle volume and 0 outside.
     *
     * @param obstacle_handler Reference to the obstacle handler managing obstacles in the domain.
     * @param brinkman_penalization_data Data required for computing the Brinkman penalization term.
     */
    IncompressibleBrinkmanPenalizationFluidForce(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data,
      number                                          constant_density);

    /**
     * Identifies and stores all relevant particles for the given cell batch, to be used
     * during the quadrature operation.
     *
     * This function is responsible for locating particles that are relevant to the current cell
     * batch and caching them internally. The cached particles are then accessed during the later
     * quadrature computation.
     *
     * @param matrix_free MatrixFree object and corresponding relevant indices.
     * @param cell_batch_id The index of the cell batch to process.
     */
    void
    cell_operation(const MatrixFreeContext<dim, number> &matrix_free,
                   unsigned int                          cell_batch_id) override;

    VectorizedArrayType
    get_damping_coeff_at_q(number, const dealii::Point<dim, VectorizedArrayType> &q_point) override;

    dealii::Tensor<1, n_components, VectorizedArrayType>
    get_rhs_at_q(number, const dealii::Point<dim, VectorizedArrayType> &q_point) override;

  private:
    /// Brinkman penalization data
    const BrinkmanPenalizationData<number> brinkman_penalization_data;

    /// Cached cell data for computing Brinkman penalty term.
    mutable CellObstacleCache<dim, number, ObstacleType> cell_obstacle_cache;

    /// Density of the incompressible flow, assumed to be constant over the complete flow domain.
    const number constant_density;
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
    using VectorizedArrayType    = dealii::VectorizedArray<number>;

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
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data);

    /**
     * Identifies and stores all relevant particles for the given cell batch, to be used
     * during the quadrature operation.
     *
     * This function is responsible for locating particles that are relevant to the current cell
     * batch and caching them internally. The cached particles are then accessed during the later
     * quadrature computation.
     *
     * @param matrix_free MatrixFree object and corresponding relevant indices.
     * @param cell_batch_id The index of the cell batch to process.
     */
    void
    cell_operation(const MatrixFreeContext<dim, number> &matrix_free,
                   unsigned int                          cell_batch_id) override;

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
    quad_operation(number                                         time_step_size,
                   const dealii::Point<dim, VectorizedArrayType> &q_point,
                   const ConservedVariablesType                  &w_q) override;

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
    using VectorizedArrayType    = dealii::VectorizedArray<number>;

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
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data);

    /**
     * Identifies and stores all relevant particles for the given cell batch, to be used
     * during the quadrature operation.
     *
     * This function is responsible for locating particles that are relevant to the current cell
     * batch and caching them internally. The cached particles are then accessed during the later
     * quadrature computation.
     *
     * @param matrix_free MatrixFree object and corresponding relevant indices.
     * @param cell_batch_id The index of the cell batch to process.
     */
    void
    cell_operation(const MatrixFreeContext<dim, number> &matrix_free,
                   unsigned int                          cell_batch_id) override;

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
    quad_operation(number                                         time_step_size,
                   const dealii::Point<dim, VectorizedArrayType> &q_point,
                   const ConservedVariablesType                  &w_q,
                   const ConservedVariablesType                  &delta_w_q) override;

  private:
    /// Brinkman penalization data
    const BrinkmanPenalizationData<number> brinkman_penalization_data;

    /// Cached cell data for computing Brinkman penalty term.
    mutable CellObstacleCache<dim, number, ObstacleType> cell_obstacle_cache;
  };
} // namespace MeltPoolDG
