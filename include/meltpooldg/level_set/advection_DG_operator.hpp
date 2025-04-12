#pragma once

/* Code adapted from:
https://github.com/kronbichler/advection_miniapp/blob/master/advection_solver_variable.cc*/

#include <deal.II/base/function.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <cmath>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class AdvectionDGOperator
  {
  public:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    using scalar = dealii::VectorizedArray<number>;
    using vector = dealii::Tensor<1, dim, scalar>;

    AdvectionDGOperator(const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
                        VectorType                                      &advection_velocity_in,
                        const unsigned int                               advec_diff_dof_idx_in,
                        const unsigned int                               advec_diff_quad_idx_in,
                        const unsigned int                               velocity_dof_idx_in,
                        const std::shared_ptr<MeltPoolDG::BoundaryConditionManager<dim, number>>
                                                                boundary_conditions_in,
                        std::shared_ptr<dealii::Function<dim>> &advection_field_in,
                        bool const enable_analytical_velocity_update_in);

    /**
     * If an analytical function for the velocity field is provided and an analytical update is
     * enabled, this function sets the velocity according to the anylytical function. Otherwise the
     * provided velocity reference is used.
     * @param time current time
     */
    void
    set_field_functions(const number time) const;


    /**
     * Applies the DG advection operator to the src vector and stores the result in the dst vector.
     * Dirichlet boundary conditions are applied in a different function The dst vector is zeroed
     * out before the operation.
     * The time needs to be explicitly passed for the case, that time dependent boundary conditions
     * or time dependent field functions are used.
     * @param time current time
     * @param dst result of the operator applied to @param src
     * @param scr source vector for the operator
     */
    void
    apply_operator(number                                                 time,
                   VectorType                                            &dst,
                   const VectorType                                      &src,
                   const std::function<void(unsigned int, unsigned int)> &func = {}) const;
    /**
     * Applies the dirichlet contribution of the DG advection operator to the @param src vector and stores
     * the result in the @param dst vector. The dst vector is NOT zeroed out before the operation.
     * @param time current time for time dependent dirichlet conditions.
     */
    void
    apply_dirichlet_boundary_operator(const number      time,
                                      VectorType       &dst,
                                      const VectorType &src) const;

    /**
     *Flag if old velocity needs to be updated
     */
    bool update_field_functions;


  private:
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_;

    VectorType &advection_velocity_;

    const unsigned int advec_diff_dof_idx  = 0;
    const unsigned int advec_diff_quad_idx = 0;
    const unsigned int velocity_dof_idx    = 0;

    const std::shared_ptr<MeltPoolDG::BoundaryConditionManager<dim, number>> boundary_conditions;
    std::shared_ptr<dealii::Function<dim>>                                   advection_field;

    bool const enable_analytical_velocity_update = false;


    void
    local_apply_domain(const dealii::MatrixFree<dim, number>       &data,
                       VectorType                                  &dst,
                       const VectorType                            &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     *  Is needed in apply_dirichlet_boundary_operator. loop needs an operator for the domain, but
     * only the boundary contribution is added to dst.
     */
    void
    local_apply_domain_dummy(
      [[maybe_unused]] const dealii::MatrixFree<dim, number>       &data,
      [[maybe_unused]] VectorType                                  &dst,
      [[maybe_unused]] const VectorType                            &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int> &cell_range) const {};

    void
    local_apply_inner_face(const dealii::MatrixFree<dim, number>       &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;


    /**
     *  Is needed in apply_dirichlet_boundary_operator. loop needs an operator for the inner faces,
     * but only the boundary contribution is added to dst.
     */
    void
    local_apply_inner_face_dummy(
      [[maybe_unused]] const dealii::MatrixFree<dim, number>       &data,
      [[maybe_unused]] VectorType                                  &dst,
      [[maybe_unused]] const VectorType                            &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int> &cell_range) const {};

    void
    local_apply_homogenous_boundary_face(
      const dealii::MatrixFree<dim, number>       &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    local_apply_inhomogenous_boundary_face(
      [[maybe_unused]] const dealii::MatrixFree<dim, number>       &data,
      [[maybe_unused]] VectorType                                  &dst,
      [[maybe_unused]] const VectorType                            &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int> &cell_range) const;
  };



} // namespace MeltPoolDG::LevelSet
