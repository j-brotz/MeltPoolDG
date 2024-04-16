#pragma once

/* Code adapted from:
https://github.com/kronbichler/advection_miniapp/blob/master/advection_solver_variable.cc*/

#include <deal.II/base/function.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <cmath>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename Number = double>
  class AdvectionDGOperator
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

    using vector = Tensor<1, dim, VectorizedArray<Number>>;
    using scalar = VectorizedArray<Number>;

    AdvectionDGOperator(
      const MeltPoolDG::ScratchData<dim>                   &scratch_data_in,
      VectorType                                           &advection_velocity_in,
      const unsigned int                                    advec_diff_dof_idx_in,
      const unsigned int                                    advec_diff_quad_idx_in,
      const unsigned int                                    velocity_dof_idx_in,
      std::shared_ptr<MeltPoolDG::BoundaryConditions<dim>> &boundary_conditions_in,
      std::shared_ptr<dealii::Function<dim>>               &advection_field_in,
      bool const                                            enable_analytical_velocity_update_in);

    /**
     * Allocates memory for the vectors based on the degrees of freedom of the DoFHandler. In this
     * case the function is empty.
     */
    void
    reinit(){};

    /**
     * If an analytical function for the velocity field is provided and an analytical update is
     * enabled, this function sets the velocity according to the anylytical function. Otherwise the
     * provided velocity reference is used.
     * @param time current time
     */
    void
    set_velocity_operator(const Number time) const;


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
    apply_operator(const Number                                      time,
                   LinearAlgebra::distributed::Vector<Number>       &dst,
                   const LinearAlgebra::distributed::Vector<Number> &src) const;
    /**
     * Applies the dirichlet contribution of the DG advection operator to the @param src vector and stores
     * the result in the @param dst vector. The dst vector is NOT zeroed out before the operation.
     * @param time current time for time dependent dirichlet conditions.
     */
    void
    apply_dirichlet_boundary_operator(const Number                                      time,
                                      LinearAlgebra::distributed::Vector<Number>       &dst,
                                      const LinearAlgebra::distributed::Vector<Number> &src) const;

    /**
     *Flag if old velocity needs to be updated
     */
    bool update_velocity_;


  private:
    const MeltPoolDG::ScratchData<dim> &scratch_data_;

    VectorType &advection_velocity_;

    const unsigned int advec_diff_dof_idx  = 0;
    const unsigned int advec_diff_quad_idx = 0;
    const unsigned int velocity_dof_idx    = 0;

    std::shared_ptr<MeltPoolDG::BoundaryConditions<dim>> boundary_conditions;
    std::shared_ptr<dealii::Function<dim>>               advection_field;

    bool const enable_analytical_velocity_update = false;


    void
    local_apply_domain(const MatrixFree<dim, Number>                    &data,
                       LinearAlgebra::distributed::Vector<Number>       &dst,
                       const LinearAlgebra::distributed::Vector<Number> &src,
                       const std::pair<unsigned int, unsigned int>      &cell_range) const;

    /**
     *  Is needed in apply_dirichlet_boundary_operator. loop needs an operator for the domain, but
     * only the boundary contribution is added to dst.
     */
    void
    local_apply_domain_dummy(
      [[maybe_unused]] const MatrixFree<dim, Number>                    &data,
      [[maybe_unused]] LinearAlgebra::distributed::Vector<Number>       &dst,
      [[maybe_unused]] const LinearAlgebra::distributed::Vector<Number> &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int>      &cell_range) const {};

    void
    local_apply_inner_face(const MatrixFree<dim, Number>                    &data,
                           LinearAlgebra::distributed::Vector<Number>       &dst,
                           const LinearAlgebra::distributed::Vector<Number> &src,
                           const std::pair<unsigned int, unsigned int>      &cell_range) const;


    /**
     *  Is needed in apply_dirichlet_boundary_operator. loop needs an operator for the inner faces,
     * but only the boundary contribution is added to dst.
     */
    void
    local_apply_inner_face_dummy(
      [[maybe_unused]] const MatrixFree<dim, Number>                    &data,
      [[maybe_unused]] LinearAlgebra::distributed::Vector<Number>       &dst,
      [[maybe_unused]] const LinearAlgebra::distributed::Vector<Number> &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int>      &cell_range) const {};

    void
    local_apply_homogenous_boundary_face(
      const MatrixFree<dim, Number>                    &data,
      LinearAlgebra::distributed::Vector<Number>       &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int>      &cell_range) const;

    void
    local_apply_inhomogenous_boundary_face(
      [[maybe_unused]] const MatrixFree<dim, Number>                    &data,
      [[maybe_unused]] LinearAlgebra::distributed::Vector<Number>       &dst,
      [[maybe_unused]] const LinearAlgebra::distributed::Vector<Number> &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int>      &cell_range) const;
  };



} // namespace MeltPoolDG::LevelSet
