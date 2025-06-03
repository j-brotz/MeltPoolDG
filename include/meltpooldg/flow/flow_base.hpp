#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  class FlowBase
  {
  public:
    virtual void
    init_time_advance() = 0;

    virtual void
    solve() = 0;

    virtual void
    set_initial_condition(const dealii::Function<dim> &) = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity_old() const = 0;

    virtual const dealii::DoFHandler<dim> &
    get_dof_handler_velocity() const = 0;

    virtual const unsigned int &
    get_dof_handler_idx_velocity() const = 0;

    virtual const unsigned int &
    get_dof_handler_idx_hanging_nodes_velocity() const = 0;

    virtual const unsigned int &
    get_quad_idx_velocity() const = 0;

    virtual const unsigned int &
    get_quad_idx_pressure() const = 0;

    virtual const dealii::AffineConstraints<number> &
    get_constraints_velocity() const = 0;

    virtual dealii::AffineConstraints<number> &
    get_constraints_velocity() = 0;

    virtual const dealii::AffineConstraints<number> &
    get_hanging_node_constraints_velocity() const = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_pressure() = 0;

    virtual const dealii::DoFHandler<dim> &
    get_dof_handler_pressure() const = 0;

    virtual const unsigned int &
    get_dof_handler_idx_pressure() const = 0;

    virtual const dealii::AffineConstraints<number> &
    get_constraints_pressure() const = 0;

    virtual dealii::AffineConstraints<number> &
    get_constraints_pressure() = 0;

    virtual const dealii::AffineConstraints<number> &
    get_hanging_node_constraints_pressure() const = 0;

    virtual void
    set_force_rhs(const dealii::LinearAlgebra::distributed::Vector<number> &vec) = 0;

    virtual void
    set_mass_balance_rhs(const dealii::LinearAlgebra::distributed::Vector<number> &vec) = 0;

    virtual void
    set_user_defined_material(std::function<dealii::Tensor<2, dim, dealii::VectorizedArray<number>>(
                                const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> &,
                                const unsigned int,
                                const unsigned int,
                                const bool)> my_user_defined_material) = 0;

    virtual dealii::VectorizedArray<number> &
    get_density(const unsigned int cell, const unsigned int q) = 0;

    virtual const dealii::VectorizedArray<number> &
    get_density(const unsigned int cell, const unsigned int q) const = 0;

    virtual dealii::VectorizedArray<number> &
    get_viscosity(const unsigned int cell, const unsigned int q) = 0;

    virtual const dealii::VectorizedArray<number> &
    get_viscosity(const unsigned int cell, const unsigned int q) const = 0;

    virtual dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) = 0;

    virtual const dealii::VectorizedArray<number> &
    get_damping(const unsigned int cell, const unsigned int q) const = 0;

    virtual void
    attach_vectors_u(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    virtual void
    attach_vectors_p(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    virtual void
    distribute_constraints() = 0;

    virtual void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;

    virtual void
    attach_output_vectors_failed_step(GenericDataOut<dim, number> &data_out) const = 0;
  };
} // namespace MeltPoolDG::Flow
