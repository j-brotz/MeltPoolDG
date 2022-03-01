/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG
{
  namespace Flow
  {
    using namespace dealii;

    template <int dim>
    class FlowBase
    {
    public:
      virtual void
      init_time_advance() = 0;

      virtual void
      solve() = 0;

      virtual void
      set_initial_condition(const Function<dim> &) = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_velocity() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_velocity() = 0;

      virtual const DoFHandler<dim> &
      get_dof_handler_velocity() const = 0;

      virtual const unsigned int &
      get_dof_handler_idx_velocity() const = 0;

      virtual const unsigned int &
      get_dof_handler_idx_hanging_nodes_velocity() const = 0;

      virtual const unsigned int &
      get_quad_idx_velocity() const = 0;

      virtual const unsigned int &
      get_quad_idx_pressure() const = 0;

      virtual const AffineConstraints<double> &
      get_constraints_velocity() const = 0;

      virtual AffineConstraints<double> &
      get_constraints_velocity() = 0;

      virtual const AffineConstraints<double> &
      get_hanging_node_constraints_velocity() const = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_pressure() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_pressure() = 0;

      virtual const DoFHandler<dim> &
      get_dof_handler_pressure() const = 0;

      virtual const unsigned int &
      get_dof_handler_idx_pressure() const = 0;

      virtual const AffineConstraints<double> &
      get_constraints_pressure() const = 0;

      virtual AffineConstraints<double> &
      get_constraints_pressure() = 0;

      virtual const AffineConstraints<double> &
      get_hanging_node_constraints_pressure() const = 0;

      virtual void
      set_force_rhs(const LinearAlgebra::distributed::Vector<double> &vec) = 0;

      virtual void
      set_mass_balance_rhs(const LinearAlgebra::distributed::Vector<double> &vec) = 0;

      virtual void
      set_user_defined_material(std::function<Tensor<2, dim, VectorizedArray<double>>(
                                  const Tensor<2, dim, VectorizedArray<double>> &,
                                  const unsigned int,
                                  const unsigned int,
                                  const bool)> my_user_defined_material) = 0;

      virtual VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q) = 0;

      virtual const VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q) const = 0;

      virtual VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q) = 0;

      virtual const VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q) const = 0;

      virtual VectorizedArray<double> &
      get_damping(const unsigned int cell, const unsigned int q) = 0;

      virtual const VectorizedArray<double> &
      get_damping(const unsigned int cell, const unsigned int q) const = 0;

      virtual void
      attach_vectors_u(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;

      virtual void
      attach_vectors_p(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;

      virtual void
      distribute_constraints() = 0;

      virtual void
      attach_output_vectors(GenericDataOut<dim> &data_out) const = 0;

      virtual void
      attach_output_vectors_failed_step(GenericDataOut<dim> &data_out) const = 0;
    };

  } // namespace Flow
} // namespace MeltPoolDG
