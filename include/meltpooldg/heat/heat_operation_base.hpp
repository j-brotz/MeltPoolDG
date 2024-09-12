#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <vector>

namespace MeltPoolDG::Heat
{
  template <int dim>
  class HeatOperationBase
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

    virtual void
    reinit() = 0;

    virtual void
    set_initial_condition(const dealii::Function<dim> &initial_temperature,
                          const double                 start_time) = 0;

    virtual void
    distribute_constraints() = 0;

    virtual void
    solve(const bool do_finish_time_step = true) = 0;

    /**
     * register vectors for adaptive mesh refinement solution transfer
     */
    virtual void
    attach_vectors(std::vector<VectorType *> &vectors) = 0;

    /**
     * attach vectors for output
     */
    virtual void
    attach_output_vectors(GenericDataOut<dim> &data_out) const = 0;

    virtual void
    attach_output_vectors_failed_step(GenericDataOut<dim> &data_out) const = 0;

    /*
     * getters
     */

    virtual const VectorType &
    get_temperature() const = 0;

    virtual VectorType &
    get_temperature() = 0;

    virtual VectorType &
    get_heat_source() = 0;

    virtual const VectorType &
    get_heat_source() const = 0;

    virtual const VectorType &
    get_user_rhs() const = 0;

    virtual VectorType &
    get_user_rhs() = 0;
  };
} // namespace MeltPoolDG::Heat