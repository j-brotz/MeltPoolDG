#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>

#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class AdvectionDiffusionOperationBase
  {
  protected:
    // determine whether solution vectors are prepared for time advance
    bool ready_for_time_advance = false;

  public:
    AdvectionDiffusionData<number> advec_diff_data;

    virtual void
    init_time_advance() = 0;

    virtual void
    finish_time_advance()
    {
      ready_for_time_advance = false;
    }

    virtual void
    solve(const bool do_finish_time_step = true) = 0;

    virtual void
    reinit() = 0;

    virtual void
    set_initial_condition(const dealii::Function<dim> & /*initial_field_function*/) = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() const = 0;

    virtual void
    attach_vectors(std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    virtual void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;
  };
} // namespace MeltPoolDG::LevelSet
