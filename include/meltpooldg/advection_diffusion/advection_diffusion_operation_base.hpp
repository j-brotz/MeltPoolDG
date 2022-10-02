/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    using namespace dealii;

    template <int dim>
    class AdvectionDiffusionOperationBase
    {
    public:
      AdvectionDiffusionData<double> advec_diff_data;

      virtual void
      init_time_advance() = 0;

      virtual void
      solve(const LinearAlgebra::distributed::Vector<double> &velocity) = 0;

      virtual void
      reinit()
      {
        AssertThrow(false, ExcNotImplemented());
      }

      virtual void
      set_initial_condition(const Function<dim> & /*initial_field_function*/,
                            const LinearAlgebra::distributed::Vector<double> & /*initial_velocity*/)
      {
        AssertThrow(false, ExcNotImplemented());
      }

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_advected_field() = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_user_rhs() = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_user_rhs() const = 0;

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;

      virtual void
      attach_output_vectors(GenericDataOut<dim> &data_out) const = 0;
    };

  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
