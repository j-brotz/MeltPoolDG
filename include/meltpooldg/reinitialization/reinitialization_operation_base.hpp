/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    template <int dim>
    class ReinitializationOperationBase
    {
    public:
      virtual void
      solve() = 0;

      virtual void
      reinit() = 0;

      virtual void
      update_dof_idx(const unsigned int &reinit_dof_idx_in) = 0;

      virtual void
      set_initial_condition(
        const LinearAlgebra::distributed::Vector<double> &solution_level_set_in) = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_level_set() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_level_set() = 0;

      virtual double
      get_max_change_level_set() const = 0;

      virtual const LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() const = 0;

      virtual LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() = 0;

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;

      virtual void
      attach_output_vectors(GenericDataOut<dim> &data_out) const = 0;
    };

  } // namespace LevelSet
} // namespace MeltPoolDG
