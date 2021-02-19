/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::HeatEquation
{
  using namespace dealii;

  template <int dim>
  class HeatOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const std::shared_ptr<const ScratchData<dim>> &scratch_data;
    /**
     *  time
     */
    const double time;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int temp_dof_idx;
    const unsigned int temp_quad_idx;
    /*
     *    This are the primary solution variables of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType temperature;

  public:
    HeatOperation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                  const double                                   time_in,
                  const unsigned int                             temp_dof_idx_in,
                  const unsigned int                             temp_quad_idx_in)
      : scratch_data(scratch_data_in)
      , time(time_in)
      , temp_dof_idx(temp_dof_idx_in)
      , temp_quad_idx(temp_quad_idx_in)
    {}

    void
    reinit()
    {
      scratch_data->initialize_dof_vector(temperature, temp_dof_idx);
    }

    void
    solve()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      temperature.update_ghost_values();
      vectors.push_back(&temperature);
    }

    void
    distribute_constraints()
    {
      scratch_data->get_constraint(temp_dof_idx).distribute(temperature);
    }

    void
    attach_output_vectors(DataOut<dim> &data_out) const
    {
      /**
       *  temperature
       */
      MeltPoolDG::VectorTools::update_ghost_values(temperature);
      data_out.add_data_vector(scratch_data->get_dof_handler(temp_dof_idx),
                               temperature,
                               "temperature");
    }

    const VectorType &
    get_temperature() const
    {
      return temperature;
    }

    VectorType &
    get_temperature()
    {
      return temperature;
    }
  };
} // namespace MeltPoolDG::HeatEquation
