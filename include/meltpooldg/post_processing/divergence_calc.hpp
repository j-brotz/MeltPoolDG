/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, October 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/post_processor_base.hpp>

namespace MeltPoolDG::PostProcessingTools
{
  /**
   * Compute the integral of the divergence over the domain Ω of
   * a vector, e.g. for the velocity u
   *
   *   ∫ ∇·u dΩ
   *  Ω
   */
  template <int dim>
  class DivergenceCalculator : public PostProcessorBase<dim>
  {
  private:
    const GenericDataOut<dim> *generic_data_out = nullptr;
    const std::string          request_variable;
    double                     diver;

  public:
    DivergenceCalculator(const GenericDataOut<dim> &generic_data_out,
                         const std::string          request_variable)
      : generic_data_out(&generic_data_out)
      , request_variable(request_variable)
    {}

    /**
     * Main result of this postprocessor.
     */
    double
    get_divergence() const
    {
      return diver;
    }

    /**
     * Process current vector requested by @p request_variable within
     * @p generic_data_out.
     *
     * TODO: We could do this calculation matrix-free, if GenericDataOut has a matrix-free
     * object.
     */
    void
    process(const unsigned int /*n_time_step*/) override
    {
      const bool update_ghosts =
        !generic_data_out->get_vector(request_variable).has_ghost_elements();

      if (update_ghosts)
        generic_data_out->get_vector(request_variable).update_ghost_values();

      QGauss<dim> quad(
        generic_data_out->get_dof_handler(request_variable).get_fe().tensor_degree() + 1);

      FEValues<dim> vel_values(generic_data_out->get_mapping(),
                               generic_data_out->get_dof_handler(request_variable).get_fe(),
                               quad,
                               update_gradients | update_JxW_values);

      const FEValuesExtractors::Vector velocities(0);

      std::vector<double> div(vel_values.get_quadrature().size());

      double diver_local = 0;
      for (const auto &cell :
           generic_data_out->get_dof_handler(request_variable).active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              vel_values.reinit(cell);
              vel_values[velocities].get_function_divergences(
                generic_data_out->get_vector(request_variable), div);

              for (const unsigned int q : vel_values.quadrature_point_indices())
                diver_local += div[q] * vel_values.JxW(q);
            }
        }

      diver =
        Utilities::MPI::sum(diver_local,
                            generic_data_out->get_vector(request_variable).get_mpi_communicator());
      if (update_ghosts)
        generic_data_out->get_vector(request_variable).zero_out_ghost_values();
    }

    /**
     * Reinit with new GenericDataOut object.
     */
    void
    reinit(const GenericDataOut<dim> &generic_data_out_in) override
    {
      generic_data_out = &generic_data_out_in;
    }
  };
} // namespace MeltPoolDG::PostProcessingTools
