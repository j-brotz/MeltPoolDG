/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, Oktober 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/post_processing/post_processor_base.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG::PostProcessingTools
{
  /**
   * Create a (dim-1,dim) slice through a (dim,dim) triangulation.
   *
   * @note The post processor only supports dim > 1.
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

    double
    get_divergence() const
    {
      return diver;
    }

    void
    process(const unsigned int /*n_time_step*/) override
    {
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
      generic_data_out->get_vector(request_variable).zero_out_ghost_values();
    }

    void
    reinit(const GenericDataOut<dim> &generic_data_out_in) override
    {
      generic_data_out = &generic_data_out_in;
    }
  };
} // namespace MeltPoolDG::PostProcessingTools
