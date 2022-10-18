/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG
{
  namespace PostProcessingTools
  {
    using namespace dealii;

    template <int dim>
    class PostProcessorBase
    {
    public:
      virtual void
      process(const unsigned int n_time_step) = 0;

      virtual void
      reinit(const GenericDataOut<dim> &generic_data_out_in) = 0;
    };
  } // namespace PostProcessingTools
} // namespace MeltPoolDG
