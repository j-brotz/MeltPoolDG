#pragma once

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::PostProcessingTools
{
  template <int dim, typename number>
  class PostProcessorBase
  {
  public:
    virtual void
    process(const unsigned int n_time_step) = 0;

    virtual void
    reinit(const GenericDataOut<dim, number> &generic_data_out_in) = 0;

    virtual ~PostProcessorBase() = default;
  };
} // namespace MeltPoolDG::PostProcessingTools
