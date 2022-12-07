// Test delta-approximation classes.

#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>

#include <iostream>

using namespace MeltPoolDG;
using namespace dealii;

template <typename Number>
void
test(const DeltaApproximationPhaseWeightedData<Number> &data, const unsigned int n_intervals = 100)
{
  const auto delta_approximation = create_phase_weighted_delta_approximation(data);

  std::cout << data.type << ": " << std::endl;
  for (unsigned int i = 0; i <= n_intervals; ++i)
    {
      const auto f = static_cast<Number>(i) / static_cast<Number>(n_intervals);
      std::cout << delta_approximation->compute_weight(f) << " ";
    }
  std::cout << std::endl << std::endl;
}

int
main()
{
  {
    DeltaApproximationPhaseWeightedData data;
    data.type = DiracDeltaFunctionApproximationType::heaviside_phase_weighted;
    test(data);
  }

  {
    DeltaApproximationPhaseWeightedData data;
    data.type = DiracDeltaFunctionApproximationType::heavy_phase_only;
    test(data);
  }

  {
    DeltaApproximationPhaseWeightedData data;
    data.type = DiracDeltaFunctionApproximationType::quad_heaviside_phase_weighted;
    test(data);
  }

  {
    DeltaApproximationPhaseWeightedData data;
    data.type = DiracDeltaFunctionApproximationType::heaviside_times_heaviside_phase_weighted;
    test(data);
  }

  {
    DeltaApproximationPhaseWeightedData data;
    data.type               = DiracDeltaFunctionApproximationType::reciprocal_phase_weighted;
    data.gas_phase_weight   = 0.1;
    data.heavy_phase_weight = 0.2;
    test(data);
  }

  {
    DeltaApproximationPhaseWeightedData data;
    data.gas_phase_weight     = 0.1;
    data.heavy_phase_weight   = 0.2;
    data.gas_phase_weight_2   = 0.1;
    data.heavy_phase_weight_2 = 0.2;

    data.type = DiracDeltaFunctionApproximationType::reciprocal_times_heaviside_phase_weighted;
    test(data);
  }
}
