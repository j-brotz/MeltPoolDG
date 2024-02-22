// Test delta-approximation classes.

#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
#include <meltpooldg/material/material_data.hpp>

#include <iostream>

using namespace MeltPoolDG::LevelSet;
using namespace dealii;

template <typename Number>
void
test(const DeltaApproximationPhaseWeightedData<Number> data, const unsigned int n_intervals = 100)
{
  const std::unique_ptr<DeltaApproximationBase<Number>> delta_approximation =
    create_phase_weighted_delta_approximation(data);

  // norm of indicator gradient
  if (!delta_approximation)
    return;

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
  DeltaApproximationPhaseWeightedData data;
  data.gas_phase_weight     = 0.1;
  data.heavy_phase_weight   = 0.2;
  data.gas_phase_weight_2   = 0.5;
  data.heavy_phase_weight_2 = 1.0;

  for (const auto type : DiracDeltaFunctionApproximationType::_values())
    {
      data.type = type;
      test(data);
    }
}
