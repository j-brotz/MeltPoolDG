#pragma once

#include <deal.II/base/point.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_utilities.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

#include <memory>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * Volumetric laser heat source model.
   */
  template <int dim>
  class LaserHeatSourceVolumetric
  {
    using VectorType = LinearAlgebra::distributed::Vector<double>;

  public:
    LaserHeatSourceVolumetric(
      std::shared_ptr<const LaserIntensityProfileBase<dim, double>> intensity_profile_in);

    /**
     * Compute a DoF vector of the volumetric heat source.
     */
    void
    compute_volumetric_heat_source(VectorType             &heat_source_vector,
                                   const ScratchData<dim> &scratch_data,
                                   const unsigned int      temp_dof_idx,
                                   const bool              zero_out = true) const;

  private:
    std::shared_ptr<const LaserIntensityProfileBase<dim, double>> intensity_profile;
  };
} // namespace MeltPoolDG::Heat
