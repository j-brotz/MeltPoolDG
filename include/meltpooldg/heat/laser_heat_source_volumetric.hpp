#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>

#include <memory>

namespace MeltPoolDG::Heat
{
  /**
   * Volumetric laser heat source model.
   */
  template <int dim, typename number>
  class LaserHeatSourceVolumetric
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    LaserHeatSourceVolumetric(
      const std::shared_ptr<const dealii::Function<dim, number>> intensity_profile_in);

    /**
     * Compute a DoF vector of the volumetric heat source.
     */
    void
    compute_volumetric_heat_source(VectorType                          &heat_source_vector,
                                   const ScratchData<dim, dim, number> &scratch_data,
                                   const unsigned int                   heat_dof_idx,
                                   const bool                           zero_out = true) const;

  private:
    const std::shared_ptr<const dealii::Function<dim, number>> intensity_profile;
  };
} // namespace MeltPoolDG::Heat
