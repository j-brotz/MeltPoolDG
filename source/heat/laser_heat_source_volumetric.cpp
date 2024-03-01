#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_values.h>

#include <meltpooldg/heat/laser_heat_source_volumetric.hpp>

#include <vector>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  LaserHeatSourceVolumetric<dim>::LaserHeatSourceVolumetric(
    const std::shared_ptr<const Function<dim, double>> intensity_profile_in)
    : intensity_profile(intensity_profile_in)
  {}

  template <int dim>
  void
  LaserHeatSourceVolumetric<dim>::compute_volumetric_heat_source(
    VectorType             &heat_source_vector,
    const ScratchData<dim> &scratch_data,
    const unsigned int      heat_source_dof_idx,
    const bool              zero_out) const
  {
    if (zero_out)
      heat_source_vector = 0;

    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(heat_source_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(heat_source_dof_idx).get_fe().get_unit_support_points()),
      update_quadrature_points);

    const unsigned int dofs_per_cell =
      scratch_data.get_dof_handler(heat_source_dof_idx).get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell :
         scratch_data.get_dof_handler(heat_source_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            heat_source_eval.reinit(cell);

            for (const auto q : heat_source_eval.quadrature_point_indices())
              heat_source_vector[local_dof_indices[q]] =
                intensity_profile->value(heat_source_eval.quadrature_point(q));
          }
      }

    scratch_data.get_constraint(heat_source_dof_idx).distribute(heat_source_vector);
  }

  template class LaserHeatSourceVolumetric<1>;
  template class LaserHeatSourceVolumetric<2>;
  template class LaserHeatSourceVolumetric<3>;
} // namespace MeltPoolDG::Heat
