#include <meltpooldg/heat/laser_heat_source.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  void
  LaserHeatSource<dim>::compute_volumetric_heat_source(VectorType &            heat_source_vector,
                                                       const ScratchData<dim> &scratch_data,
                                                       const unsigned int      temp_dof_idx,
                                                       const double            laser_power,
                                                       const Point<dim> &      laser_position,
                                                       bool                    zero_out) const
  {
    if (zero_out)
      scratch_data.initialize_dof_vector(heat_source_vector, temp_dof_idx);

    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(temp_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
      update_quadrature_points);

    const unsigned int dofs_per_cell =
      scratch_data.get_dof_handler(temp_dof_idx).get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : scratch_data.get_dof_handler(temp_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            heat_source_eval.reinit(cell);

            for (const auto q : heat_source_eval.quadrature_point_indices())
              heat_source_vector[local_dof_indices[q]] =
                local_compute_volumetric_heat_source(heat_source_eval.quadrature_point(q),
                                                     laser_position,
                                                     laser_power);
          }
      }
  }

  template class LaserHeatSource<1>;
  template class LaserHeatSource<2>;
  template class LaserHeatSource<3>;
} // namespace MeltPoolDG::Heat
