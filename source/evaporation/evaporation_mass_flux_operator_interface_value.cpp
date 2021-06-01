#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_interface_value.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim>
  EvaporationMassFluxOperatorInterfaceValue<dim>::EvaporationMassFluxOperatorInterfaceValue(
    const ScratchData<dim> &    scratch_data,
    const EvaporationModelBase &evaporation_model,
    const VectorType &          level_set_as_heaviside,
    const VectorType &          distance,
    const BlockVectorType &     normal_vector,
    const unsigned int          ls_dof_idx,
    const unsigned int          temp_dof_idx)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
    , level_set_as_heaviside(level_set_as_heaviside)
    , distance(distance)
    , normal_vector(normal_vector)
    , ls_dof_idx(ls_dof_idx)
    , temp_dof_idx(temp_dof_idx)
  {}


  template <int dim>
  void
  EvaporationMassFluxOperatorInterfaceValue<dim>::compute_evaporative_mass_flux(
    VectorType &      evaporative_mass_flux,
    const VectorType &temperature) const
  {
    /*
     * collect evaluation points
     */
    distance.update_ghost_values();
    normal_vector.update_ghost_values();

    FEValues<dim, dim> ls_values(
      scratch_data.get_mapping(),
      scratch_data.get_fe(ls_dof_idx),
      Quadrature<dim>(scratch_data.get_fe(ls_dof_idx).base_element(0).get_unit_support_points()),
      update_quadrature_points);

    std::vector<Point<dim>>              evaluation_points;
    std::vector<types::global_dof_index> dof_indices;

    const unsigned int n_q_points = ls_values.get_quadrature().size();

    // temporary values at cell nodes nodes
    Vector<double>                       hs_temp(n_q_points);
    Vector<double>                       distance_temp(n_q_points);
    std::vector<Vector<double>>          normal_temp(dim, Vector<double>(n_q_points));
    std::vector<Point<dim>>              normalized_normal_temp(n_q_points, Point<dim>());
    std::vector<types::global_dof_index> temp_local_dof_indices(n_q_points);

    const auto bounding_box    = GridTools::compute_bounding_box(scratch_data.get_triangulation());
    const auto boundary_points = bounding_box.get_boundary_points();

    for (const auto &cell : scratch_data.get_dof_handler(ls_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            ls_values.reinit(cell);

            cell->get_dof_indices(temp_local_dof_indices);

            cell->get_dof_values(level_set_as_heaviside, hs_temp);
            cell->get_dof_values(distance, distance_temp);
            for (unsigned int d = 0; d < dim; ++d)
              cell->get_dof_values(normal_vector.block(d), normal_temp[d]);

            // normalize normal vector values
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int q = 0; q < n_q_points; ++q)
                normalized_normal_temp[q][d] = normal_temp[d][q];

            for (auto &n : normalized_normal_temp)
              n = n / n.norm();

            for (const auto q : ls_values.quadrature_point_indices())
              {
                if (hs_temp[q] < 1.0 && hs_temp[q] > 0.0)
                  {
                    // compute corresponding point at level set == 0.0

                    Point<dim> evaluation_point_temp = Point<dim>();
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        evaluation_point_temp[d] = ls_values.quadrature_point(q)[d] -
                                                   distance_temp[q] * normalized_normal_temp[q][d];

                        // check if point is outside domain and if so then project it back to the
                        // domain
                        if (evaluation_point_temp[d] < boundary_points.first[d])
                          evaluation_point_temp[d] = boundary_points.first[d];
                        else if (evaluation_point_temp[d] > boundary_points.second[d])
                          evaluation_point_temp[d] = boundary_points.second[d];
                      }

                    evaluation_points.push_back(evaluation_point_temp);
                    dof_indices.push_back(temp_local_dof_indices[q]);
                  }
              }
          }
      }

    /*
     * get temperature values at evaluation points
     */
    Utilities::MPI::RemotePointEvaluation<dim, dim> cache;

    temperature.update_ghost_values();
    const auto temperature_evaluation_values =
      dealii::VectorTools::point_values<1>(scratch_data.get_mapping(),
                                           scratch_data.get_dof_handler(temp_dof_idx),
                                           temperature,
                                           evaluation_points,
                                           cache,
                                           dealii::VectorTools::EvaluationFlags::max);
    temperature.zero_out_ghost_values();

    Assert(temperature_evaluation_values.size() == evaluation_points.size(),
           ExcMessage("The size of vectors must match."));

    /*
     * compute evaporative mass flux
     */
    evaporative_mass_flux.zero_out_ghost_values();
    evaporative_mass_flux = 0.0;

    for (unsigned int i = 0; i < evaluation_points.size(); i++)
      {
        evaporative_mass_flux[temp_local_dof_indices[i]] =
          evaporation_model.local_compute_evaporative_mass_flux(
            temperature[temp_local_dof_indices[i]]);
      }

    evaporative_mass_flux.update_ghost_values();
  }

  template class EvaporationMassFluxOperatorInterfaceValue<1>;
  template class EvaporationMassFluxOperatorInterfaceValue<2>;
  template class EvaporationMassFluxOperatorInterfaceValue<3>;
} // namespace MeltPoolDG::Evaporation
