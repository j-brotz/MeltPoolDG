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
    const unsigned int          temp_dof_idx,
    const unsigned int          n_iterations)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
    , level_set_as_heaviside(level_set_as_heaviside)
    , distance(distance)
    , normal_vector(normal_vector)
    , ls_dof_idx(ls_dof_idx)
    , temp_dof_idx(temp_dof_idx)
    , n_iterations(n_iterations)
    , tolerance_normal_vector(
        UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                              scratch_data.get_mapping()))
  {}


  template <int dim>
  void
  EvaporationMassFluxOperatorInterfaceValue<dim>::compute_evaporative_mass_flux(
    VectorType &      evaporative_mass_flux,
    const VectorType &temperature) const
  {
    //@todo: add assert that the algorithm only works if temp_degree = ls_degree
    //
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

    /*
     * fill initial evaluation points with node coordinates
     */
    for (const auto &cell : scratch_data.get_dof_handler(ls_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            ls_values.reinit(cell);

            cell->get_dof_indices(temp_local_dof_indices);

            cell->get_dof_values(level_set_as_heaviside, hs_temp);

            for (const auto q : ls_values.quadrature_point_indices())
              {
                // consider only points in narrow band
                if (hs_temp[q] < 1.0 && hs_temp[q] > 0.0)
                  {
                    evaluation_points.push_back(ls_values.quadrature_point(q));
                    dof_indices.push_back(temp_local_dof_indices[q]);
                  }
              }
          }
      }

    /*
     * update evaluation points n_iterations times by iteratively projecting
     * the point to the interface.
     */
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(
      1e-6 /*tolerance*/, true /*unique mapping*/);

    for (unsigned int j = 0; j < n_iterations; ++j)
      {
        remote_point_evaluation.reinit(evaluation_points,
                                       scratch_data.get_triangulation(),
                                       scratch_data.get_mapping());

        const auto evaluation_values_distance =
          VectorTools::point_values<1>(remote_point_evaluation,
                                       scratch_data.get_dof_handler(ls_dof_idx),
                                       distance);

        std::array<std::vector<double>, dim> evaluation_values_normal;

        for (unsigned int comp = 0; comp < dim; ++comp)
          evaluation_values_normal[comp] =
            VectorTools::point_values<1>(remote_point_evaluation,
                                         scratch_data.get_dof_handler(ls_dof_idx),
                                         normal_vector.block(comp));
        for (unsigned int counter = 0; counter < evaluation_points.size(); ++counter)
          {
            /*
             * compute unit normal vector
             */
            Point<dim> unit_normal;
            for (unsigned int comp = 0; comp < dim; ++comp)
              unit_normal[comp] = evaluation_values_normal[comp][counter];

            const auto n_norm = unit_normal.norm();
            unit_normal = (n_norm > tolerance_normal_vector) ? unit_normal / n_norm : Point<dim>();

            // compute corresponding point at level set == 0.0
            for (unsigned int d = 0; d < dim; ++d)
              {
                evaluation_points[counter][d] -=
                  evaluation_values_distance[counter] * unit_normal[d];

                // check if point is outside domain and if so then project it back to
                // the domain
                if (evaluation_points[counter][d] < boundary_points.first[d])
                  evaluation_points[counter][d] = boundary_points.first[d];
                else if (evaluation_points[counter][d] > boundary_points.second[d])
                  evaluation_points[counter][d] = boundary_points.second[d];
              }
          }
      }


    if (false)
      {
        /*
         * debug
         */
        const auto global_points_normal_to_interface_all =
          Utilities::MPI::reduce<std::vector<Point<dim>>>(
            evaluation_points, scratch_data.get_mpi_comm(), [](const auto &a, const auto &b) {
              auto result = a;
              result.insert(result.end(), b.begin(), b.end());
              return result;
            });

        std::ofstream myfile;
        myfile.open("generated_points.dat");

        for (const auto &p : global_points_normal_to_interface_all)
          {
            for (unsigned int d = 0; d < dim; ++d)
              myfile << p[d] << " ";
            myfile << std::endl;
          }

        myfile.close();
      }

    /*
     * get temperature values at evaluation points
     */
    remote_point_evaluation.reinit(evaluation_points,
                                   scratch_data.get_triangulation(),
                                   scratch_data.get_mapping());

    temperature.update_ghost_values();

    const auto temperature_evaluation_values =
      dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                           scratch_data.get_dof_handler(temp_dof_idx),
                                           temperature);
    temperature.zero_out_ghost_values();

    Assert(temperature_evaluation_values.size() == evaluation_points.size(),
           ExcMessage("The size of vectors must match."));

    /*
     * compute evaporative mass flux from the temperature value at the interface
     */
    evaporative_mass_flux.zero_out_ghost_values();
    evaporative_mass_flux = 0.0;

    for (unsigned int i = 0; i < evaluation_points.size(); ++i)
      {
        evaporative_mass_flux[dof_indices[i]] =
          evaporation_model.local_compute_evaporative_mass_flux(temperature_evaluation_values[i]);
      }

    evaporative_mass_flux.update_ghost_values();
  }

  template class EvaporationMassFluxOperatorInterfaceValue<1>;
  template class EvaporationMassFluxOperatorInterfaceValue<2>;
  template class EvaporationMassFluxOperatorInterfaceValue<3>;
} // namespace MeltPoolDG::Evaporation
