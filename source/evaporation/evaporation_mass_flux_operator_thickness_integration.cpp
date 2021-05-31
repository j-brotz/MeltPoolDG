#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_thickness_integration.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim>
  EvaporationMassFluxOperatorThicknessIntegration<dim>::
    EvaporationMassFluxOperatorThicknessIntegration(const ScratchData<dim> &    scratch_data,
                                                    const EvaporationModelBase &evaporation_model,
                                                    const VectorType &     level_set_as_heaviside,
                                                    const BlockVectorType &normal_vector,
                                                    const unsigned int     ls_dof_idx,
                                                    const unsigned int     normal_dof_idx,
                                                    const unsigned int     temp_dof_idx)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
    , level_set_as_heaviside(level_set_as_heaviside)
    , normal_vector(normal_vector)
    , ls_dof_idx(ls_dof_idx)
    , normal_dof_idx(normal_dof_idx)
    , temp_dof_idx(temp_dof_idx)
    , epsilon(scratch_data.get_min_cell_size()) //@todo: how to determine epsilon???
  {}

  template <int dim>
  void
  EvaporationMassFluxOperatorThicknessIntegration<dim>::compute_evaporative_mass_flux(
    VectorType &      evaporative_mass_flux,
    const VectorType &temperature) const
  {
    Assert(dim > 1, ExcMessage("The requested operation can only be performed for dim>1."))

      if constexpr (dim > 1)
    {
      /*
       * generate point cloud normal to interface
       */

      std::vector<Point<dim>>   global_points_normal_to_interface;
      std::vector<unsigned int> global_points_normal_to_interface_pointer;

      //@todo
      FESystem<dim> fe_dim(FE_Q<dim>(scratch_data.get_degree(normal_dof_idx)), dim);

      UtilityFunctions::generate_points_along_normal<dim>(
        global_points_normal_to_interface,
        global_points_normal_to_interface_pointer,
        scratch_data.get_dof_handler(ls_dof_idx),
        fe_dim,
        scratch_data.get_mapping(),
        level_set_as_heaviside,
        normal_vector,
        5 * epsilon, // todo how to determine epsilon?
        10,
        true,
        /* bidirectional */ // todo: make parameter
        0.0,
        /* contour_value */ // todo: make parameter
        1);
      /* n_subdivisions_MCA */ // todo: make parameter

      /*
       * evaluate result at points normal to interface
       */
      Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(
        1e-10 /*tolerance*/, true /*unique mapping*/);

      remote_point_evaluation.reinit(global_points_normal_to_interface,
                                     scratch_data.get_triangulation(),
                                     scratch_data.get_mapping());

      const auto temperature_evaluation_values =
        dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                             scratch_data.get_dof_handler(temp_dof_idx),
                                             temperature);

      const std::vector<Tensor<1, dim, double>> level_set_gradient_values =
        dealii::VectorTools::point_gradients<1>(remote_point_evaluation,
                                                scratch_data.get_dof_handler(ls_dof_idx),
                                                level_set_as_heaviside);
      /*
       * evaluate line integral
       *
       * @todo: how to determine start and end point for a given level set
       * value better than 5*epsilon?
       *
       */
      std::vector<double> mass_flux_val;
      mass_flux_val.resize(global_points_normal_to_interface_pointer.back());

      // loop over points located at the interface and determined by means of MCA
      for (unsigned int i = 0; i < global_points_normal_to_interface_pointer.size() - 1; ++i)
        {
          const auto start = global_points_normal_to_interface_pointer[i];
          const auto end   = global_points_normal_to_interface_pointer[i + 1];
          const auto size  = end - start;

          std::vector<double> func_vals(size);

          // loop over all points along normal at one MC point
          for (unsigned int l = 0; l < size; ++l)
            {
              func_vals[l] = evaporation_model.local_compute_evaporative_mass_flux(
                               temperature_evaluation_values[start + l]) *
                             level_set_gradient_values[start + l].norm();
            }

          double mass_flux_average = UtilityFunctions::integrate_over_line<dim>(
            func_vals,
            std::vector(global_points_normal_to_interface.begin() + start,
                        global_points_normal_to_interface.begin() + end));

          /*
           * set for each point along the interface the value equal to the one determined
           * by the line integral
           */
          for (unsigned int l = 0; l < size; ++l)
            mass_flux_val[start + l] = mass_flux_average;
        }

      /*
       * Broadcast values determined along the normal direction to nodal points by means
       * of a simple averaging procedure. Subsequently, a DoF-vector is filled.
       *
       * @note We also tried the extrapolation via a radial basis function, however it was
       * too sensitive with respect to the inherent distance parameter.
       */
      VectorType vector_multiplicity;
      vector_multiplicity.reinit(temperature);

      const auto broadcast_function = [&](const auto &values, const auto &cell_data) {
        // loop over all cells where points along normal are interior
        for (unsigned int i = 0; i < cell_data.cells.size(); ++i)
          {
            typename DoFHandler<dim>::active_cell_iterator cell = {
              &remote_point_evaluation.get_triangulation(),
              cell_data.cells[i].first,
              cell_data.cells[i].second,
              &scratch_data.get_dof_handler(ls_dof_idx)};

            // values at the points along normal in reference coordinates
            const ArrayView<const double> temp_vals(values.data() +
                                                      cell_data.reference_point_ptrs[i],
                                                    cell_data.reference_point_ptrs[i + 1] -
                                                      cell_data.reference_point_ptrs[i]);

            // extract dof indices of cell
            std::vector<types::global_dof_index> local_dof_indices(
              cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(local_dof_indices);

            std::vector<double> values_at_points;

            {
              // loop over arbitrary points of the cell
              for (unsigned int q = 0; q < temp_vals.size(); ++q)
                values_at_points.emplace_back(temp_vals[q]);
            }

            /*
             * Compute mean value from the values of points within the cell and set the values at
             * the cell nodes equal to the latter.
             */
            Vector<double> nodal_values(cell->get_fe().n_dofs_per_cell());

            double average = 0;
            for (const auto &v : values_at_points)
              average += v;
            average /= values_at_points.size();

            for (auto &n : nodal_values)
              n = average;

            cell->set_dof_values(nodal_values, evaporative_mass_flux);

            // count the number of written values (multiplicity)
            for (auto &val : nodal_values)
              val = 1.0;

            cell->set_dof_values(nodal_values, vector_multiplicity);
          }
      };

      // fill dof vector
      std::vector<double> buffer;

      remote_point_evaluation.template process_and_evaluate<double>(mass_flux_val,
                                                                    buffer,
                                                                    broadcast_function);
    }
  }

  template class EvaporationMassFluxOperatorThicknessIntegration<1>;
  template class EvaporationMassFluxOperatorThicknessIntegration<2>;
  template class EvaporationMassFluxOperatorThicknessIntegration<3>;
} // namespace MeltPoolDG::Evaporation
