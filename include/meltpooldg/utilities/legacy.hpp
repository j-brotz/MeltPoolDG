/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, March 2021
 *
 * ---------------------------------------------------------------------*/

#pragma once

// a file to ensure compatibility with the 9.3 release of deal.II
// see also: https://github.com/dealii/dealii/releases/tag/v9.3.0
#if DEAL_II_VERSION_MAJOR < 10

#  include <deal.II/base/mpi.h>

#  include <deal.II/numerics/vector_tools.h>

namespace dealii ::VectorTools
{
  template <int n_components, int dim, int spacedim, typename VectorType>
  inline std::vector<typename FEPointEvaluation<n_components, dim>::gradient_type>
  point_gradients(const Utilities::MPI::RemotePointEvaluation<dim, spacedim> &cache,
                  const DoFHandler<dim, spacedim> &                           dof_handler,
                  const VectorType &                                          vector,
                  const EvaluationFlags::EvaluationFlags flags = EvaluationFlags::avg)
  {
    using value_type = typename FEPointEvaluation<n_components, dim>::gradient_type;

    Assert(cache.is_ready(),
           ExcMessage("Utilities::MPI::RemotePointEvaluation is not ready yet! "
                      "Please call Utilities::MPI::RemotePointEvaluation::reinit() "
                      "yourself or the other point_values(), which does this for "
                      "you."));

    Assert(&dof_handler.get_triangulation() == &cache.get_triangulation(),
           ExcMessage("The provided Utilities::MPI::RemotePointEvaluation and DoFHandler "
                      "object have been set up with different Triangulation objects, "
                      "a scenario not supported!"));

    // evaluate values at points if possible
    const auto evaluation_point_results = [&]() {
      // helper function for accessing the global vector and interpolating
      // the results onto the points
      const auto evaluation_function = [&](auto &values, const auto &cell_data) {
        std::vector<typename VectorType::value_type> solution_values;

        std::vector<std::unique_ptr<FEPointEvaluation<n_components, dim>>> evaluators(
          dof_handler.get_fe_collection().size());

        const auto get_evaluator =
          [&](const unsigned int active_fe_index) -> FEPointEvaluation<n_components, dim> & {
          if (evaluators[active_fe_index] == nullptr)
            evaluators[active_fe_index] = std::make_unique<FEPointEvaluation<n_components, dim>>(
              cache.get_mapping(), dof_handler.get_fe(active_fe_index), update_gradients);

          return *evaluators[active_fe_index];
        };

        for (unsigned int i = 0; i < cell_data.cells.size(); ++i)
          {
            typename DoFHandler<dim>::active_cell_iterator cell = {&cache.get_triangulation(),
                                                                   cell_data.cells[i].first,
                                                                   cell_data.cells[i].second,
                                                                   &dof_handler};

            const ArrayView<const Point<dim>> unit_points(cell_data.reference_point_values.data() +
                                                            cell_data.reference_point_ptrs[i],
                                                          cell_data.reference_point_ptrs[i + 1] -
                                                            cell_data.reference_point_ptrs[i]);

            solution_values.resize(dof_handler.get_fe(cell->active_fe_index()).n_dofs_per_cell());
            cell->get_dof_values(vector, solution_values.begin(), solution_values.end());

            auto &evaluator = get_evaluator(cell->active_fe_index());

            evaluator.reinit(cell, unit_points);
            evaluator.evaluate(solution_values, dealii::EvaluationFlags::gradients);

            for (unsigned int q = 0; q < unit_points.size(); ++q)
              values[q + cell_data.reference_point_ptrs[i]] = evaluator.get_gradient(q);
          }
      };

      std::vector<value_type> evaluation_point_results;
      std::vector<value_type> buffer;

      cache.template evaluate_and_process<value_type>(evaluation_point_results,
                                                      buffer,
                                                      evaluation_function);

      return evaluation_point_results;
    }();

    if (cache.is_map_unique())
      {
        // each point has exactly one result (unique map)
        return evaluation_point_results;
      }
    else
      {
        // map is not unique (multiple or no results): postprocessing is needed
        std::vector<value_type> unique_evaluation_point_results(cache.get_point_ptrs().size() - 1);

        const auto &ptr = cache.get_point_ptrs();

        for (unsigned int i = 0; i < ptr.size() - 1; ++i)
          {
            const auto n_entries = ptr[i + 1] - ptr[i];
            if (n_entries == 0)
              continue;

            unique_evaluation_point_results[i] =
              internal::reduce(flags,
                               ArrayView<const value_type>(evaluation_point_results.data() + ptr[i],
                                                           n_entries));
          }

        return unique_evaluation_point_results;
      }
  }
} // namespace dealii::VectorTools

namespace dealii::internal
{
  template <int fe_degree, int n_q_points_1d, int dim, typename Number>
  bool
  transform_from_q_points_to_basis(
    const unsigned int n_desired_components,
    const FEEvaluationBaseData<dim, typename Number::value_type, false, Number> &fe_eval,
    const Number *                                                               in_array,
    Number *                                                                     out_array)
  {
    static const bool do_inplace = fe_degree > -1 && (fe_degree + 1 == n_q_points_1d);

    Assert(fe_eval.get_shape_info().element_type != MatrixFreeFunctions::ElementType::tensor_none,
           ExcNotImplemented());

    const auto &inverse_shape = do_inplace ?
                                  fe_eval.get_shape_info().data.front().inverse_shape_values_eo :
                                  fe_eval.get_shape_info().data.front().inverse_shape_values;

    const unsigned int dofs_per_component = do_inplace ?
                                              Utilities::pow(fe_degree + 1, dim) :
                                              fe_eval.get_shape_info().dofs_per_component_on_cell;
    const unsigned int n_q_points =
      do_inplace ? Utilities::pow(fe_degree + 1, dim) : fe_eval.get_shape_info().n_q_points;

    internal::EvaluatorTensorProduct<do_inplace ? internal::evaluate_evenodd :
                                                  internal::evaluate_general,
                                     dim,
                                     fe_degree + 1,
                                     n_q_points_1d,
                                     Number>
      evaluator(AlignedVector<Number>(),
                AlignedVector<Number>(),
                inverse_shape,
                fe_eval.get_shape_info().data.front().fe_degree + 1,
                fe_eval.get_shape_info().data.front().n_q_points_1d);

    for (unsigned int d = 0; d < n_desired_components; ++d)
      {
        const Number *in  = in_array + d * n_q_points;
        Number *      out = out_array + d * dofs_per_component;

        auto temp_1 = do_inplace ? out : fe_eval.get_scratch_data().begin();
        auto temp_2 = do_inplace ? out : (temp_1 + std::max(n_q_points, dofs_per_component));

        if (dim == 3)
          {
            evaluator.template hessians<2, false, false>(in, temp_1);
            evaluator.template hessians<1, false, false>(temp_1, temp_2);
            evaluator.template hessians<0, false, false>(temp_2, out);
          }
        if (dim == 2)
          {
            evaluator.template hessians<1, false, false>(in, temp_1);
            evaluator.template hessians<0, false, false>(temp_1, out);
          }
        if (dim == 1)
          evaluator.template hessians<0, false, false>(in, out);
      }
    return false;
  }
} // namespace dealii::internal

namespace dealii::Utilities::MPI
{
  template <typename T>
  T
  reduce(const T &                                     local_value,
         const MPI_Comm &                              comm,
         const std::function<T(const T &, const T &)> &combiner)
  {
    return all_reduce(local_value, comm, combiner);
  }
} // namespace dealii::Utilities::MPI

#endif
