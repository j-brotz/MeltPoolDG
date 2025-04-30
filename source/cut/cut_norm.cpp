#include <meltpooldg/cut/cut_norm.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <cmath>
#include <functional>

namespace MeltPoolDG::CutUtil
{
  template <int dim, typename number>
  number
  compute_cut_norm(
    const dealii::LinearAlgebra::distributed::Vector<number>               &solution,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
                            &mapping_info_cells,
    const bool               is_two_phase,
    const dealii::FE_Q<dim> &reference_element,
    const unsigned int       dof_idx,
    const unsigned int       quad_idx,
    const NormType           norm_type)
  {
    static constexpr int n_components = 1;

#ifdef DEBUG
    auto n_comp = matrix_free.get_dof_handler(dof_idx).get_fe().n_components();
    // for two phase cut, dof_handler_req.get_fe().n_components() returns double the number
    // of components
    if (is_two_phase)
      n_comp /= 2;
    Assert(n_components == n_comp,
           dealii::ExcMessage("For now, this function is only implemened for n_components = 1!"));
#endif

    const bool update_ghosts = not solution.has_ghost_elements();
    if (update_ghosts)
      solution.update_ghost_values();

    number sum = 0.;
    std::function<void(const dealii::VectorizedArray<number> &,
                       const dealii::VectorizedArray<number> &,
                       const unsigned int)>
      operation;
    switch (norm_type)
      {
          case NormType::L2_norm: {
            operation = [&](const dealii::VectorizedArray<number> &values,
                            const dealii::VectorizedArray<number> &JxW,
                            const unsigned int                     n_active_lanes) {
              const auto contrib = dealii::Utilities::fixed_power<2>(values) * JxW;
              for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                sum += contrib[lane];
            };
            break;
          }
          case NormType::L1_norm: {
            operation = [&](const dealii::VectorizedArray<number> &values,
                            const dealii::VectorizedArray<number> &JxW,
                            const unsigned int                     n_active_lanes) {
              const auto contrib = values * JxW;
              for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                sum += contrib[lane];
            };
            break;
          }
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }


    for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
      {
        const auto cell_category = matrix_free.get_cell_category(cell_batch);
        if (cell_category == CellCategory::liquid)
          {
            FECellIntegrator<dim, n_components, number> eval_l(matrix_free,
                                                               dof_idx /*dof_no*/,
                                                               quad_idx /*quad_no*/,
                                                               0 /*selected component*/,
                                                               cell_category /*active_fe_index*/);

            eval_l.reinit(cell_batch);
            eval_l.read_dof_values_plain(solution);
            eval_l.evaluate(dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_l.quadrature_point_indices())
              operation(eval_l.get_value(q),
                        eval_l.JxW(q),
                        matrix_free.n_active_entries_per_cell_batch(cell_batch));
          }
        else if (cell_category == CellCategory::gas and is_two_phase)
          {
            FECellIntegrator<dim, n_components, number> eval_g(matrix_free,
                                                               dof_idx /*dof_no*/,
                                                               quad_idx /*quad_no*/,
                                                               1 /*selected component*/,
                                                               cell_category /*active_fe_index*/);

            eval_g.reinit(cell_batch);
            eval_g.read_dof_values_plain(solution);
            eval_g.evaluate(dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_g.quadrature_point_indices())
              operation(eval_g.get_value(q),
                        eval_g.JxW(q),
                        matrix_free.n_active_entries_per_cell_batch(cell_batch));
          }
        else if (cell_category == CellCategory::intersected and not is_two_phase)
          {
            // use FEEvaluation and FEdealii::FEPointEvaluation<n_components, dim, dim,
            // dealii::VectorizedArray<number>>uation in combination for intersected cells
            FECellIntegrator<dim, n_components, number> eval_cell_l(
              matrix_free,
              dof_idx /*dof_no*/,
              quad_idx /*quad_no*/,
              0 /*selected component*/,
              cell_category /*active_fe_index*/);
            dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>
              eval_subdomain_l(*mapping_info_cells[0], reference_element);

            eval_cell_l.reinit(cell_batch);
            eval_cell_l.read_dof_values_plain(solution);

            for (unsigned int cell_lane = 0;
                 cell_lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
                 ++cell_lane)
              {
                evaluate_intersected_domain<dim, number>(eval_subdomain_l,
                                                         eval_cell_l,
                                                         dealii::EvaluationFlags::values,
                                                         cell_batch,
                                                         cell_lane,
                                                         reference_element.n_dofs_per_cell());

                for (const unsigned int q_batch : eval_subdomain_l.quadrature_point_indices())
                  operation(eval_subdomain_l.get_value(q_batch),
                            eval_subdomain_l.JxW(q_batch),
                            eval_subdomain_l.n_active_entries_per_quadrature_batch(q_batch));
              }
          }
        else if (cell_category == CellCategory::intersected and is_two_phase)
          {
            // use FEEvaluation and FEdealii::FEPointEvaluation<n_components, dim, dim,
            // dealii::VectorizedArray<number>>uation in combination for intersected cells
            FECellIntegrator<dim, n_components, number> eval_cell_l(
              matrix_free,
              dof_idx /*dof_no*/,
              quad_idx /*quad_no*/,
              0 /*selected component*/,
              cell_category /*active_fe_index*/);
            dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>
              eval_subdomain_l(*mapping_info_cells[0], reference_element);
            FECellIntegrator<dim, n_components, number> eval_cell_g(
              matrix_free,
              dof_idx /*dof_no*/,
              quad_idx /*quad_no*/,
              1 /*selected component*/,
              cell_category /*active_fe_index*/);
            dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>
              eval_subdomain_g(*mapping_info_cells[1], reference_element);


            eval_cell_l.reinit(cell_batch);
            eval_cell_l.read_dof_values_plain(solution);
            eval_cell_g.reinit(cell_batch);
            eval_cell_g.read_dof_values_plain(solution);

            for (unsigned int cell_lane = 0;
                 cell_lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
                 ++cell_lane)
              {
                evaluate_intersected_domain<dim, number>(eval_subdomain_l,
                                                         eval_cell_l,
                                                         dealii::EvaluationFlags::values,
                                                         cell_batch,
                                                         cell_lane,
                                                         reference_element.n_dofs_per_cell());
                evaluate_intersected_domain<dim, number>(eval_subdomain_g,
                                                         eval_cell_g,
                                                         dealii::EvaluationFlags::values,
                                                         cell_batch,
                                                         cell_lane,
                                                         reference_element.n_dofs_per_cell());

                for (const unsigned int q_batch : eval_subdomain_l.quadrature_point_indices())
                  operation(eval_subdomain_l.get_value(q_batch),
                            eval_subdomain_l.JxW(q_batch),
                            eval_subdomain_l.n_active_entries_per_quadrature_batch(q_batch));

                for (const unsigned int q_batch : eval_subdomain_g.quadrature_point_indices())
                  operation(eval_subdomain_g.get_value(q_batch),
                            eval_subdomain_g.JxW(q_batch),
                            eval_subdomain_g.n_active_entries_per_quadrature_batch(q_batch));
              }
          }
      }

    if (update_ghosts)
      solution.zero_out_ghost_values();

    switch (norm_type)
      {
        case NormType::L2_norm:
          return std::sqrt(dealii::Utilities::MPI::sum(sum, solution.get_mpi_communicator()));
        case NormType::L1_norm:
          return dealii::Utilities::MPI::sum(sum, solution.get_mpi_communicator());
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    return 0.0;
  }


  template double
  compute_cut_norm(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<1, double, dealii::VectorizedArray<double>> &,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>>>> &,
    const bool,
    const dealii::FE_Q<1> &,
    const unsigned int,
    const unsigned int,
    const NormType);
  template double
  compute_cut_norm(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<2, double, dealii::VectorizedArray<double>> &,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>>>> &,
    const bool,
    const dealii::FE_Q<2> &,
    const unsigned int,
    const unsigned int,
    const NormType);
  template double
  compute_cut_norm(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<3, double, dealii::VectorizedArray<double>> &,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>>>> &,
    const bool,
    const dealii::FE_Q<3> &,
    const unsigned int,
    const unsigned int,
    const NormType);
} // namespace MeltPoolDG::CutUtil