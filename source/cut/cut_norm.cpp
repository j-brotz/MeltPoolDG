#include <meltpooldg/cut/cut_norm.hpp>
//
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <cmath>

namespace MeltPoolDG::CutUtil
{
  template <int dim, typename number>
  number
  compute_cut_L2_norm(
    const dealii::LinearAlgebra::distributed::Vector<number>               &solution,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
                            &mapping_info_cells,
    const bool               is_two_phase,
    const dealii::FE_Q<dim> &reference_element,
    const unsigned int       dof_idx,
    const unsigned int       quad_idx)
  {
    static constexpr int n_components = 1;

    if (not solution.has_ghost_elements())
      solution.update_ghost_values();

    number error_L2_squared = 0.;

    for (unsigned int cell_batch = 0; cell_batch < matrix_free.n_cell_batches(); ++cell_batch)
      {
        const auto cell_category = matrix_free.get_cell_category(cell_batch);
        if (cell_category == CellCategory::liquid)
          {
            dealii::FECellIntegrator<dim, n_components, number> eval_l(
              matrix_free,
              dof_idx /*dof_no*/,
              quad_idx /*quad_no*/,
              0 /*selected component*/,
              cell_category /*active_fe_index*/);

            eval_l.reinit(cell_batch);
            eval_l.read_dof_values_plain(solution);
            eval_l.evaluate(dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_l.quadrature_point_indices())
              {
                const auto error_squared =
                  dealii::Utilities::fixed_power<2>(eval_l.get_value(q)) * eval_l.JxW(q);
                for (unsigned int lane = 0;
                     lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
                     ++lane)
                  error_L2_squared += error_squared[lane];
              }
          }
        else if (cell_category == CellCategory::gas and is_two_phase)
          {
            dealii::FECellIntegrator<dim, n_components, number> eval_g(
              matrix_free,
              dof_idx /*dof_no*/,
              quad_idx /*quad_no*/,
              1 /*selected component*/,
              cell_category /*active_fe_index*/);

            eval_g.reinit(cell_batch);
            eval_g.read_dof_values_plain(solution);
            eval_g.evaluate(dealii::EvaluationFlags::values);

            for (const unsigned int q : eval_g.quadrature_point_indices())
              {
                const auto error_squared =
                  dealii::Utilities::fixed_power<2>(eval_g.get_value(q)) * eval_g.JxW(q);
                for (unsigned int lane = 0;
                     lane < matrix_free.n_active_entries_per_cell_batch(cell_batch);
                     ++lane)
                  error_L2_squared += error_squared[lane];
              }
          }
        else if (cell_category == CellCategory::intersected and not is_two_phase)
          {
            // use FEEvaluation and FEdealii::FEPointEvaluation<n_components, dim, dim,
            // dealii::VectorizedArray<number>>uation in combination for intersected cells
            dealii::FECellIntegrator<dim, n_components, number> eval_cell_l(
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
                  {
                    const auto error_squared =
                      dealii::Utilities::fixed_power<2>(eval_subdomain_l.get_value(q_batch)) *
                      eval_subdomain_l.JxW(q_batch);
                    for (unsigned int q_lane = 0;
                         q_lane < eval_subdomain_l.n_active_entries_per_quadrature_batch(q_batch);
                         ++q_lane)
                      error_L2_squared += error_squared[q_lane];
                  }
              }
          }
        else if (cell_category == CellCategory::intersected and is_two_phase)
          {
            // use FEEvaluation and FEdealii::FEPointEvaluation<n_components, dim, dim,
            // dealii::VectorizedArray<number>>uation in combination for intersected cells
            dealii::FECellIntegrator<dim, n_components, number> eval_cell_l(
              matrix_free,
              dof_idx /*dof_no*/,
              quad_idx /*quad_no*/,
              0 /*selected component*/,
              cell_category /*active_fe_index*/);
            dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>>
              eval_subdomain_l(*mapping_info_cells[0], reference_element);
            dealii::FECellIntegrator<dim, n_components, number> eval_cell_g(
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
                  {
                    const auto error_squared =
                      dealii::Utilities::fixed_power<2>(eval_subdomain_l.get_value(q_batch)) *
                      eval_subdomain_l.JxW(q_batch);
                    for (unsigned int q_lane = 0;
                         q_lane < eval_subdomain_l.n_active_entries_per_quadrature_batch(q_batch);
                         ++q_lane)
                      error_L2_squared += error_squared[q_lane];
                  }

                for (const unsigned int q_batch : eval_subdomain_g.quadrature_point_indices())
                  {
                    const auto error_squared =
                      dealii::Utilities::fixed_power<2>(eval_subdomain_g.get_value(q_batch)) *
                      eval_subdomain_g.JxW(q_batch);
                    for (unsigned int q_lane = 0;
                         q_lane < eval_subdomain_g.n_active_entries_per_quadrature_batch(q_batch);
                         ++q_lane)
                      error_L2_squared += error_squared[q_lane];
                  }
              }
          }
      }

    return std::sqrt(
      dealii::Utilities::MPI::sum(error_L2_squared, solution.get_mpi_communicator()));
  }


  template double
  compute_cut_L2_norm(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<1, double, dealii::VectorizedArray<double>> &,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>>>> &,
    const bool,
    const dealii::FE_Q<1> &,
    const unsigned int,
    const unsigned int);
  template double
  compute_cut_L2_norm(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<2, double, dealii::VectorizedArray<double>> &,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>>>> &,
    const bool,
    const dealii::FE_Q<2> &,
    const unsigned int,
    const unsigned int);
  template double
  compute_cut_L2_norm(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<3, double, dealii::VectorizedArray<double>> &,
    const std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>>>> &,
    const bool,
    const dealii::FE_Q<3> &,
    const unsigned int,
    const unsigned int);
} // namespace MeltPoolDG::CutUtil