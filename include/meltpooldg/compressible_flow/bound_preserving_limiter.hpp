#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/state_views_n_species.hpp>
#include <meltpooldg/utilities/limiters.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <limits>
#include <utility>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number, int n_species>
  void
  bound_preserving_limiter(const MatrixFreeContext<dim, number>                     &mf_context,
                           dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                           const dealii::LinearAlgebra::distributed::Vector<number> &src,
                           const Utilities::LimiterData<number>                     &limiter_data,
                           const MaterialPhaseData<number>                          &material_data)
  {
    if (not limiter_data.apply_limiter)
      return;

    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using CellStateView =
      NSpeciesDofStateView<dim, n_species, number, ConservedVariablesType<dim, number, n_species>>;

    constexpr number tolerance = 1e-13;

    // TODO: Ensure the usage of Gauss-Lobatto quadrature points in the matrix-free object

    const std::function<void(const dealii::MatrixFree<dim, number, VectorizedArrayType> &,
                             dealii::LinearAlgebra::distributed::Vector<number> &,
                             const dealii::LinearAlgebra::distributed::Vector<number> &,
                             const std::pair<unsigned int, unsigned int> &)>
      limit_loop = [&](const dealii::MatrixFree<dim, number>                    &matrix_free,
                       dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                       const dealii::LinearAlgebra::distributed::Vector<number> &src,
                       const std::pair<unsigned int, unsigned int>              &cell_range) {
        FECellIntegrator<dim, n_conserved_variables<dim, n_species>, number> phi(
          matrix_free, mf_context.dof_idx, mf_context.quad_idx);

        //
        //
        const auto compute_cell_average = [&]() -> ConservedVariablesType<dim, number, n_species> {
          ConservedVariablesType<dim, number, n_species> cell_average_value;
          VectorizedArrayType                            cell_volume = 0.;
          for (const unsigned int q : phi.quadrature_point_indices())
            {
              cell_average_value += phi.get_value(q) * phi.JxW(q);
              cell_volume += phi.JxW(q);
            }
          cell_average_value /= cell_volume;

          return cell_average_value;
        };

        //
        //
        const auto limit_density = [&](const CellStateView &cell_average_state_view) {
          VectorizedArrayType min_density = std::numeric_limits<number>::max();
          for (const unsigned int q : phi.quadrature_point_indices())
            {
              const auto density = phi.get_value(q)[0];
              min_density = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                density, min_density, density, min_density);
            }

          VectorizedArrayType density_limiter_mask =
            dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              min_density,
              VectorizedArrayType(0.0),
              VectorizedArrayType(1),
              VectorizedArrayType(0));

          VectorizedArrayType theta = 0;
          if (density_limiter_mask.sum() > 0)
            {
              theta = (cell_average_state_view.density() - VectorizedArrayType(tolerance)) /
                      (cell_average_state_view.density() - min_density);

              theta = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                theta, VectorizedArrayType(1.), theta, VectorizedArrayType(1.));

              for (unsigned int dof_index = 0; dof_index < phi.dofs_per_component; ++dof_index)
                {
                  ConservedVariablesType<dim, number, n_species> limited_value;
                  CellStateView limited_value_view(limited_value, material_data);

                  ConservedVariablesType<dim, number, n_species> w_dof =
                    phi.get_dof_value(dof_index);
                  CellStateView w_dof_view(w_dof, material_data);

                  const VectorizedArrayType limited_density =
                    cell_average_state_view.density() +
                    theta * (w_dof_view.density() - cell_average_state_view.density());

                  limited_value_view.density() =
                    dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                      density_limiter_mask,
                      VectorizedArrayType(1.0),
                      limited_density,
                      w_dof_view.density());

                  if constexpr (n_species > 1)
                    {
                      for (unsigned int s = 0; s < n_species - 1; ++s)
                        {
                          const VectorizedArrayType limited_partial_density =
                            cell_average_state_view.partial_density(s) +
                            theta * (w_dof_view.partial_density(s) -
                                     cell_average_state_view.partial_density(s));

                          limited_value_view.partial_density(s) =
                            dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                              density_limiter_mask,
                              VectorizedArrayType(1.0),
                              limited_partial_density,
                              w_dof_view.partial_density(s));
                        }
                    }

                  phi.submit_dof_value(limited_value, dof_index);
                }
            }
        };


        //
        //
        const auto limit_partial_density = [&](const CellStateView &cell_average_state_view) {
          VectorizedArrayType theta = 0;
          for (const unsigned int q : phi.quadrature_point_indices())
            {
              ConservedVariablesType<dim, number, n_species> w_q = phi.get_value(q);
              CellStateView                                  w_q_view(w_q, material_data);

              for (unsigned int s = 0; s < n_species; ++s)
                {
                  VectorizedArrayType theta_q =
                    (-w_q_view.partial_density(s) * cell_average_state_view.density()) /
                    (cell_average_state_view.partial_density(s) * w_q_view.density() -
                     w_q_view.partial_density(s) * cell_average_state_view.density());

                  theta_q =
                    dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
                      w_q_view.partial_density(s),
                      VectorizedArrayType(0.0),
                      theta_q,
                      VectorizedArrayType(std::numeric_limits<number>::min()));

                  theta = dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                    theta_q, theta, theta_q, theta);
                }
            }

          ConservedVariablesType<dim, number, n_species> limited_value;
          CellStateView limited_value_view(limited_value, material_data);
          for (unsigned int dof_index = 0; dof_index < phi.dofs_per_component; ++dof_index)
            {
              ConservedVariablesType<dim, number, n_species> w_dof = phi.get_dof_value(dof_index);
              CellStateView                                  w_dof_view(w_dof, material_data);
              for (unsigned int s = 0; s < n_species; ++s)
                {
                  w_dof_view.partial_density(s) +=
                    theta * (cell_average_state_view.partial_density(s) /
                               cell_average_state_view.density() * w_dof_view.density() -
                             w_dof_view.partial_density(s));
                }
              phi.submit_dof_value(w_dof, dof_index);
            }
        };

        const auto modify_pressure = [&](const CellStateView &cell_average_state_view) {
          VectorizedArrayType theta = std::numeric_limits<number>::max();
          for (unsigned int q : phi.quadrature_point_indices())
            {
              ConservedVariablesType<dim, number, n_species> w_q = phi.get_value(q);
              CellStateView                                  w_q_view(w_q, material_data);
              VectorizedArrayType                            theta_q =
                cell_average_state_view.pressure() /
                (cell_average_state_view.pressure() - w_q_view.pressure());

              // TODO: Do not only check pressure but also density and partial density
              theta_q = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                w_q_view.pressure(), VectorizedArrayType(0.0), theta_q, VectorizedArrayType(1.0));

              theta = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(theta_q,
                                                                                        theta,
                                                                                        theta_q,
                                                                                        theta);
            }

          for (unsigned int dof_index = 0; dof_index < phi.dofs_per_component; ++dof_index)
            {
              ConservedVariablesType<dim, number, n_species> w_dof = phi.get_dof_value(dof_index);
              w_dof =
                cell_average_state_view.value() + theta * (w_dof - cell_average_state_view.value());

              phi.submit_dof_value(w_dof, dof_index);
            }
        };

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, dealii::EvaluationFlags::values);

            // Step 1: Compute the cell average values of each component
            ConservedVariablesType<dim, number, n_species> cell_average_value =
              compute_cell_average();
            CellStateView cell_average_state_view(cell_average_value, material_data);

            VectorizedArrayType mask_average_density_greater_tolerance =
              dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                cell_average_state_view.density(),
                VectorizedArrayType(tolerance),
                VectorizedArrayType(1),
                VectorizedArrayType(0));

            if (mask_average_density_greater_tolerance.sum() == 0)
              {
                // If the cell average density is below the tolerance, set all values to the cell
                // average
                for (unsigned int dof_index = 0; dof_index < phi.dofs_per_component; ++dof_index)
                  {
                    phi.submit_dof_value(cell_average_value, dof_index);
                  }
              }
            else
              {
                // Step 2: Limit the density if necessary
                limit_density(cell_average_state_view);
                phi.evaluate(dealii::EvaluationFlags::values);

                // Step 3: Limit the partial densities
                limit_partial_density(cell_average_state_view);
                phi.evaluate(dealii::EvaluationFlags::values);

                // Step 4: Modify the pressure
                modify_pressure(cell_average_state_view);

                if (mask_average_density_greater_tolerance.sum() < VectorizedArrayType::size())
                  {
                    // If some average densities are below the tolerance
                    for (unsigned int dof_index = 0; dof_index < phi.dofs_per_component;
                         ++dof_index)
                      {
                        ConservedVariablesType<dim, number, n_species> limited_dof =
                          phi.get_dof_value(dof_index);
                        for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
                          {
                            limited_dof[c] =
                              dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                                mask_average_density_greater_tolerance,
                                VectorizedArrayType(1.0),
                                limited_dof[c],
                                cell_average_value[c]);
                          }
                        phi.submit_dof_value(limited_dof, dof_index);
                      }
                  }
              }
            phi.set_dof_values(dst);
          }
      };

    dst.zero_out_ghost_values();
    mf_context.mf.cell_loop(limit_loop, dst, src, true);
    dst.update_ghost_values();
  }
} // namespace MeltPoolDG::CompressibleFlow
