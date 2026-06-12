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

        dealii::MatrixFreeOperators::
          CellwiseInverseMassMatrix<dim, -1, n_conserved_variables<dim, n_species>, number>
            inverse(phi);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, dealii::EvaluationFlags::values);

            // Step 1: Compute the cell average values of each component
            ConservedVariablesType<dim, number, n_species> cell_average_value;
            VectorizedArrayType                            cell_volume = 0.;
            for (const unsigned int q : phi.quadrature_point_indices())
              {
                cell_average_value += phi.get_value(q) * phi.JxW(q);
                cell_volume += phi.JxW(q);
              }
            cell_average_value /= cell_volume;

            CellStateView cell_average_state_view(cell_average_value, material_data);

            // Step 2: Check if average density is above the minimum threshold. If not, set the
            // solution to the cell average value.
            VectorizedArrayType below_threshold =
              dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                cell_average_state_view.density(),
                VectorizedArrayType(tolerance),
                VectorizedArrayType(1),
                VectorizedArrayType(0));

            std::vector<ConservedVariablesType<dim, number, n_species>> limited_value_at_q;
            limited_value_at_q.reserve(phi.n_q_points);

            if (below_threshold.sum() == VectorizedArrayType::size())
              {
                for (unsigned int i = 0; i < phi.dofs_per_component; ++i)
                  {
                    phi.submit_dof_value(cell_average_value, i);
                  }
                phi.set_dof_values(dst);
              }
            else
              {
                // the limiting procedure as described in the algorithm needs to be performed and
                // later a check is required to also consider those cells where case 1 applies


                // 1.1 Density limiter
                // determine minimum density at quadrature points
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

                VectorizedArrayType theta_partial_density = 0.0;
                for (const unsigned int q : phi.quadrature_point_indices())
                  {
                    ConservedVariablesType<dim, number, n_species> limited_value;
                    CellStateView limited_value_view(limited_value, material_data);

                    ConservedVariablesType<dim, number, n_species> w_q = phi.get_value(q);
                    CellStateView                                  w_q_view(w_q, material_data);

                    // Step 1: Limit the density
                    VectorizedArrayType theta = 0;
                    if (density_limiter_mask.sum() > 0)
                      {
                        theta =
                          (cell_average_state_view.density() - VectorizedArrayType(tolerance)) /
                          (cell_average_state_view.density() - min_density);

                        theta = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                          theta, VectorizedArrayType(1.), theta, VectorizedArrayType(1.));

                        const VectorizedArrayType limited_density =
                          cell_average_state_view.density() +
                          theta * (w_q_view.density() - cell_average_state_view.density());

                        limited_value_view.density() =
                          dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                            density_limiter_mask,
                            VectorizedArrayType(1.0),
                            limited_density,
                            w_q_view.density());

                        // first limiting of partial density
                        for (unsigned int s = 0; s < n_species - 1; ++s)
                          {
                            const VectorizedArrayType limited_partial_density =
                              cell_average_state_view.partial_density(s) +
                              theta * (w_q_view.partial_density(s) -
                                       cell_average_state_view.partial_density(s));

                            limited_value_view.partial_density(s) =
                              dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                                density_limiter_mask,
                                VectorizedArrayType(1.0),
                                limited_partial_density,
                                w_q_view.partial_density(s));
                          }
                      }
                    else
                      {
                        limited_value_view.density() = w_q_view.density();
                      }

                    // Compute theta for subsequent mass fraction limiting
                    for (unsigned int s = 0; s < n_species - 1; ++s)
                      {
                        VectorizedArrayType theta_q_possibility =
                          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                            limited_value_view.partial_density(s),
                            VectorizedArrayType(0.0),
                            -limited_value_view.partial_density(s) *
                              cell_average_state_view.density() /
                              (cell_average_state_view.partial_density(s) *
                                 limited_value_view.density() -
                               limited_value_view.partial_density(s) *
                                 cell_average_state_view.density()),
                            VectorizedArrayType(-1.0));

                        theta_partial_density =
                          dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                            theta_q_possibility,
                            theta_partial_density,
                            theta_q_possibility,
                            theta_partial_density);
                      }

                    limited_value_at_q.push_back(limited_value);
                  }

                VectorizedArrayType theta_final = std::numeric_limits<number>::max();
                for (unsigned int q : phi.quadrature_point_indices())
                  {
                    // Compute limited partial density
                    CellStateView limited_value_view(limited_value_at_q[q], material_data);
                    for (unsigned int s = 0; s < n_species - 1; ++s)
                      {
                        limited_value_view.partial_density(s) +=
                          theta_partial_density *
                          (cell_average_state_view.partial_density(s) /
                             cell_average_state_view.density() * limited_value_view.density() -
                           limited_value_view.partial_density(s) / limited_value_view.density());
                      }

                    // Compute theta for pressure limiting
                    VectorizedArrayType theta =
                      cell_average_state_view.pressure() /
                      (cell_average_state_view.pressure() - limited_value_view.pressure());
                    theta_final = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                      theta, theta_final, theta, theta_final);
                  }

                std::vector<VectorizedArrayType> final_limited_values(
                  n_conserved_variables<dim, n_species> * phi.quadrature_point_indices().size());
                for (unsigned int q : phi.quadrature_point_indices())
                  {
                    // Perform final pressure limiting
                    ConservedVariablesType<dim, number, n_species> w_new =
                      cell_average_value +
                      theta_final * (limited_value_at_q[q] - cell_average_value);

                    for (unsigned int c = 0; c < n_conserved_variables<dim, n_species>; ++c)
                      final_limited_values[c * phi.quadrature_point_indices().size() + q] =
                        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                          cell_average_state_view.density(),
                          VectorizedArrayType(tolerance),
                          cell_average_value[c],
                          w_new[c]);
                  }
                inverse.transform_from_q_points_to_basis(n_conserved_variables<dim, n_species>,
                                                         final_limited_values.data(),
                                                         phi.begin_dof_values());
                phi.set_dof_values(dst);
              }
          }
      };

    mf_context.mf.cell_loop(limit_loop, dst, src, true);
  }
} // namespace MeltPoolDG::CompressibleFlow
