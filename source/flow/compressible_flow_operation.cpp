#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <meltpooldg/flow/compressible_flow_operation.hpp>
#include <meltpooldg/flow/compressible_flow_operator_explicit.hpp>
#include <meltpooldg/flow/compressible_flow_operator_implicit.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>


namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  CompressibleFlowOperation<dim, number>::CompressibleFlowOperation(
    const ScratchData<dim>     &scratch_data_in,
    const CompressibleFlowData &comp_flow_data_in,
    const unsigned int          comp_flow_dof_idx_in,
    const unsigned int          comp_flow_quad_idx_in)
    : scratch_data_(scratch_data_in)
    , comp_flow_data_(comp_flow_data_in)
    , comp_flow_dof_idx(comp_flow_dof_idx_in)
    , comp_flow_quad_idx(comp_flow_quad_idx_in)
  {
    setup_operator_and_time_integrator();
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::reinit()
  {
    solution_history_.apply(
      [&scratch_data = scratch_data_, comp_flow_dof_idx = comp_flow_dof_idx](VectorType &v) {
        scratch_data.initialize_dof_vector(v, comp_flow_dof_idx);
      });
    comp_flow_operator_->reinit();
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::distribute_dofs(DoFHandler<dim> &dof_handler) const
  {
    FiniteElementUtils::distribute_dofs<dim, dim + 2>(comp_flow_data_.fe, dof_handler);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::solve(double current_time, double time_step)
  {
    solution_history_.commit_old_solutions();
    std::function<void(number, VectorType &, const VectorType &)> pre_processing =
      [&](number time, VectorType &, const VectorType &) -> void {
      comp_flow_operator_->update_boundary_conditions(time);
    };

    solution_history_.update_ghost_values();

    comp_flow_operator_->advance_time_step(current_time, time_step, pre_processing);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::set_body_force(
    std::unique_ptr<Function<dim>> body_force_in)
  {
    comp_flow_operator_->set_body_force(std::move(body_force_in));
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::set_inflow_boundary(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &inflow_bc)
  {
    comp_flow_operator_->set_inflow_boundary(inflow_bc);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::set_subsonic_outflow_with_fixed_static_pressure(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &outflow_fixed_pressure_bc)
  {
    comp_flow_operator_->set_subsonic_outflow_with_fixed_static_pressure(outflow_fixed_pressure_bc);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::set_subsonic_outflow_with_fixed_energy(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &outflow_fixed_energy_bc)
  {
    comp_flow_operator_->set_subsonic_outflow_with_fixed_energy(outflow_fixed_energy_bc);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::set_slip_wall_boundary(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &slip_wall_bc)
  {
    comp_flow_operator_->set_slip_wall_boundary(slip_wall_bc);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::set_no_slip_adiabatic_wall_boundary(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &no_slip_wall_bc)
  {
    comp_flow_operator_->set_no_slip_adiabatic_wall_boundary(no_slip_wall_bc);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::set_initial_condition(const Function<dim> &function)
  {
    FECellIntegrator<dim, dim + 2, number> phi(scratch_data_.get_matrix_free(),
                                               comp_flow_dof_idx,
                                               comp_flow_quad_idx);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, number> inverse(phi);
    solution_history_.get_current_solution().zero_out_ghost_values();
    for (unsigned int cell = 0; cell < scratch_data_.get_matrix_free().n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
          phi.submit_dof_value(
            VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
              function, phi.quadrature_point(q)),
            q);
        inverse.transform_from_q_points_to_basis(dim + 2,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(solution_history_.get_current_solution());
      }
  }

  template <int dim, typename number>
  number
  CompressibleFlowOperation<dim, number>::compute_minimum_density() const
  {
    TimerOutput::Scope t(scratch_data_.get_timer(), "compute transport speed");
    // only read density
    FECellIntegrator<dim, 1, number> phi(scratch_data_.get_matrix_free(),
                                         comp_flow_dof_idx,
                                         comp_flow_quad_idx);
    solution_history_.get_current_solution().update_ghost_values();

    double min_density = std::numeric_limits<double>::max();

    for (unsigned int cell = 0; cell < scratch_data_.get_matrix_free().n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution_history_.get_current_solution(), EvaluationFlags::values);
        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto density = phi.get_value(q);
            for (unsigned int lane = 0;
                 lane < scratch_data_.get_matrix_free().n_active_entries_per_cell_batch(cell);
                 ++lane)
              min_density = std::min(density[lane], min_density);
          }
      }

    min_density = Utilities::MPI::min(min_density, scratch_data_.get_mpi_comm());

    return min_density;
  }

  template <int dim, typename number>
  number
  CompressibleFlowOperation<dim, number>::compute_convective_time_step_limit() const
  {
    TimerOutput::Scope                     t(scratch_data_.get_timer(), "compute transport speed");
    number                                 max_transport              = 0;
    number                                 convective_time_step_limit = 0.;
    FECellIntegrator<dim, dim + 2, number> phi(scratch_data_.get_matrix_free(),
                                               comp_flow_dof_idx,
                                               comp_flow_quad_idx);

    for (unsigned int cell = 0; cell < scratch_data_.get_matrix_free().n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution_history_.get_current_solution(), EvaluationFlags::values);
        VectorizedArray<number> local_max = 0.;
        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto conserved_variables = phi.get_value(q);
            const auto velocity =
              CompressibleFlowCalculators<dim, number>::calculate_velocity(conserved_variables);
            const auto pressure =
              CompressibleFlowCalculators<dim, number>::calculate_pressure(conserved_variables,
                                                                           comp_flow_data_.gamma);

            const auto              inverse_jacobian = phi.inverse_jacobian(q);
            const auto              convective_speed = inverse_jacobian * velocity;
            VectorizedArray<number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

            const auto speed_of_sound =
              std::sqrt(comp_flow_data_.gamma * pressure / conserved_variables[0]);

            Tensor<1, dim, VectorizedArray<number>> eigenvector;
            for (unsigned int d = 0; d < dim; ++d)
              eigenvector[d] = 1.;
            for (unsigned int i = 0; i < 5 /* number of iterations */; ++i)
              {
                eigenvector = transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
                VectorizedArray<number> eigenvector_norm = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  eigenvector_norm = std::max(eigenvector_norm, std::abs(eigenvector[d]));
                eigenvector /= eigenvector_norm;
              }
            const auto jac_times_ev = inverse_jacobian * eigenvector;
            const auto max_eigenvalue =
              std::sqrt((jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
            local_max = std::max(local_max, max_eigenvalue * speed_of_sound + convective_limit);
          }

        // Similarly to the previous function, we must make sure to accumulate
        // speed only on the valid cells of a cell batch.
        for (unsigned int v = 0;
             v < scratch_data_.get_matrix_free().n_active_entries_per_cell_batch(cell);
             ++v)
          max_transport = std::max(max_transport, local_max[v]);
      }

    max_transport = Utilities::MPI::max(max_transport, scratch_data_.get_mpi_comm());

    convective_time_step_limit = comp_flow_data_.courant_number /
                                 std::pow(scratch_data_.get_degree(comp_flow_dof_idx), 1.5) /
                                 max_transport;

    return convective_time_step_limit;
  }

  template <int dim, typename number>
  number
  CompressibleFlowOperation<dim, number>::compute_time_step_size(const bool do_print) const
  {
    const double min_density = compute_minimum_density();

    AssertThrow(min_density > 0, ExcMessage("Minimum density must not be zero."));

    const double viscous_time_step_limit =
      (comp_flow_data_.dynamic_viscosity > 0) ?
        comp_flow_data_.viscous_courant_number /
          std::pow(scratch_data_.get_degree(comp_flow_dof_idx), 3) *
          std::pow(scratch_data_.get_min_cell_size(), 2) * min_density /
          comp_flow_data_.dynamic_viscosity :
        std::numeric_limits<number>::max();

    const double convective_time_step_limit = compute_convective_time_step_limit();
    const double time_step = std::min(convective_time_step_limit, viscous_time_step_limit);

    if (do_print)
      {
        scratch_data_.get_pcout(1) << "Time step size: " << time_step
                                   << ", convective time step limit: " << convective_time_step_limit
                                   << ", viscous time step limit: " << viscous_time_step_limit
                                   << ",\nminimum h: " << scratch_data_.get_min_cell_size()
                                   << ", minimum density: " << min_density << std::endl
                                   << std::endl;
      }

    return time_step;
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    std::vector<std::string> names;
    names.emplace_back("density");
    for (unsigned int d = 0; d < dim; ++d)
      names.emplace_back("momentum");

    names.emplace_back("energy");

    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(scratch_data_.get_dof_handler(comp_flow_dof_idx),
                             solution_history_.get_current_solution(),
                             names,
                             interpretation);

    // TODO: Output primitive variables
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperation<dim, number>::setup_operator_and_time_integrator()
  {
    // cut operator was already created in the constructor
    if (comp_flow_data_.domain_representation_type == "cut")
      return;

    if (time_integrator_scheme_is_explicit(comp_flow_data_.time_integrator.integrator_type))
      {
        comp_flow_operator_ = std::make_unique<CompressibleFlowOperatorExplicit<dim, number>>(
          comp_flow_data_, scratch_data_, solution_history_, comp_flow_dof_idx, comp_flow_quad_idx);
      }
    else if (time_integrator_scheme_is_implicit(comp_flow_data_.time_integrator.integrator_type))
      {
        comp_flow_operator_ = std::make_unique<CompressibleFlowOperatorImplicit<dim, number>>(
          comp_flow_data_, scratch_data_, solution_history_, comp_flow_dof_idx, comp_flow_quad_idx);
      }
    else
      AssertThrow(false,
                  dealii::ExcMessage(
                    "The provided time integration scheme '" +
                    std::to_string(comp_flow_data_.time_integrator.integrator_type) +
                    "' is not supported!"));
  }


  template class CompressibleFlowOperation<1, double>;
  template class CompressibleFlowOperation<2, double>;
  template class CompressibleFlowOperation<3, double>;

} // namespace MeltPoolDG::Flow
