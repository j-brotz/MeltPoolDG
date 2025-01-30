#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/advection_diffusion/advection_DG_operation.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  AdvectionDGOperation<dim>::AdvectionDGOperation(
    const ScratchData<dim>                                &scratch_data_in,
    const AdvectionDiffusionData<double>                  &advec_diff_data_in,
    const TimeIterator<double>                            &time_iterator,
    VectorType                                            &advection_velocity,
    const unsigned int                                     advec_diff_dof_idx_in,
    const unsigned int                                     advec_diff_quad_idx_in,
    const unsigned int                                     velocity_dof_idx_in,
    const std::shared_ptr<BoundaryConditionManager<dim>> &&boundary_conditions_in,
    std::shared_ptr<dealii::Function<dim>>               &&advection_field_in,
    bool const                                             enable_analytical_velocity_update_in)
    : scratch_data(scratch_data_in)
    , time_iterator(time_iterator)
    , advection_velocity(advection_velocity)
    , advec_diff_dof_idx(advec_diff_dof_idx_in)
    , advec_diff_quad_idx(advec_diff_quad_idx_in)
    , velocity_dof_idx(velocity_dof_idx_in)
    , solution_history(std::max(advec_diff_data_in.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
    , advection_DG_operator(scratch_data_in,
                            advection_velocity,
                            advec_diff_dof_idx_in,
                            advec_diff_quad_idx_in,
                            velocity_dof_idx_in,
                            boundary_conditions_in,
                            advection_field_in,
                            enable_analytical_velocity_update_in)

  {
    this->advec_diff_data = advec_diff_data_in;

    advection_integration = std::shared_ptr<TimeIntegratorBase<double, AdvectionDGOperator<dim>>>(
      time_integrator_factory<double, AdvectionDGOperator<dim>>(
        advec_diff_data_in.time_integrator_data,
        advec_diff_data_in.linear_solver,
        scratch_data.get_timer()));
  }

  template <int dim>
  void
  AdvectionDGOperation<dim>::set_initial_condition(const Function<dim> &initial_field_function)
  {
    FECellIntegrator<dim, 1, double> phi(scratch_data.get_matrix_free(),
                                         advec_diff_dof_idx,
                                         advec_diff_quad_idx);


    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, double> inverse(phi);

    solution_history.get_current_solution().zero_out_ghost_values();

    for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            // Workaround for the Vectorized Array
            dealii::VectorizedArray<double> helper;
            for (unsigned int i = 0; i < VectorizedArray<double>::size(); i++)
              {
                if constexpr (dim == 1)
                  {
                    Point<dim> p(phi.quadrature_point(q)[0][i]);
                    helper[i] = initial_field_function.value(p);
                  }

                if (dim == 2)
                  {
                    Point<dim> p(phi.quadrature_point(q)[0][i], phi.quadrature_point(q)[1][i]);
                    helper[i] = initial_field_function.value(p);
                  }
                if (dim == 3)
                  {
                    Point<dim> p(phi.quadrature_point(q)[0][i],
                                 phi.quadrature_point(q)[1][i],
                                 phi.quadrature_point(q)[2][i]);
                    helper[i] = initial_field_function.value(p);
                  }
              }
            phi.submit_dof_value(helper, q);
          }
        inverse.transform_from_q_points_to_basis(1, phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(solution_history.get_current_solution());
      }

    solution_history.set_recent_old_solution(solution_history.get_current_solution());

    solution_history.update_ghost_values();
  }

  template <int dim>
  void
  AdvectionDGOperation<dim>::set_initial_condition(const VectorType &solution_level_set_in)
  {
    // copy the given solution into the member variable
    solution_history.get_current_solution().zero_out_ghost_values();
    solution_history.get_current_solution().copy_locally_owned_data_from(solution_level_set_in);

    solution_history.set_recent_old_solution(solution_history.get_current_solution());

    solution_history.update_ghost_values();
  }

  template <int dim>
  void
  AdvectionDGOperation<dim>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, advec_diff_dof_idx); });

    scratch_data.initialize_dof_vector(rhs, advec_diff_dof_idx);
    scratch_data.initialize_dof_vector(user_rhs, advec_diff_dof_idx);

    advection_integration->reinit(solution_history);
    advection_integration->set_monitoring_vector(user_rhs);
  }


  template <int dim>
  void
  AdvectionDGOperation<dim>::init_time_advance()
  {
    const bool solution_update_ghosts =
      !solution_history.get_current_solution().has_ghost_elements();
    if (solution_update_ghosts)
      solution_history.get_current_solution().update_ghost_values();

    this->ready_for_time_advance = true;
  }

  template <int dim>
  void
  AdvectionDGOperation<dim>::solve(const bool do_finish_time_step)
  {
    ScopedName         sc("advection_diffusion::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    if (!this->ready_for_time_advance)
      init_time_advance();

    const bool velocity_update_ghosts = !advection_velocity.has_ghost_elements();
    if (velocity_update_ghosts)
      advection_velocity.update_ghost_values();

    const bool solution_update_ghosts =
      !solution_history.get_recent_old_solution().has_ghost_elements();
    if (solution_update_ghosts)
      solution_history.get_recent_old_solution().update_ghost_values();

    rhs = user_rhs;

    // type alias for the pre- and post-processing functions
    std::function<void(double, VectorType &, const VectorType &)> pre_processing;

    if (time_integrator_scheme_is_explicit(advection_integration->get_integrator_type()))
      pre_processing =
        [&](double time, VectorType &dst, const VectorType &current_solution) -> void {
        advection_DG_operator.set_field_functions(time);
        advection_DG_operator.apply_dirichlet_boundary_operator(time, dst, current_solution);
      };

    advection_integration->perform_time_step(advection_DG_operator,
                                             time_iterator.get_current_time(),
                                             time_iterator.get_current_time_increment(),
                                             solution_history,
                                             pre_processing);

    solution_history.commit_old_solutions();

    Journal::print_formatted_norm(
      scratch_data.get_pcout(0),
      [&]() -> double {
        return MeltPoolDG::VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                                          scratch_data,
                                                          advec_diff_dof_idx,
                                                          advec_diff_quad_idx);
      },
      "advected field",
      "advection_diffusion",
      10 /*precision*/
    );

    if (solution_update_ghosts)
      solution_history.get_recent_old_solution().zero_out_ghost_values();

    if (velocity_update_ghosts)
      advection_velocity.zero_out_ghost_values();

    if (do_finish_time_step)
      {
        this->finish_time_advance();
      }

    // update ghost values of solution
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDGOperation<dim>::get_advected_field() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDGOperation<dim>::get_advected_field()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDGOperation<dim>::get_advected_field_old() const
  {
    return solution_history.get_recent_old_solution();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDGOperation<dim>::get_advected_field_old()
  {
    return solution_history.get_recent_old_solution();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDGOperation<dim>::get_user_rhs()
  {
    return user_rhs;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDGOperation<dim>::get_user_rhs() const
  {
    return user_rhs;
  }

  template <int dim>
  void
  AdvectionDGOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim>
  void
  AdvectionDGOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(advec_diff_dof_idx),
                             solution_history.get_current_solution(),
                             "advected_field");
    data_out.add_data_vector(scratch_data.get_dof_handler(advec_diff_dof_idx),
                             user_rhs,
                             "advec_diff_user_rhs");
  }


  template class AdvectionDGOperation<1>;
  template class AdvectionDGOperation<2>;
  template class AdvectionDGOperation<3>;
} // namespace MeltPoolDG::LevelSet
