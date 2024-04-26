#include <meltpooldg/linear_algebra/preconditioner_trilinos_factory.hpp>
#include <meltpooldg/reinitialization/reinitialization_DG_operation.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include "../simulations/reinit_circle/reinit_circle_DG.hpp"

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  ReinitializationDGOperation<dim>::ReinitializationDGOperation(
    const ScratchData<dim>             &scratch_data_in,
    const ReinitializationData<double> &reinit_data,
    const TimeIterator<double>         &time_iterator,
    const unsigned int                  reinit_dof_idx_in,
    const unsigned int                  reinit_quad_idx_in,
    const unsigned int                  ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data)
    , time_iterator(time_iterator)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , solution_history(std::max(reinit_data.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
    , reinit_DG_operator(scratch_data_in, reinit_data, reinit_dof_idx_in, reinit_quad_idx_in)
  {
    reinitilization_integration =
      TimeIntegratorConcretization::concretize(reinit_data.time_integration_scheme,
                                               reinit_DG_operator,
                                               scratch_data_in,
                                               reinit_dof_idx_in,
                                               reinit_quad_idx_in,
                                               reinit_data.linear_solver);
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, reinit_dof_idx); });

    scratch_data.initialize_dof_vector(rhs, reinit_dof_idx);

    reinit_DG_operator.reinit();
    reinitilization_integration->reinit();
  }


  template <int dim>
  void
  ReinitializationDGOperation<dim>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    FECellIntegrator<dim, 1, double> phi(scratch_data.get_matrix_free(),
                                         reinit_dof_idx,
                                         reinit_quad_idx);


    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, double> inverse(phi);

    if (solution_history.get_current_solution().has_ghost_elements())
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


    if (solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().update_ghost_values();

    solution_history.commit_old_solutions();
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::update_dof_idx(const unsigned int &reinit_dof_idx_in)
  {
    reinit_dof_idx = reinit_dof_idx_in;
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::init_time_advance()
  {
    if (solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().zero_out_ghost_values();
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::solve()
  {
    ScopedName         sc("reinitialization::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    const bool ls_update_ghosts = !solution_history.get_current_solution().has_ghost_elements();
    if (ls_update_ghosts)
      solution_history.get_current_solution().update_ghost_values();

    init_time_advance();


    if (reinit_data.linear_solver.do_matrix_free)
      {
        if (reinit_data.use_IMEX)
          {
            reinit_DG_operator.apply_IMEX(time_iterator.get_old_time(),
                                          time_iterator.get_current_time_increment(),
                                          solution_history,
                                          rhs);
            solution_history.set_recent_old_solution(solution_history.get_current_solution());
          }

        reinitilization_integration->perform_time_step(time_iterator.get_old_time(),
                                                       time_iterator.get_current_time_increment(),
                                                       solution_history,
                                                       rhs);
        // update ghost values of solution
        if (ls_update_ghosts)
          solution_history.update_ghost_values();

        solution_history.get_recent_old_solution().add(-1.0,
                                                       solution_history.get_current_solution());
        max_change_level_set = solution_history.get_recent_old_solution().linfty_norm();

        solution_history.commit_old_solutions();
      }
  }

  template <int dim>
  double
  ReinitializationDGOperation<dim>::get_max_change_level_set() const
  {
    return max_change_level_set;
  }

  template <int dim>
  const VectorType &
  ReinitializationDGOperation<dim>::get_level_set() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  VectorType &
  ReinitializationDGOperation<dim>::get_level_set()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                             solution_history.get_current_solution(),
                             "psi");

    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx), rhs, "rhs");
  }


  template <int dim>
  void
  ReinitializationDGOperation<dim>::prepare_reinitilization()
  {
    reinit_DG_operator.compute_penalty_parameter();
    reinit_DG_operator.compute_viscosity_value();
    reinit_DG_operator.prepare_operator(solution_history.get_current_solution());
  }

  template <int dim>
  double
  ReinitializationDGOperation<dim>::compute_CFL_based_timestep() const
  {
    const double minimal_vertex_distance = scratch_data.get_min_cell_size();
    const double fe_degree               = (double)reinit_data.fe.degree;

    if (reinit_data.use_IMEX)
      {
        return reinit_data.CFL * minimal_vertex_distance / (fe_degree * fe_degree);
      }
    else
      {
        return reinit_data.CFL / reinit_data.IP_diffusion /
               (std::pow(fe_degree, 2.0) / minimal_vertex_distance +
                reinit_DG_operator.get_viscosity() * std::pow(fe_degree, 4.0) /
                  (minimal_vertex_distance * minimal_vertex_distance));
      }
  }

  template class ReinitializationDGOperation<1>;
  template class ReinitializationDGOperation<2>;
  template class ReinitializationDGOperation<3>;
} // namespace MeltPoolDG::LevelSet
