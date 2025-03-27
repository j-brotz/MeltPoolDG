#include <meltpooldg/level_set/curvature_operation.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <any>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  CurvatureOperation<dim, number>::CurvatureOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const CurvatureData<number>         &curvature_data,
    const NormalVectorData<number>      &normal_vec_data,
    const VectorType                    &solution_levelset,
    const unsigned int                   curv_dof_idx_in,
    const unsigned int                   curv_quad_idx_in,
    const unsigned int                   normal_dof_idx_in,
    const unsigned int                   ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , curvature_data(curvature_data)
    , solution_levelset(solution_levelset)
    , curv_dof_idx(curv_dof_idx_in)
    , curv_quad_idx(curv_quad_idx_in)
    , normal_dof_idx(normal_dof_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , normal_vector_operation(scratch_data,
                              normal_vec_data,
                              solution_levelset,
                              normal_dof_idx_in,
                              curv_quad_idx,
                              ls_dof_idx_in)
    , solution_history(curvature_data.predictor.n_old_solution_vectors)
  {
    if (!curvature_operator)
      create_operator(solution_levelset);
  }

  template <int dim, typename number>
  void
  CurvatureOperation<dim, number>::update_normal_vector()
  {
    const ScopedName   sc("curvature::update_normal_vector");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    normal_vector_operation.solve();
  }

  template <int dim, typename number>
  void
  CurvatureOperation<dim, number>::solve()
  {
    const ScopedName   sc("curvature::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    // compute and solve the normal vector field for the given level set
    normal_vector_operation.solve();

    if (!curvature_data.enable)
      return;

    const bool update_ghosts = !solution_levelset.has_ghost_elements();
    if (update_ghosts)
      solution_levelset.update_ghost_values();

    const bool normal_update_ghosts =
      !normal_vector_operation.get_solution_normal_vector().has_ghost_elements();
    if (normal_update_ghosts)
      normal_vector_operation.get_solution_normal_vector().update_ghost_values();

    // compute predictor
    if (!predictor)
      predictor =
        std::make_unique<Predictor<VectorType, number>>(curvature_data.predictor, solution_history);

    if (curvature_data.linear_solver.do_matrix_free &&
        curvature_data.predictor.type == PredictorType::least_squares_projection)
      {
        curvature_operator->create_rhs(rhs, normal_vector_operation.get_solution_normal_vector());
      }

    predictor->vmult(*curvature_operator, solution_curvature_predictor, rhs);

    // no need to compute curvature in 1d
    if constexpr (dim == 1)
      return;

    unsigned int iter = 0;

    if (curvature_data.linear_solver.do_matrix_free)
      {
        curvature_operator->create_rhs(rhs, normal_vector_operation.get_solution_normal_vector());
        iter = LinearSolver::solve<VectorType>(*curvature_operator,
                                               solution_history.get_current_solution(),
                                               rhs,
                                               curvature_data.linear_solver,
                                               preconditioner,
                                               "curvature_operation");
      }
    else
      {
        curvature_operator->compute_system_matrix_and_rhs(
          normal_vector_operation.get_solution_normal_vector(), rhs);
        iter = LinearSolver::solve<VectorType>(curvature_operator->get_system_matrix(),
                                               solution_history.get_current_solution(),
                                               rhs,
                                               curvature_data.linear_solver,
                                               preconditioner,
                                               "curvature_operation");
      }

    if (update_ghosts)
      solution_levelset.zero_out_ghost_values();

    if (normal_update_ghosts)
      normal_vector_operation.get_solution_normal_vector().zero_out_ghost_values();

    scratch_data.get_constraint(curv_dof_idx).distribute(solution_history.get_current_solution());

    // update ghost values of solution
    solution_history.get_current_solution().update_ghost_values();

    const int                 verbosity_l2_norm = dim > 1 ? 1 : 2;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(curvature_data.verbosity_level, verbosity_l2_norm));


    Journal::print_formatted_norm(
      pcout,
      [&]() -> number {
        return VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                              scratch_data,
                                              curv_dof_idx,
                                              curv_quad_idx);
      },
      "curvature",
      "curvature",
      11 /*precision*/
    );

    Journal::print_line(scratch_data.get_pcout(2),
                        "     * CG: i = " + std::to_string(iter),
                        "curvature");

    IterationMonitor::add_linear_iterations(sc, iter);
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  CurvatureOperation<dim, number>::get_curvature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  CurvatureOperation<dim, number>::get_curvature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::BlockVector<number> &
  CurvatureOperation<dim, number>::get_normal_vector() const
  {
    return normal_vector_operation.get_solution_normal_vector();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::BlockVector<number> &
  CurvatureOperation<dim, number>::get_normal_vector()
  {
    return normal_vector_operation.get_solution_normal_vector();
  }

  template <int dim, typename number>
  void
  CurvatureOperation<dim, number>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, curv_dof_idx); });

    scratch_data.initialize_dof_vector(rhs, curv_dof_idx);
    scratch_data.initialize_dof_vector(solution_curvature_predictor, curv_dof_idx);

    if (curvature_operator)
      curvature_operator->reinit();

    preconditioner.reinit(scratch_data, curv_dof_idx);
    normal_vector_operation.reinit();
    if (curvature_data.linear_solver.do_matrix_free)
      preconditioner.update();
    else
      {
        bool is_ghosted = false;
        if ((is_ghosted =
               not normal_vector_operation.get_solution_normal_vector().has_ghost_elements()))
          normal_vector_operation.get_solution_normal_vector().update_ghost_values();
        curvature_operator->compute_system_matrix_and_rhs(
          normal_vector_operation.get_solution_normal_vector(), rhs);
        if (not is_ghosted)
          normal_vector_operation.get_solution_normal_vector().zero_out_ghost_values();
        preconditioner.update(&curvature_operator->get_system_matrix());
      }
  }

  template <int dim, typename number>
  void
  CurvatureOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    normal_vector_operation.attach_vectors(vectors);

    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim, typename number>
  void
  CurvatureOperation<dim, number>::create_operator(const VectorType &solution_levelset)
  {
    curvature_operator = std::make_shared<CurvatureOperator<dim, number>>(scratch_data,
                                                                          curvature_data,
                                                                          curv_dof_idx,
                                                                          curv_quad_idx,
                                                                          normal_dof_idx,
                                                                          ls_dof_idx,
                                                                          &solution_levelset);


    preconditioner = make_preconditioner<dim, CurvatureOperator<dim, number>, VectorType>(
      curvature_data.linear_solver.preconditioner_type,
      curvature_operator.get(),
      curvature_data.linear_solver.do_matrix_free);
  }

  template class CurvatureOperation<1, double>;
  template class CurvatureOperation<2, double>;
  template class CurvatureOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
