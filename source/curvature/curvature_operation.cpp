#include <meltpooldg/curvature/curvature_operation.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <any>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  CurvatureOperation<dim>::CurvatureOperation(const ScratchData<dim>         &scratch_data_in,
                                              const CurvatureData<double>    &curvature_data,
                                              const NormalVectorData<double> &normal_vec_data,
                                              const VectorType               &solution_levelset,
                                              const unsigned int              curv_dof_idx_in,
                                              const unsigned int              curv_quad_idx_in,
                                              const unsigned int              normal_dof_idx_in,
                                              const unsigned int              ls_dof_idx_in)
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

  template <int dim>
  void
  CurvatureOperation<dim>::update_normal_vector()
  {
    const ScopedName   sc("curvature::update_normal_vector");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    normal_vector_operation.solve();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::solve()
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
        std::make_unique<Predictor<VectorType, double>>(curvature_data.predictor, solution_history);

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

    const unsigned int        verbosity_l2_norm = dim > 1 ? 0 : 1;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(curvature_data.verbosity_level, verbosity_l2_norm));


    Journal::print_formatted_norm(
      pcout,
      [&]() -> double {
        return VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                              scratch_data,
                                              curv_dof_idx,
                                              curv_quad_idx);
      },
      "curvature",
      "curvature",
      11 /*precision*/
    );

    Journal::print_line(scratch_data.get_pcout(1),
                        "     * CG: i = " + std::to_string(iter),
                        "curvature");

    IterationMonitor::add_linear_iterations(sc, iter);
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  CurvatureOperation<dim>::get_curvature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  CurvatureOperation<dim>::get_curvature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  const LinearAlgebra::distributed::BlockVector<double> &
  CurvatureOperation<dim>::get_normal_vector() const
  {
    return normal_vector_operation.get_solution_normal_vector();
  }

  template <int dim>
  LinearAlgebra::distributed::BlockVector<double> &
  CurvatureOperation<dim>::get_normal_vector()
  {
    return normal_vector_operation.get_solution_normal_vector();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::reinit()
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
        curvature_operator->compute_system_matrix_and_rhs(
          normal_vector_operation.get_solution_normal_vector(), rhs);
        preconditioner.update(&curvature_operator->get_system_matrix());
      }
  }

  template <int dim>
  void
  CurvatureOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    normal_vector_operation.attach_vectors(vectors);

    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim>
  void
  CurvatureOperation<dim>::create_operator(const VectorType &solution_levelset)
  {
    curvature_operator = std::make_shared<CurvatureOperator<dim>>(scratch_data,
                                                                  curvature_data,
                                                                  curv_dof_idx,
                                                                  curv_quad_idx,
                                                                  normal_dof_idx,
                                                                  ls_dof_idx,
                                                                  &solution_levelset);


    preconditioner = make_preconditioner<dim, CurvatureOperator<dim>, VectorType>(
      curvature_data.linear_solver.preconditioner_type,
      curvature_operator.get(),
      curvature_data.linear_solver.do_matrix_free);
  }

  template class CurvatureOperation<1>;
  template class CurvatureOperation<2>;
  template class CurvatureOperation<3>;
} // namespace MeltPoolDG::LevelSet
