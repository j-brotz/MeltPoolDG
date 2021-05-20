#include <meltpooldg/curvature/curvature_operation.hpp>
//

namespace MeltPoolDG::Curvature
{
  template <int dim>
  void
  CurvatureOperation<dim>::initialize(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &                     data_in,
    const unsigned int                             curv_dof_idx_in,
    const unsigned int                             curv_quad_idx_in,
    const unsigned int                             normal_dof_idx_in,
    const unsigned int                             ls_dof_idx_in)
  {
    scratch_data   = scratch_data_in;
    curv_dof_idx   = curv_dof_idx_in;
    curv_quad_idx  = curv_quad_idx_in;
    normal_dof_idx = normal_dof_idx_in;
    /*
     *  initialize curvature data
     */
    curvature_data = data_in.curv;
    /*
     *    initialize normal_vector_operation for computing the normal vector to the given
     *    scalar function for which the curvature should be calculated.
     */
    normal_vector_operation.initialize(
      scratch_data, data_in, normal_dof_idx_in, curv_quad_idx, ls_dof_idx_in);
    /*
     *  initialize the operator (input-dependent: matrix-based or matrix-free)
     */
    create_operator();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::solve(const VectorType &solution_levelset)
  {
    /*
     *    compute and solve the normal vector field for the given level set
     */
    normal_vector_operation.solve(solution_levelset);

    VectorType rhs;

    scratch_data->initialize_dof_vector(rhs, curv_dof_idx);
    scratch_data->initialize_dof_vector(solution_curvature, curv_dof_idx);
    int iter = 0;

    if (curvature_data.do_matrix_free)
      {
        curvature_operator->create_rhs(rhs, normal_vector_operation.get_solution_normal_vector());
        iter =
          LinearSolve::solve<VectorType,
                             SolverCG<VectorType>,
                             OperatorBase<double, VectorType, BlockVectorType>>(*curvature_operator,
                                                                                solution_curvature,
                                                                                rhs);
      }
    else
      {
        curvature_operator->assemble_matrixbased(
          normal_vector_operation.get_solution_normal_vector(),
          curvature_operator->system_matrix,
          rhs);

        iter = LinearSolve::solve<VectorType, SolverCG<VectorType>, SparseMatrixType>(
          curvature_operator->system_matrix, solution_curvature, rhs);
      }

    scratch_data->get_constraint(curv_dof_idx).distribute(solution_curvature);

    const int                 verbosity_l2_norm = dim > 1 ? 0 : 1;
    const ConditionalOStream &pcout             = scratch_data->get_pcout(verbosity_l2_norm);
    scratch_data->get_pcout(1) << "| curvature:         i=" << iter << " \t";
    pcout << "|k| = " << std::setprecision(11) << std::setw(15) << std::left
          << VectorTools::compute_L2_norm<dim>(solution_curvature,
                                               *scratch_data,
                                               curv_dof_idx,
                                               curv_quad_idx)
          << " ";
    pcout << std::endl;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  CurvatureOperation<dim>::get_curvature() const
  {
    return solution_curvature;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  CurvatureOperation<dim>::get_curvature()
  {
    return solution_curvature;
  }

  template <int dim>
  const LinearAlgebra::distributed::BlockVector<double> &
  CurvatureOperation<dim>::get_normal_vector() const
  {
    return normal_vector_operation.get_solution_normal_vector();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::reinit()
  {
    if (!curvature_data.do_matrix_free)
      curvature_operator->initialize_matrix_based<dim>(*scratch_data);
    normal_vector_operation.reinit();
  }

  template <int dim>
  void
  CurvatureOperation<dim>::create_operator()
  {
    const double damping_parameter =
      scratch_data->get_min_cell_size(curv_dof_idx) * curvature_data.damping_scale_factor;
    curvature_operator = std::make_unique<CurvatureOperator<dim>>(
      *scratch_data, damping_parameter, curv_dof_idx, curv_quad_idx, normal_dof_idx);
    /*
     *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */
    if (!curvature_data.do_matrix_free)
      curvature_operator->initialize_matrix_based<dim>(*scratch_data);
  }

  template class CurvatureOperation<1>;
  template class CurvatureOperation<2>;
  template class CurvatureOperation<3>;
} // namespace MeltPoolDG::Curvature
