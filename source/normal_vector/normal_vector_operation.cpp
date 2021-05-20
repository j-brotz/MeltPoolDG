#include <meltpooldg/normal_vector/normal_vector_operation.hpp>

namespace MeltPoolDG::NormalVector
{
  template <int dim>
  void
  NormalVectorOperation<dim>::initialize(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &                     data_in,
    const unsigned int                             normal_dof_idx_in,
    const unsigned int                             normal_quad_idx_in,
    const unsigned int                             ls_dof_idx_in)
  {
    scratch_data    = scratch_data_in;
    normal_dof_idx  = normal_dof_idx_in;
    normal_quad_idx = normal_quad_idx_in;
    ls_dof_idx      = ls_dof_idx_in;
    /*
     *  initialize normal vector data
     */
    normal_vector_data = data_in.normal_vec;
    /*
     *  initialize normal vector operator
     */
    create_operator();
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::reinit()
  {
    if (!normal_vector_data.do_matrix_free)
      normal_vector_operator->initialize_matrix_based<dim>(*scratch_data);
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::solve(const VectorType &solution_levelset_in)
  {
    BlockVectorType rhs;

    scratch_data->initialize_dof_vector(rhs, normal_dof_idx);
    scratch_data->initialize_dof_vector(solution_normal_vector, normal_dof_idx);

    int iter = 0;

    if (normal_vector_data.do_matrix_free)
      {
        normal_vector_operator->create_rhs(rhs, solution_levelset_in);
        iter = LinearSolve::solve<BlockVectorType,
                                  SolverCG<BlockVectorType>,
                                  OperatorBase<double, BlockVectorType, VectorType>>(
          *normal_vector_operator, solution_normal_vector, rhs);
      }
    else
      {
        normal_vector_operator->assemble_matrixbased(solution_levelset_in,
                                                     normal_vector_operator->system_matrix,
                                                     rhs);

        for (unsigned int d = 0; d < dim; ++d)
          iter = LinearSolve::solve<VectorType, SolverCG<VectorType>, SparseMatrixType>(
            normal_vector_operator->system_matrix, solution_normal_vector.block(d), rhs.block(d));
      }
    for (unsigned int d = 0; d < dim; ++d)
      scratch_data->get_constraint(normal_dof_idx).distribute(solution_normal_vector.block(d));

    scratch_data->get_pcout(1) << "| normal vector:         i=" << iter;
    const int                 verbosity_l2_norm = dim > 1 ? 0 : 1;
    const ConditionalOStream &pcout             = scratch_data->get_pcout(verbosity_l2_norm);
    pcout << " \t";
    for (unsigned int d = 0; d < dim; ++d)
      pcout << " |n_" << d << "| = " << std::setprecision(11) << std::setw(15) << std::left
            << MeltPoolDG::VectorTools::compute_L2_norm<dim>(solution_normal_vector.block(d),
                                                             *scratch_data,
                                                             normal_dof_idx,
                                                             normal_quad_idx)
            << " ";
    pcout << std::endl;
  }

  template <int dim>
  const NormalVectorOperation<dim>::BlockVectorType &
  NormalVectorOperation<dim>::get_solution_normal_vector() const
  {
    return solution_normal_vector;
  }

  template <int dim>
  NormalVectorOperation<dim>::BlockVectorType &
  NormalVectorOperation<dim>::get_solution_normal_vector()
  {
    return solution_normal_vector;
  }

  template <int dim>
  void
  NormalVectorOperation<dim>::create_operator()
  {
    const double damping_parameter = std::pow(scratch_data->get_min_cell_size(normal_dof_idx), 2) *
                                     normal_vector_data.damping_scale_factor;
    normal_vector_operator = std::make_unique<NormalVectorOperator<dim>>(
      *scratch_data, damping_parameter, normal_dof_idx, normal_quad_idx, ls_dof_idx);
    /*
     *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
     *  apply it to the system matrix. This functionality is part of the OperatorBase class.
     */
    if (!normal_vector_data.do_matrix_free)
      normal_vector_operator->initialize_matrix_based<dim>(*scratch_data);
  }


  template class NormalVectorOperation<1>;
  template class NormalVectorOperation<2>;
  template class NormalVectorOperation<3>;
} // namespace MeltPoolDG::NormalVector
