#include <meltpooldg/interface/operator_base.hpp>

namespace MeltPoolDG
{
  template <int dim, typename number, typename DoFVectorType, typename SrcRhsVectorType>
  void
  OperatorBase<dim, number, DoFVectorType, SrcRhsVectorType>::set_time_increment(const double dt)
  {
    d_tau     = dt;
    d_tau_inv = 1. / d_tau;
  }

  template <int dim, typename number, typename DoFVectorType, typename SrcRhsVectorType>
  void
  OperatorBase<dim, number, DoFVectorType, SrcRhsVectorType>::reset_indices(
    const unsigned int dof_idx_in,
    const unsigned int quad_idx_in)
  {
    this->dof_idx  = dof_idx_in;
    this->quad_idx = quad_idx_in;
  }

  template <int dim, typename number, typename DoFVectorType, typename SrcRhsVectorType>
  void
  OperatorBase<dim, number, DoFVectorType, SrcRhsVectorType>::initialize_matrix_based(
    const ScratchData<dim> &scratch_data)
  {
    const MPI_Comm mpi_communicator = scratch_data.get_mpi_comm(this->dof_idx);
    dsp.reinit(scratch_data.get_locally_owned_dofs(this->dof_idx),
               scratch_data.get_locally_owned_dofs(this->dof_idx),
               scratch_data.get_locally_relevant_dofs(this->dof_idx),
               mpi_communicator);

    DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(this->dof_idx),
                                    this->dsp,
                                    scratch_data.get_constraint(this->dof_idx),
                                    true,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));
    this->dsp.compress();

    this->system_matrix.reinit(dsp);
  }

  template <int dim, typename number, typename DoFVectorType, typename SrcRhsVectorType>
  void
  OperatorBase<dim, number, DoFVectorType, SrcRhsVectorType>::create_rhs_and_apply_dirichlet_mf(
    DoFVectorType &         rhs,
    const SrcRhsVectorType &src,
    const ScratchData<dim> &scratch_data,
    const unsigned int      dof_idx,
    const unsigned int      dof_no_bc_idx)
  {
    this->reset_indices(dof_no_bc_idx, this->quad_idx);

    DoFVectorType temp_rhs;
    scratch_data.initialize_dof_vector(temp_rhs, dof_no_bc_idx);

    SrcRhsVectorType temp_src;
    scratch_data.initialize_dof_vector(temp_src, dof_no_bc_idx);
    temp_src = src;

    DoFVectorType bc_values;
    scratch_data.initialize_dof_vector(bc_values, dof_no_bc_idx);
    scratch_data.get_constraint(dof_idx).distribute(bc_values);
    /*
     * perform matrix-vector multiplication (with unconstrained system and constrained set in
     * Vector)
     */
    this->vmult(temp_rhs, bc_values);
    /*
     * Modify right-hand side
     */
    temp_rhs *= -1.0;
    this->create_rhs(temp_rhs, temp_src);
    /*
     * Clear constrained values
     */
    rhs = temp_rhs;
    scratch_data.get_constraint(dof_idx).set_zero(rhs);
    this->reset_indices(dof_idx, this->quad_idx);
  }

  template <int dim, typename number, typename DoFVectorType, typename SrcRhsVectorType>
  const OperatorBase<dim, number, DoFVectorType, SrcRhsVectorType>::SparseMatrixType &
  OperatorBase<dim, number, DoFVectorType, SrcRhsVectorType>::get_system_matrix() const
  {
    return this->system_matrix;
  }

  template class OperatorBase<1, double>;
  template class OperatorBase<2, double>;
  template class OperatorBase<3, double>;

  template class OperatorBase<1,
                              double,
                              LinearAlgebra::distributed::Vector<double>,
                              LinearAlgebra::distributed::BlockVector<double>>;
  template class OperatorBase<2,
                              double,
                              LinearAlgebra::distributed::Vector<double>,
                              LinearAlgebra::distributed::BlockVector<double>>;
  template class OperatorBase<3,
                              double,
                              LinearAlgebra::distributed::Vector<double>,
                              LinearAlgebra::distributed::BlockVector<double>>;
  template class OperatorBase<1,
                              double,
                              LinearAlgebra::distributed::BlockVector<double>,
                              LinearAlgebra::distributed::Vector<double>>;
  template class OperatorBase<2,
                              double,
                              LinearAlgebra::distributed::BlockVector<double>,
                              LinearAlgebra::distributed::Vector<double>>;
  template class OperatorBase<3,
                              double,
                              LinearAlgebra::distributed::BlockVector<double>,
                              LinearAlgebra::distributed::Vector<double>>;
  template class OperatorBase<1,
                              double,
                              LinearAlgebra::distributed::BlockVector<double>,
                              LinearAlgebra::distributed::BlockVector<double>>;
  template class OperatorBase<2,
                              double,
                              LinearAlgebra::distributed::BlockVector<double>,
                              LinearAlgebra::distributed::BlockVector<double>>;
  template class OperatorBase<3,
                              double,
                              LinearAlgebra::distributed::BlockVector<double>,
                              LinearAlgebra::distributed::BlockVector<double>>;
} // namespace MeltPoolDG
