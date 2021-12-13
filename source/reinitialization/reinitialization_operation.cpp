#include <meltpooldg/linear_algebra/preconditioner_trilinos_factory.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::Reinitialization
{
  template <int dim>
  void
  ReinitializationOperation<dim>::initialize(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &                     data_in,
    const unsigned int                             reinit_dof_idx_in,
    const unsigned int                             reinit_quad_idx_in,
    const unsigned int                             ls_dof_idx_in,
    const unsigned int                             normal_dof_idx_in)
  {
    scratch_data    = scratch_data_in;
    reinit_dof_idx  = reinit_dof_idx_in;
    reinit_quad_idx = reinit_quad_idx_in;
    ls_dof_idx      = ls_dof_idx_in;
    normal_dof_idx  = normal_dof_idx_in;
    scratch_data->initialize_dof_vector(solution_level_set, ls_dof_idx_in);
    /*
     *    initialize the (local) parameters of the reinitialization
     *    from the global user-defined parameters
     */
    set_reinitialization_parameters(data_in);
    /*
     *    initialize normal_vector_field
     */
    AssertThrow(data_in.normal_vec.do_matrix_free == data_in.reinit.solver.do_matrix_free,
                ExcMessage("For the reinitialization problem both the "
                           " normal vector and the reinitialization operation have to be "
                           " computed either matrix-based or matrix-free."));

    if (data_in.normal_vec.implementation == "meltpooldg")
      {
        normal_vector_operation = std::make_shared<NormalVector::NormalVectorOperation<dim>>();

        normal_vector_operation->initialize(
          scratch_data_in, data_in, normal_dof_idx, reinit_quad_idx, ls_dof_idx);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (data_in.normal_vec.implementation == "adaflo")
      {
        AssertThrow(data_in.normal_vec.do_matrix_free, ExcNotImplemented());

        normal_vector_operation =
          std::make_shared<NormalVector::NormalVectorOperationAdaflo<dim>>(*scratch_data_in,
                                                                           ls_dof_idx_in,
                                                                           normal_dof_idx,
                                                                           reinit_quad_idx,
                                                                           solution_level_set,
                                                                           data_in);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *   create reinitialization operator. This class supports matrix-based
     *   and matrix-free computation.
     */
    create_operator();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::reinit()
  {
    scratch_data->initialize_dof_vector(solution_level_set, ls_dof_idx);
    update_operator();
    normal_vector_operation->reinit();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::set_initial_condition(const VectorType &solution_level_set_in)
  {
    /*
     *    copy the given solution into the member variable
     */
    scratch_data->initialize_dof_vector(solution_level_set, ls_dof_idx);
    solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
    solution_level_set.update_ghost_values();
    /*
     *    update the normal vector field corresponding to the given solution of the
     *    level set; the normal vector field is called by reference within the
     *    operator class
     */
    normal_vector_operation->solve(solution_level_set);
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::update_dof_idx(const unsigned int &reinit_dof_idx_in)
  {
    reinit_dof_idx = reinit_dof_idx_in;
    create_operator();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::solve(const double d_tau)
  {
    /**
     * update the distributed sparsity pattern for matrix-based amr
     */
    VectorType src, rhs;

    scratch_data->initialize_dof_vector(src, reinit_dof_idx);
    scratch_data->initialize_dof_vector(rhs, reinit_dof_idx);

    reinit_operator->set_time_increment(d_tau);

    int iter = 0;

    if (reinit_data.solver.do_matrix_free)
      {
        reinit_operator->create_rhs(rhs, solution_level_set);
        iter = LinearSolve::solve<VectorType, SolverCG<VectorType>, OperatorBase<dim, double>>(
          *reinit_operator, src, rhs);
      }
    else
      {
        reinit_operator->system_matrix.reinit(reinit_operator->dsp);
        reinit_operator->assemble_matrixbased(solution_level_set,
                                              reinit_operator->system_matrix,
                                              rhs);

        if (reinit_data.solver.solver_type == SolverType::CG)
          {
            auto preconditioner =
              Preconditioner::get_preconditioner_trilinos(reinit_operator->system_matrix,
                                                          reinit_data.solver.preconditioner_type);
            iter = LinearSolve::solve<VectorType,
                                      SolverCG<VectorType>,
                                      SparseMatrixType,
                                      TrilinosWrappers::PreconditionBase>(
              reinit_operator->system_matrix,
              src,
              rhs,
              reinit_data.solver.rel_tolerance,
              reinit_data.solver.max_iterations,
              *preconditioner);
          }
        else if (reinit_data.solver.solver_type == SolverType::GMRES)
          {
            auto preconditioner =
              Preconditioner::get_preconditioner_trilinos(reinit_operator->system_matrix,
                                                          reinit_data.solver.preconditioner_type);
            iter = LinearSolve::solve<VectorType,
                                      SolverGMRES<VectorType>,
                                      SparseMatrixType,
                                      TrilinosWrappers::PreconditionBase>(
              reinit_operator->system_matrix,
              src,
              rhs,
              reinit_data.solver.rel_tolerance,
              reinit_data.solver.max_iterations,
              *preconditioner);
          }

        Journal::print_formatted_norm(scratch_data->get_pcout(0),
                                      reinit_operator->system_matrix.frobenius_norm(),
                                      "matrix",
                                      "reinitialization",
                                      15 /*precision*/,
                                      "F");
      }
    scratch_data->get_constraint(reinit_dof_idx).distribute(src);
    solution_level_set.zero_out_ghost_values();

    solution_level_set += src;

    solution_level_set.update_ghost_values();

    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  MeltPoolDG::VectorTools::compute_L2_norm<dim>(
                                    rhs, *scratch_data, reinit_dof_idx, reinit_quad_idx),
                                  "RHS",
                                  "reinitialization",
                                  15 /*precision*/
    );
    Journal::print_formatted_norm(
      scratch_data->get_pcout(0),
      VectorTools::compute_L2_norm<dim>(src, *scratch_data, reinit_dof_idx, reinit_quad_idx),
      "delta phi",
      "reinitialization",
      15 /*precision*/
    );
    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  src.linfty_norm(),
                                  "delta phi",
                                  "reinitialization",
                                  15 /*precision*/,
                                  "∞ ",
                                  2);
    Journal::print_formatted_norm(scratch_data->get_pcout(1),
                                  VectorTools::compute_L2_norm<dim>(solution_level_set,
                                                                    *scratch_data,
                                                                    reinit_dof_idx,
                                                                    reinit_quad_idx),
                                  "phi",
                                  "reinitialization",
                                  15 /*precision*/
    );

    Journal::print_line(scratch_data->get_pcout(1),
                        "     * CG: i = " + std::to_string(iter),
                        "reinitialization");
  }

  template <int dim>
  const ReinitializationOperation<dim>::BlockVectorType &
  ReinitializationOperation<dim>::get_normal_vector() const
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim>
  const VectorType &
  ReinitializationOperation<dim>::get_level_set() const
  {
    return solution_level_set;
  }

  template <int dim>
  VectorType &
  ReinitializationOperation<dim>::get_level_set()
  {
    return solution_level_set;
  }

  template <int dim>
  ReinitializationOperation<dim>::BlockVectorType &
  ReinitializationOperation<dim>::get_normal_vector()
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&solution_level_set);
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data->get_dof_handler(reinit_dof_idx), get_level_set(), "psi");

    //@todo: attach_output_vectors from normal_vector_operation
    get_normal_vector().update_ghost_values();
    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data->get_dof_handler(reinit_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::set_reinitialization_parameters(const Parameters<double> &data_in)
  {
    reinit_data = data_in.reinit;
    reinit_data.scale_factor_epsilon /= data_in.ls.n_subdivisions;
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::create_operator()
  {
    if (reinit_data.modeltype == "olsson2007")
      {
        reinit_operator = std::make_unique<OlssonOperator<dim, double>>(
          *scratch_data,
          normal_vector_operation->get_solution_normal_vector(),
          reinit_data.constant_epsilon,
          reinit_data.scale_factor_epsilon,
          reinit_dof_idx,
          reinit_quad_idx,
          ls_dof_idx,
          normal_dof_idx);
      }
    /*
     * add your desired operators here
     *
     * else if (reinit_data.reinitmodel == "my_model")
     *    ....
     */
    else
      AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."))
        /*
         *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
         *  apply it to the system matrix. This functionality is part of the OperatorBase class.
         */

        if (!reinit_data.solver.do_matrix_free)
          reinit_operator->initialize_matrix_based(*scratch_data);
  }

  template <int dim>
  void
  ReinitializationOperation<dim>::update_operator()
  {
    if (!reinit_data.solver.do_matrix_free)
      reinit_operator->initialize_matrix_based(*scratch_data);
  }

  template class ReinitializationOperation<1>;
  template class ReinitializationOperation<2>;
  template class ReinitializationOperation<3>;
} // namespace MeltPoolDG::Reinitialization
