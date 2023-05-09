#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/radiative_transport/rte_operation.hpp>
#include <meltpooldg/radiative_transport/rte_problem.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>



namespace MeltPoolDG::RadiativeTransport
{
  using namespace dealii;

  template <int dim>
  RadiativeTransportOperation<dim>::RadiativeTransportOperation(
    const ScratchData<dim> &              scratch_data_in,
    const RadiativeTransportData<double> &rte_data_in,
    const VectorType &                    heaviside_in,
    const unsigned int                    rte_dof_idx_in,
    const unsigned int                    rte_hanging_nodes_dof_idx_in,
    const unsigned int                    rte_quad_idx_in,
    const unsigned int                    hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , rte_data(rte_data_in)
    , heaviside(heaviside_in)
    , rte_dof_idx(rte_dof_idx_in)
    , rte_hanging_nodes_dof_idx(rte_hanging_nodes_dof_idx_in)
    , rte_quad_idx(rte_quad_idx_in)
    , hs_dof_idx(hs_dof_idx_in)
  {
    rte_operator = std::make_unique<RadiativeTransportOperator<dim, double>>(
      scratch_data, rte_data, intensity, heaviside, rte_dof_idx, rte_quad_idx, hs_dof_idx);

    // matrix-based simulation is not supported
    AssertThrow(rte_data.linear_solver.do_matrix_free, ExcNotImplemented());

    /*
     * setup preconditioner for matrix-free computation
     */
    if (rte_data.linear_solver.do_matrix_free)
      {
        rte_operator->reinit();
        preconditioner_matrixfree = std::make_shared<
          Preconditioner::PreconditionerMatrixFreeGeneric<dim, OperatorBase<dim, double>>>(
          scratch_data, rte_dof_idx, rte_data.linear_solver.preconditioner_type, *rte_operator);
      }


    reinit();
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::reinit()
  {
    {
      ScopedName sc("rte::n_dofs");
      DoFMonitor::add_n_dofs(sc, scratch_data.get_dof_handler(rte_dof_idx).n_dofs());
    }

    scratch_data.initialize_dof_vector(intensity, rte_dof_idx);
    scratch_data.initialize_dof_vector(rhs, rte_dof_idx);

    if (rte_data.linear_solver.do_matrix_free)
      preconditioner_matrixfree->reinit();
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::distribute_constraints()
  {
    scratch_data.get_constraint(rte_dof_idx).distribute(intensity);
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::solve()
  {
    ScopedName         sc("rte::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    heaviside.update_ghost_values();

    unsigned int iter = 0;
    intensity         = 0;

    if (rte_data.linear_solver.do_matrix_free)
      {
        AssertThrow(preconditioner_matrixfree, ExcNotImplemented());

        // apply dirichlet boundary values
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree(*rte_operator,
                                                                         rhs,
                                                                         heaviside,
                                                                         scratch_data,
                                                                         rte_dof_idx,
                                                                         rte_hanging_nodes_dof_idx,
                                                                         true /*zero out rhs*/);

        if (rte_data.linear_solver.preconditioner_type == PreconditionerType::Diagonal)
          {
            diag_preconditioner_matrixfree =
              preconditioner_matrixfree->compute_diagonal_preconditioner();
            iter = LinearSolver::solve<VectorType>(*rte_operator,
                                                   intensity,
                                                   rhs,
                                                   rte_data.linear_solver,
                                                   *diag_preconditioner_matrixfree);
          }
        else
          {
            trilinos_preconditioner_matrixfree =
              preconditioner_matrixfree->compute_trilinos_preconditioner();
            iter = LinearSolver::solve<VectorType>(*rte_operator,
                                                   intensity,
                                                   rhs,
                                                   rte_data.linear_solver,
                                                   *trilinos_preconditioner_matrixfree);
          }
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }

    heaviside.zero_out_ghost_values();

    scratch_data.get_constraint(rte_dof_idx).distribute(intensity);

    const ConditionalOStream &pcout = scratch_data.get_pcout(rte_data.verbosity_level);
    Journal::print_formatted_norm(
      pcout,
      [&]() -> double {
        return VectorTools::compute_L2_norm<dim>(intensity,
                                                 scratch_data,
                                                 rte_dof_idx,
                                                 rte_quad_idx);
      },
      "intensity",
      "RTE",
      11 /*precision*/
    );

    IterationMonitor::add_linear_iterations(sc, iter);
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  RadiativeTransportOperation<dim>::get_intensity() const
  {
    return intensity;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  RadiativeTransportOperation<dim>::get_intensity()
  {
    return intensity;
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&intensity);
  }

  template <int dim>
  void
  RadiativeTransportOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(rte_dof_idx), intensity, "intensity");
    data_out.add_data_vector(scratch_data.get_dof_handler(rte_dof_idx), rhs, "rte_rhs");
  }

  template class RadiativeTransportOperation<1>;
  template class RadiativeTransportOperation<2>;
  template class RadiativeTransportOperation<3>;
} // namespace MeltPoolDG::RadiativeTransport
