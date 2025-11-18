#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Constraints
{
  /**
   * Check whether @param constraints are consistent in parallel with the
   * @param dof_handler.
   */
  template <int dim, int spacedim, typename number>
  void
  check_constraints([[maybe_unused]] const dealii::DoFHandler<dim, spacedim> &dof_handler,
                    [[maybe_unused]] const dealii::AffineConstraints<number> &constraints)
  {
#ifndef DEBUG
    return;
#endif

    dealii::IndexSet locally_active_dofs =
      dealii::DoFTools::extract_locally_active_dofs(dof_handler);

    Assert(constraints.is_consistent_in_parallel(
             dealii::Utilities::MPI::all_gather(dof_handler.get_mpi_communicator(),
                                                dof_handler.locally_owned_dofs()),
             locally_active_dofs,
             dof_handler.get_mpi_communicator()),
           dealii::ExcInternalError());
  }

  /**
   * Fill AffineConstraints according to given Dirichlet boundary conditions
   * @param bc_data and assign it to a given @param dof_idx inside @param scratch_data.
   *
   * Note that the AffineConstraints object must be reinited with the locally relevant dof
   * index set before calling this function.
   *
   * If @param set_inhomogeneities is true, inhomogeneities are considered else
   * homogeneous DBC are assumed.
   *
   * If @param close is true, the AffineConstraints object is closed after filling it.
   * In Debug, the constraints are also checked with check_constraints().
   */
  template <int dim, typename number>
  void
  fill_DBC(
    ScratchData<dim, dim, number> &scratch_data,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> &bc_data,
    const unsigned int                                                                  dof_idx,
    const bool set_inhomogeneities = true,
    const bool close               = true)
  {
    if (!bc_data.empty())
      {
        for (const auto &bc : bc_data)
          {
            if (set_inhomogeneities)
              dealii::VectorTools::interpolate_boundary_values<dim>(
                scratch_data.get_mapping(),
                scratch_data.get_dof_handler(dof_idx),
                bc.first,
                *bc.second,
                scratch_data.get_constraint(dof_idx));
            else
              dealii::DoFTools::make_zero_boundary_constraints<dim>(
                scratch_data.get_dof_handler(dof_idx),
                bc.first,
                scratch_data.get_constraint(dof_idx));
          }
      }

    if (close)
      {
        scratch_data.get_constraint(dof_idx).close();
        check_constraints(scratch_data.get_dof_handler(dof_idx),
                          scratch_data.get_constraint(dof_idx));
      }
  }

  /**
   * Reinit and fill AffineConstraints according to given Dirichlet boundary conditions
   * @param bc_data and assign it to a given @param dof_idx inside @param scratch_data.
   *
   * If @param set_inhomogeneities is true, inhomogeneities are considered else
   * homogeneous DBC are assumed.
   *
   * If @param close is true, the AffineConstraints object is closed after filling it.
   * In Debug, the constraints are also checked with check_constraints().
   */
  template <int dim, typename number>
  void
  make_DBC(
    ScratchData<dim, dim, number> &scratch_data,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> &bc_data,
    const unsigned int                                                                  dof_idx,
    const bool set_inhomogeneities = true,
    const bool close               = true)
  {
    scratch_data.get_constraint(dof_idx).reinit(scratch_data.get_locally_owned_dofs(dof_idx),
                                                scratch_data.get_locally_relevant_dofs(dof_idx));

    fill_DBC(scratch_data, bc_data, dof_idx, set_inhomogeneities, close);
  }

  /**
   * Reinit and fill AffineConstraints corresponding to a given @param dof_hanging_nodes_idx
   * inside @param scratch_data, considering hanging nodes.
   *
   * If @param close is true, the AffineConstraints object is closed after filling it.
   * In Debug, the constraints are also checked with check_constraints().
   */
  template <int dim, typename number>
  void
  make_HNC(ScratchData<dim, dim, number> &scratch_data,
           const unsigned int             dof_hanging_nodes_idx,
           const bool                     close = true)
  {
    scratch_data.get_constraint(dof_hanging_nodes_idx)
      .reinit(scratch_data.get_locally_owned_dofs(dof_hanging_nodes_idx),
              scratch_data.get_locally_relevant_dofs(dof_hanging_nodes_idx));
    dealii::DoFTools::make_hanging_node_constraints(
      scratch_data.get_dof_handler(dof_hanging_nodes_idx),
      scratch_data.get_constraint(dof_hanging_nodes_idx));

    if (close)
      {
        scratch_data.get_constraint(dof_hanging_nodes_idx).close();
        check_constraints(scratch_data.get_dof_handler(dof_hanging_nodes_idx),
                          scratch_data.get_constraint(dof_hanging_nodes_idx));
      }
  }

  /**
   * Merge the constraints of the AffineConstraints corresponding to a given @param dof_hanging_nodes_idx
   * into the AffineConstraints corresponding to a given @param dof_idx.
   */
  template <int dim, typename number>
  void
  merge_HNC_into_DBC(ScratchData<dim, dim, number> &scratch_data,
                     const unsigned int             dof_idx,
                     const unsigned int             dof_hanging_nodes_idx)
  {
    scratch_data.get_constraint(dof_idx).merge(
      scratch_data.get_constraint(dof_hanging_nodes_idx),
      dealii::AffineConstraints<number>::MergeConflictBehavior::right_object_wins);
  }

  /**
   * Reinit and fill AffineConstraints according to given Dirichlet boundary conditions
   * @param bc_data and assign it to a given @param dof_idx inside @param scratch_data.
   *
   * If @param set_inhomogeneities is true, inhomogeneities are considered else
   * homogeneous DBC are assumed.
   *
   * Reinit and fill AffineConstraints corresponding to a given @param dof_hanging_nodes_idx
   * inside @param scratch_data, considering hanging nodes.
   *
   * Merge the constraints of the AffineConstraints corresponding to a given @param dof_hanging_nodes_idx
   * into the AffineConstraints corresponding to a given @param dof_idx.
   *
   * The AffineConstraints objects are closed after filling them.
   * In Debug, the constraints are also checked with check_constraints().
   */
  template <int dim, typename number>
  void
  make_DBC_and_HNC_and_merge_HNC_into_DBC(
    ScratchData<dim, dim, number> &scratch_data,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> &bc_data,
    const unsigned int                                                                  dof_idx,
    const unsigned int dof_hanging_nodes_idx,
    const bool         set_inhomogeneities = true)
  {
    make_DBC(scratch_data, bc_data, dof_idx, set_inhomogeneities, true /* close */);
    make_HNC(scratch_data, dof_hanging_nodes_idx, true /* close */);

    merge_HNC_into_DBC<dim, number>(scratch_data, dof_idx, dof_hanging_nodes_idx);

    check_constraints(scratch_data.get_dof_handler(dof_idx), scratch_data.get_constraint(dof_idx));
  }

  /**
   * Insert algebraic periodicity constraints according to @param pbc into the
   * AffineConstraints corresponding to a given @param dof_idx inside @param scratch_data.
   *
   * Note that the AffineConstraints object must not be closed.
   */
  template <int dim, typename number>
  void
  insert_PBC(ScratchData<dim, dim, number>         &scratch_data,
             const PeriodicBoundaryConditions<dim> &pbc,
             const unsigned int                     dof_idx)
  {
    for (const auto &bc : pbc.get_data())
      {
        const auto [id_in, id_out, direction] = bc;
        dealii::DoFTools::make_periodicity_constraints(scratch_data.get_dof_handler(dof_idx),
                                                       id_in,
                                                       id_out,
                                                       direction,
                                                       scratch_data.get_constraint(dof_idx));
      }
  }

  /**
   * Reinit and fill AffineConstraints corresponding to a given @param dof_hanging_nodes_idx
   * inside @param scratch_data, considering hanging nodes.
   *
   * Insert algebraic periodicity constraints according to @param pbc into the
   * AffineConstraints corresponding to a given @param dof_idx inside @param scratch_data.
   *
   * The AffineConstraints objects are closed after filling them.
   * In Debug, the constraints are also checked with check_constraints().
   */
  template <int dim, typename number>
  void
  make_HNC_plus_PBC(ScratchData<dim, dim, number>         &scratch_data,
                    const PeriodicBoundaryConditions<dim> &pbc,
                    const unsigned int                     dof_hanging_nodes_idx)
  {
    make_HNC(scratch_data, dof_hanging_nodes_idx, false /* close */);

    insert_PBC(scratch_data, pbc, dof_hanging_nodes_idx);

    scratch_data.get_constraint(dof_hanging_nodes_idx).close();
    check_constraints(scratch_data.get_dof_handler(dof_hanging_nodes_idx),
                      scratch_data.get_constraint(dof_hanging_nodes_idx));
  }

  /**
   * Reinit and fill AffineConstraints according to given Dirichlet boundary conditions
   * @param bc_data and assign it to a given @param dof_idx inside @param scratch_data.
   *
   * If @param set_inhomogeneities is true, inhomogeneities are considered else
   * homogeneous DBC are assumed.
   *
   * Reinit and fill AffineConstraints corresponding to a given @param dof_hanging_nodes_idx
   * inside @param scratch_data, considering hanging nodes.
   *
   * Insert algebraic periodicity constraints according to @param pbc into the
   * AffineConstraints corresponding to a given @param dof_idx and @param dof_hanging_nodes_idx
   * inside @param scratch_data.
   *
   * Merge the constraints of the AffineConstraints corresponding to a given @param dof_hanging_nodes_idx
   * into the AffineConstraints corresponding to a given @param dof_idx.
   *
   * The AffineConstraints objects are closed after filling them.
   * In Debug, the constraints are also checked with check_constraints().
   */
  template <int dim, typename number>
  void
  make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC(
    ScratchData<dim, dim, number> &scratch_data,
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> &bc_data,
    const PeriodicBoundaryConditions<dim>                                              &pbc,
    const unsigned int                                                                  dof_idx,
    const unsigned int dof_hanging_nodes_idx,
    const bool         set_inhomogeneities = true)
  {
    make_HNC_plus_PBC(scratch_data, pbc, dof_hanging_nodes_idx);
    make_DBC(scratch_data, bc_data, dof_idx, set_inhomogeneities, true /* close */);
    merge_HNC_into_DBC<dim, number>(scratch_data, dof_idx, dof_hanging_nodes_idx);

    check_constraints(scratch_data.get_dof_handler(dof_idx), scratch_data.get_constraint(dof_idx));
  }
} // namespace MeltPoolDG::Constraints
