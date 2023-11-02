/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, June 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/boundary_conditions.hpp>
#include <meltpooldg/interface/periodic_boundary_conditions.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::UtilityFunctions
{
  /**
   * Check whether @p constraints are consistent in parallel with the
   * @p dof_handler.
   */
  template <int dim, int spacedim, typename number>
  void
  check_constraints(const DoFHandler<dim, spacedim> &dof_handler,
                    const AffineConstraints<number> &constraints)
  {
#ifndef DEBUG
    return;
#endif

    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

    AssertThrow(constraints.is_consistent_in_parallel(
                  dealii::Utilities::MPI::all_gather(dof_handler.get_communicator(),
                                                     dof_handler.locally_owned_dofs()),
                  locally_active_dofs,
                  dof_handler.get_communicator()),
                ExcInternalError());
  }
  /**
   * Setup AffineConstraints according to given Dirichlet boundary conditions
   * @p bc_data and assign it to a given @p dof_idx inside @p scratch_data.
   * If @p set_inhomogeneities is true, inhomogeneities are considered else
   * homogeneous DBC are assumed.
   */
  template <int dim>
  void
  setup_constraints(ScratchData<dim>                       &scratch_data,
                    const DirichletBoundaryConditions<dim> &bc_data,
                    const unsigned int                      dof_idx,
                    const bool                              set_inhomogeneities = true)
  {
    // setup dirichlet constraints
    scratch_data.get_constraint(dof_idx).clear();
    scratch_data.get_constraint(dof_idx).reinit(scratch_data.get_locally_relevant_dofs(dof_idx));

    if (!bc_data.get_data().empty())
      {
        for (const auto &bc : bc_data.get_data())
          {
            if (set_inhomogeneities)
              dealii::VectorTools::interpolate_boundary_values(
                scratch_data.get_mapping(),
                scratch_data.get_dof_handler(dof_idx),
                bc.first,
                *bc.second,
                scratch_data.get_constraint(dof_idx));
            else
              dealii::DoFTools::make_zero_boundary_constraints(
                scratch_data.get_dof_handler(dof_idx),
                bc.first,
                scratch_data.get_constraint(dof_idx));
          }
      }

    scratch_data.get_constraint(dof_idx).close();
    check_constraints(scratch_data.get_dof_handler(dof_idx), scratch_data.get_constraint(dof_idx));
  }

  /**
   * Setup AffineConstraints according to given Dirichlet boundary conditions
   * @p bc_data and assign it to a given @p dof_idx inside @p scratch_data.
   * It will be automatically merged with the AffineConstraints corresponding
   * to @p dof_hanging_nodes_idx. If @p set_inhomogeneities is true,
   * inhomogeneities are considered else homogeneous DBC are assumed.
   */
  template <int dim>
  void
  setup_and_merge_constraints(ScratchData<dim>                       &scratch_data,
                              const DirichletBoundaryConditions<dim> &bc_data,
                              const unsigned int                      dof_idx,
                              const unsigned int                      dof_hanging_nodes_idx,
                              const bool                              set_inhomogeneities = true)
  {
    // setup dirichlet constraints
    setup_constraints(scratch_data, bc_data, dof_idx, set_inhomogeneities);

    scratch_data.get_constraint(dof_idx).merge(
      scratch_data.get_constraint(dof_hanging_nodes_idx),
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
    scratch_data.get_constraint(dof_idx).close();
    check_constraints(scratch_data.get_dof_handler(dof_idx), scratch_data.get_constraint(dof_idx));
  }

  /**
   * Setup AffineConstraints corresponding to a given @p dof_hanging_nodes_idx
   * inside @p scratch_data, considering hanging nodes.
   */
  template <int dim>
  void
  setup_constraints(ScratchData<dim> &scratch_data, const unsigned int dof_hanging_nodes_idx)
  {
    // setup hanging node constraints and periodic boundary constraints
    scratch_data.get_constraint(dof_hanging_nodes_idx).clear();
    scratch_data.get_constraint(dof_hanging_nodes_idx)
      .reinit(scratch_data.get_locally_relevant_dofs(dof_hanging_nodes_idx));
    DoFTools::make_hanging_node_constraints(scratch_data.get_dof_handler(dof_hanging_nodes_idx),
                                            scratch_data.get_constraint(dof_hanging_nodes_idx));
    scratch_data.get_constraint(dof_hanging_nodes_idx).close();

    check_constraints(scratch_data.get_dof_handler(dof_hanging_nodes_idx),
                      scratch_data.get_constraint(dof_hanging_nodes_idx));
  }

  /**
   * Setup AffineConstraints corresponding to a given @p dof_hanging_nodes_idx
   * inside @p scratch_data, considering periodic boundary conditions @p
   * pbc and hanging nodes.
   */
  template <int dim>
  void
  setup_constraints(ScratchData<dim>                      &scratch_data,
                    const PeriodicBoundaryConditions<dim> &pbc,
                    const unsigned int                     dof_hanging_nodes_idx)
  {
    // setup hanging node constraints and periodic boundary constraints
    scratch_data.get_constraint(dof_hanging_nodes_idx).clear();
    scratch_data.get_constraint(dof_hanging_nodes_idx)
      .reinit(scratch_data.get_locally_relevant_dofs(dof_hanging_nodes_idx));
    DoFTools::make_hanging_node_constraints(scratch_data.get_dof_handler(dof_hanging_nodes_idx),
                                            scratch_data.get_constraint(dof_hanging_nodes_idx));

    for (const auto &bc : pbc.get_data())
      {
        const auto [id_in, id_out, direction] = bc;
        DoFTools::make_periodicity_constraints(scratch_data.get_dof_handler(dof_hanging_nodes_idx),
                                               id_in,
                                               id_out,
                                               direction,
                                               scratch_data.get_constraint(dof_hanging_nodes_idx));
      }

    scratch_data.get_constraint(dof_hanging_nodes_idx).close();

    check_constraints(scratch_data.get_dof_handler(dof_hanging_nodes_idx),
                      scratch_data.get_constraint(dof_hanging_nodes_idx));
  }

  /**
   * Setup AffineConstraints according to given Dirichlet boundary conditions
   * @p bc_data, periodic boundary conditions @p pbc and hanging nodes @p dof_hanging_nodes_idx,
   * and assign it to a given @p dof_idx inside @p scratch_data.
   * If @p set_inhomogeneities is true, inhomogeneities are considered else
   * homogeneous DBC are assumed.
   */
  template <int dim>
  void
  setup_constraints(ScratchData<dim>                       &scratch_data,
                    const DirichletBoundaryConditions<dim> &bc_data,
                    const PeriodicBoundaryConditions<dim>  &pbc,
                    const unsigned int                      dof_idx,
                    const unsigned int                      dof_hanging_nodes_idx,
                    const bool                              set_inhomogeneities = true)
  {
    setup_constraints(scratch_data, pbc, dof_hanging_nodes_idx);
    setup_and_merge_constraints(
      scratch_data, bc_data, dof_idx, dof_hanging_nodes_idx, set_inhomogeneities);
  }
} // namespace MeltPoolDG::UtilityFunctions
