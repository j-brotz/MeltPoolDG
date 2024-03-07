#include <meltpooldg/melt_pool/melt_front_propagation.hpp>
//

#include <deal.II/dofs/dof_tools.h>

#include <meltpooldg/material/material.templates.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::MeltPool
{
  template <int dim>
  MeltFrontPropagation<dim>::MeltFrontPropagation(const ScratchData<dim>   &scratch_data_in,
                                                  const Parameters<double> &data_in,
                                                  const unsigned int        ls_dof_idx_in,
                                                  const VectorType         &temperature,
                                                  const unsigned int        reinit_dof_idx_in,
                                                  const unsigned int reinit_no_solid_dof_idx_in,
                                                  const unsigned int flow_vel_dof_idx_in,
                                                  const unsigned int flow_vel_no_solid_dof_idx_in,
                                                  const unsigned int temp_hanging_nodes_dof_idx_in)
    : scratch_data(scratch_data_in)
    , mp_data(data_in.mp)
    , melting_solidification(data_in.material, MaterialTypes::liquid_solid)
    , ls_dof_idx(ls_dof_idx_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_no_solid_dof_idx(reinit_no_solid_dof_idx_in)
    , flow_vel_dof_idx(flow_vel_dof_idx_in)
    , flow_vel_no_solid_dof_idx(flow_vel_no_solid_dof_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , temperature(temperature)
  {
    /*
     * initialize the dof_vectors
     */
    scratch_data.initialize_dof_vector(solid, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(liquid, temp_hanging_nodes_dof_idx);
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::set_initial_condition(const VectorType &level_set_as_heaviside,
                                                   VectorType       &level_set)
  {
    /*
     *  Compute the initial solid and liquid phases
     */
    compute_solid_and_liquid_phases(level_set_as_heaviside);
    /*
     *  Constraint the solid domain if requested
     */
    make_constraints_in_spatially_fixed_solid_domain();
    /*
     * distribute the constraints for the level set in the solid region
     */
    if (mp_data.solid.do_not_reinitialize)
      scratch_data.get_constraint(ls_dof_idx).distribute(level_set);
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::compute_melt_front_propagation(
    const VectorType &level_set_as_heaviside)
  {
    // 1) update phases
    compute_solid_and_liquid_phases(level_set_as_heaviside);


    // 2) update the constraints such that the solid domain is spatially fixed
    make_constraints_in_spatially_fixed_solid_domain();
    /*
     * distribute the constraints for the level set in the solid region
     */
    //@todo:
    // scratch_data.get_constraint(ls_dof_idx).distribute(level_set);
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::reinit()
  {
    scratch_data.initialize_dof_vector(solid, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(liquid, temp_hanging_nodes_dof_idx);
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    // TODO: remove -- not needed
    vectors.push_back(&solid);
    vectors.push_back(&liquid);
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    /**
     *  solid
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             solid,
                             "solid");
    /**
     *  liquid
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             liquid,
                             "liquid");
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::distribute_constraints()
  {
    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(solid);
    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(liquid);
  }

  template <int dim>
  const VectorType &
  MeltFrontPropagation<dim>::get_solid() const
  {
    return solid;
  }

  template <int dim>
  const VectorType &
  MeltFrontPropagation<dim>::get_liquid() const
  {
    return liquid;
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::make_constraints_in_spatially_fixed_solid_domain()
  {
    /*
     *  Do not reinitialize the level set field in the solid domain
     */
    if (mp_data.solid.do_not_reinitialize)
      ignore_reinitialization_in_solid_regions(scratch_data.get_dof_handler(reinit_dof_idx),
                                               scratch_data.get_constraint(reinit_no_solid_dof_idx),
                                               const_cast<AffineConstraints<double> &>(
                                                 scratch_data.get_constraint(reinit_dof_idx)));

    /*
     *  Set the flow velocity to zero in the solid domain
     */
    if (mp_data.solid.set_velocity_to_zero)
      set_flow_field_in_solid_regions_to_zero(
        scratch_data.get_dof_handler(flow_vel_dof_idx),
        scratch_data.get_constraint(flow_vel_no_solid_dof_idx),
        const_cast<AffineConstraints<double> &>(scratch_data.get_constraint(flow_vel_dof_idx)));

    // In the melt pool simulations, the solid domain
    // can be considered as rigid by setting the constraints
    // for the velocity field to zero. If the solid domain
    // changes, the AffineConstraints for the velocity field
    // will be updated accordingly. In this case, also the
    // constrained indices in matrix-free have to be updated
    // which is done in the following by rebuilding matrix-free.
    if (mp_data.solid.set_velocity_to_zero || mp_data.solid.do_not_reinitialize)
      const_cast<ScratchData<dim> &>(scratch_data)
        .build(scratch_data.enable_boundary_faces, scratch_data.enable_inner_faces);
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::compute_solid_and_liquid_phases(
    const VectorType &level_set_as_heaviside)
  {
    const bool update_ghosts = !level_set_as_heaviside.has_ghost_elements();
    if (update_ghosts)
      level_set_as_heaviside.update_ghost_values();
    const bool temp_update_ghosts = !temperature.has_ghost_elements();
    if (temp_update_ghosts)
      temperature.update_ghost_values();

    std::vector<types::global_dof_index> local_dof_indices(
      scratch_data.get_n_dofs_per_cell(temp_hanging_nodes_dof_idx));

    FEValues<dim> ls_heaviside_eval(scratch_data.get_mapping(),
                                    scratch_data.get_dof_handler(ls_dof_idx).get_fe(),
                                    Quadrature<dim>(
                                      scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx)
                                        .get_fe()
                                        .get_unit_support_points()),
                                    update_values);

    std::vector<double> ls_heaviside_at_q(ls_heaviside_eval.n_quadrature_points);

    liquid = 0;
    solid  = 0;

    typename DoFHandler<dim>::active_cell_iterator ls_cell =
      scratch_data.get_dof_handler(ls_dof_idx).begin_active();

    for (const auto &cell :
         scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            ls_heaviside_eval.reinit(ls_cell);
            ls_heaviside_eval.get_function_values(level_set_as_heaviside, ls_heaviside_at_q);

            for (const auto q : ls_heaviside_eval.quadrature_point_indices())
              {
                solid[local_dof_indices[q]] =
                  compute_solid_fraction((temperature)[local_dof_indices[q]]) *
                  ls_heaviside_at_q[q];

                liquid[local_dof_indices[q]] =
                  (1. - compute_solid_fraction((temperature)[local_dof_indices[q]])) *
                  ls_heaviside_at_q[q];
              }
          }
        ++ls_cell;
      }

    liquid.compress(VectorOperation::insert);
    solid.compress(VectorOperation::insert);

    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(solid);
    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(liquid);

    liquid.update_ghost_values();
    solid.update_ghost_values();

    if (temp_update_ghosts)
      temperature.zero_out_ghost_values();
    if (update_ghosts)
      level_set_as_heaviside.zero_out_ghost_values();
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::set_flow_field_in_solid_regions_to_zero(
    const DoFHandler<dim>           &flow_dof_handler,
    const AffineConstraints<double> &flow_constraints_no_solid,
    AffineConstraints<double>       &flow_constraints)
  {
    const bool update_ghosts = !solid.has_ghost_elements();
    if (update_ghosts)
      solid.update_ghost_values();

    flow_constraints.copy_from(flow_constraints_no_solid);

    IndexSet flow_locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(flow_dof_handler, flow_locally_relevant_dofs);

    AffineConstraints<double> solid_constraints;
    solid_constraints.reinit(flow_locally_relevant_dofs);

    FEValues<dim> flow_eval(scratch_data.get_mapping(),
                            flow_dof_handler.get_fe(),
                            Quadrature<dim>(flow_dof_handler.get_fe().get_unit_support_points()),
                            update_quadrature_points);

    FEValues<dim> solid_eval(scratch_data.get_mapping(),
                             scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx).get_fe(),
                             Quadrature<dim>(flow_dof_handler.get_fe().get_unit_support_points()),
                             update_values);

    const unsigned int dofs_per_cell = flow_dof_handler.get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  solid_at_q(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator solid_cell =
      scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx).begin_active();

    for (const auto &cell : flow_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            flow_eval.reinit(cell);

            solid_eval.reinit(solid_cell);
            solid_eval.get_function_values(solid, solid_at_q);

            for (const auto q : flow_eval.quadrature_point_indices())
              if (solid_at_q[q] >= mp_data.solid.solid_fraction_lower_limit)
                solid_constraints.add_line(local_dof_indices[q]);
          }
        ++solid_cell;
      }

    solid_constraints.make_consistent_in_parallel(flow_dof_handler.locally_owned_dofs(),
                                                  flow_locally_relevant_dofs,
                                                  scratch_data.get_mpi_comm());
    solid_constraints.close();

    Constraints::check_constraints(flow_dof_handler, solid_constraints);

    flow_constraints.merge(solid_constraints,
                           AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
    flow_constraints.close();

    Constraints::check_constraints(flow_dof_handler, flow_constraints);

    if (update_ghosts)
      solid.zero_out_ghost_values();
  }

  template <int dim>
  void
  MeltFrontPropagation<dim>::ignore_reinitialization_in_solid_regions(
    const DoFHandler<dim>           &level_set_dof_handler,
    const AffineConstraints<double> &reinit_dirichlet_constraints_no_solid,
    AffineConstraints<double>       &reinit_dirichlet_constraints)
  {
    reinit_dirichlet_constraints.copy_from(reinit_dirichlet_constraints_no_solid);

    const bool update_ghosts = !solid.has_ghost_elements();
    if (update_ghosts)
      solid.update_ghost_values();

    AffineConstraints<double> solid_constraints;

    IndexSet ls_locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(level_set_dof_handler, ls_locally_relevant_dofs);

    solid_constraints.reinit(ls_locally_relevant_dofs);

    FEValues<dim> ls_eval(scratch_data.get_mapping(),
                          level_set_dof_handler.get_fe(),
                          Quadrature<dim>(level_set_dof_handler.get_fe().get_unit_support_points()),
                          update_quadrature_points);

    FEValues<dim> solid_eval(scratch_data.get_mapping(),
                             scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx).get_fe(),
                             Quadrature<dim>(
                               level_set_dof_handler.get_fe().get_unit_support_points()),
                             update_values);

    const unsigned int dofs_per_cell = level_set_dof_handler.get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  solid_at_q(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator solid_cell =
      scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx).begin_active();

    for (const auto &cell : level_set_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            ls_eval.reinit(cell);

            solid_eval.reinit(solid_cell);
            solid_eval.get_function_values(solid, solid_at_q);

            for (const auto q : ls_eval.quadrature_point_indices())
              if (solid_at_q[q] >= mp_data.solid.solid_fraction_lower_limit)
                solid_constraints.add_line(local_dof_indices[q]);
          }
        ++solid_cell;
      }

    solid_constraints.make_consistent_in_parallel(level_set_dof_handler.locally_owned_dofs(),
                                                  ls_locally_relevant_dofs,
                                                  scratch_data.get_mpi_comm());
    solid_constraints.close();

    Constraints::check_constraints(level_set_dof_handler, solid_constraints);

    reinit_dirichlet_constraints.merge(
      solid_constraints, AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
    reinit_dirichlet_constraints.close();

    Constraints::check_constraints(level_set_dof_handler, reinit_dirichlet_constraints);

    if (update_ghosts)
      solid.zero_out_ghost_values();
  }

  template <int dim>
  double
  MeltFrontPropagation<dim>::compute_solid_fraction(const double T) const
  {
    return melting_solidification.compute_parameters(T, MaterialUpdateFlags::phase_fractions)
      .solid_fraction;
  }

  template <int dim>
  VectorizedArray<double>
  MeltFrontPropagation<dim>::compute_solid_fraction(
    const VectorizedArray<double> &current_temperature) const
  {
    VectorizedArray<double> result;
    for (unsigned int i = 0; i < VectorizedArray<double>::size(); ++i)
      result[i] = compute_solid_fraction(current_temperature[i]);
    return result;
  }

  template class MeltFrontPropagation<1>;
  template class MeltFrontPropagation<2>;
  template class MeltFrontPropagation<3>;
} // namespace MeltPoolDG::MeltPool
