#include <meltpooldg/melt_pool/melt_pool_operation.hpp>
//

#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG::MeltPool
{
  template <int dim>
  MeltPoolOperation<dim>::MeltPoolOperation(
    const std::shared_ptr<ScratchData<dim>> &       scratch_data_in,
    const std::shared_ptr<BoundaryConditions<dim>> &heat_bc,
    const Parameters<double> &                      data_in,
    const unsigned int                              ls_dof_idx_in,
    const unsigned int                              reinit_dof_idx_in,
    const unsigned int                              flow_vel_dof_idx_in,
    const unsigned int                              flow_vel_quad_idx_in,
    const unsigned int                              temp_dof_idx_in,
    const unsigned int                              temp_quad_idx_in,
    const double                                    start_time_in,
    bool                                            do_recoil_pressure)
    : scratch_data(scratch_data_in)
    , ls_dof_idx(ls_dof_idx_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , flow_vel_dof_idx(flow_vel_dof_idx_in)
    , flow_vel_quad_idx(flow_vel_quad_idx_in)
    , temp_dof_idx(temp_dof_idx_in)
    , temp_quad_idx(temp_quad_idx_in)
  {
    /*
     *  set the parameters for the melt pool operation
     */
    set_melt_pool_parameters(data_in);
    /*
     *  initialize the laser operation class
     */
    laser_operation =
      std::make_shared<Heat::LaserOperation<dim>>(*scratch_data, data_in.laser, data_in.material);
    /*
     *  Initialize the laser operation
     */
    laser_operation->set_initial_condition(start_time_in);
    /*
     *  initialize the heat operation class
     */
    heat_operation =
      std::make_shared<Heat::HeatTransferOperation<dim>>(heat_bc,
                                                         *scratch_data,
                                                         data_in.heat,
                                                         data_in.material,
                                                         temp_dof_idx,
                                                         temp_dof_idx, //@todo: hanging nodes
                                                         temp_quad_idx,
                                                         flow_vel_dof_idx);

    /*
     * initialize the recoil pressure operation class
     */
    if (do_recoil_pressure)
      recoil_pressure_operation =
        std::make_shared<RecoilPressureOperation<dim>>(*scratch_data_in,
                                                       data_in,
                                                       flow_vel_dof_idx_in,
                                                       flow_vel_quad_idx_in,
                                                       ls_dof_idx_in,
                                                       temp_dof_idx_in);
    /*
     * initialize the dof_vectors
     */
    scratch_data->initialize_dof_vector(solid, temp_dof_idx);
    scratch_data->initialize_dof_vector(liquid, temp_dof_idx);
    heat_operation->reinit();

    if (mp_data.temperature_formulation == "solve_heat_equation")
      {
        if (data_in.laser.heat_source_model == "Gusarov")
          {
            laser_heat_source_operation =
              std::make_shared<Heat::LaserHeatSourceGusarov<dim>>(data_in.laser.gusarov);
          }
        else
          AssertThrow(false,
                      ExcMessage("No requested laser model found. Please speficy the "
                                 "heat source model in the laser section of the input parameters."))
      }
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::set_initial_condition(const VectorType &level_set_as_heaviside,
                                                VectorType &      level_set,
                                                const double &    density_gas,
                                                const double &    density_liquid)
  {
    /*
     *  Compute analytical temperature field
     */
    if (mp_data.temperature_formulation == "analytical")
      laser_operation->compute_analytical_temperature_field(
        level_set_as_heaviside,
        heat_operation->get_temperature(),
        temp_dof_idx,
        density_gas,
        density_liquid,
        mp_data,
        -1.0 /* level set value for gas @todo */);
    else if (mp_data.temperature_formulation == "solve_heat_equation")
      {
        heat_operation->get_temperature() = mp_data.ambient_temperature;
      }
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *  Compute the initial solid and liquid phases
     */
    compute_solid_and_liquid_phases(level_set_as_heaviside);
    /*
     *  Constraint the solid domain if requested
     */
    make_constraints_in_spatially_fixed_solid_domain();
    scratch_data->get_constraint(ls_dof_idx).distribute(level_set);
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::solve(VectorType &      vel_force_rhs,
                                const VectorType &level_set_as_heaviside,
                                const double &    density_gas,
                                const double &    density_liquid,
                                const double &    dt)
  {
    // 0) move laser
    laser_operation->move_laser(dt);

    // 1) compute temperature field
    if (mp_data.temperature_formulation == "solve_heat_equation")
      {
        // @todo: support other heat source models
        laser_heat_source_operation->compute_volumetric_heat_source(
          heat_operation->get_heat_source(),
          *scratch_data,
          temp_dof_idx,
          laser_operation->get_laser_power(),
          laser_operation->get_laser_position(),
          true /* zero_out */);
        heat_operation->solve(dt);
      }
    else if (mp_data.temperature_formulation == "analytical")
      laser_operation->compute_analytical_temperature_field(
        level_set_as_heaviside,
        heat_operation->get_temperature(),
        temp_dof_idx,
        density_gas,
        density_liquid,
        mp_data,
        -1.0 /* level set value for gas @todo */);
    else
      AssertThrow(false, ExcNotImplemented());

    // 2) update phases
    if (mp_data.temperature_formulation != "analytical")
      {
        compute_solid_and_liquid_phases(level_set_as_heaviside);

        // 3) update the constraints such that the solid domain is spatially fixed
        make_constraints_in_spatially_fixed_solid_domain();
      }

    // 4) compute forces
    //   i)  ... recoil pressure
    if (recoil_pressure_operation)
      recoil_pressure_operation->compute_recoil_pressure_force(
        vel_force_rhs,
        level_set_as_heaviside,
        heat_operation->get_temperature(),
        false /*false means add to force vector*/);

    //   ii) ... or evaporative flux
    //@todo
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::reinit()
  {
    heat_operation->reinit();
    scratch_data->initialize_dof_vector(solid, temp_dof_idx);
    scratch_data->initialize_dof_vector(liquid, temp_dof_idx);
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    heat_operation->attach_vectors(vectors);
    solid.update_ghost_values();
    liquid.update_ghost_values();
    vectors.push_back(&solid);
    vectors.push_back(&liquid);
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    heat_operation->attach_output_vectors(data_out);
    MeltPoolDG::VectorTools::update_ghost_values(solid, liquid);
    /**
     *  solid
     */
    data_out.add_data_vector(scratch_data->get_dof_handler(temp_dof_idx), solid, "solid");
    /**
     *  liquid
     */
    data_out.add_data_vector(scratch_data->get_dof_handler(temp_dof_idx), liquid, "liquid");
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::distribute_constraints()
  {
    heat_operation->distribute_constraints();
    scratch_data->get_constraint(temp_dof_idx).distribute(solid);
    scratch_data->get_constraint(temp_dof_idx).distribute(liquid);
  }

  template <int dim>
  const VectorType &
  MeltPoolOperation<dim>::get_temperature() const
  {
    return heat_operation->get_temperature();
  }

  template <int dim>
  const VectorType &
  MeltPoolOperation<dim>::get_solid() const
  {
    return solid;
  }

  template <int dim>
  const VectorType &
  MeltPoolOperation<dim>::get_liquid() const
  {
    return liquid;
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::make_constraints_in_spatially_fixed_solid_domain()
  {
    /*
     *    limit the level set interface to the touching regions of liquid/gas
     */
    if (mp_data.set_level_set_to_zero_in_solid)
      {
        remove_the_level_set_from_solid_regions(scratch_data->get_dof_handler(ls_dof_idx),
                                                scratch_data->get_constraint(ls_dof_idx));
        remove_the_level_set_from_solid_regions(scratch_data->get_dof_handler(reinit_dof_idx),
                                                scratch_data->get_constraint(reinit_dof_idx));
      }

    if (mp_data.set_velocity_to_zero_in_solid)
      set_flow_field_in_solid_regions_to_zero(scratch_data->get_dof_handler(flow_vel_dof_idx),
                                              scratch_data->get_constraint(flow_vel_dof_idx));
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::compute_solid_and_liquid_phases(const VectorType &level_set_as_heaviside)
  {
    level_set_as_heaviside.update_ghost_values();
    heat_operation->get_temperature().update_ghost_values();

    const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(temp_dof_idx);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                         scratch_data->get_dof_handler(temp_dof_idx),
                                         support_points);

    liquid = 0;
    solid  = 0;

    for (const auto &cell : scratch_data->get_dof_handler(temp_dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              solid[local_dof_indices[i]] =
                is_solid_region(level_set_as_heaviside[local_dof_indices[i]],
                                heat_operation->get_temperature()[local_dof_indices[i]]);
              liquid[local_dof_indices[i]] =
                is_liquid_region(level_set_as_heaviside[local_dof_indices[i]],
                                 heat_operation->get_temperature()[local_dof_indices[i]]);
            }
        }

    liquid.compress(VectorOperation::insert);
    solid.compress(VectorOperation::insert);

    scratch_data->get_constraint(temp_dof_idx).distribute(solid);
    scratch_data->get_constraint(temp_dof_idx).distribute(liquid);

    heat_operation->get_temperature().zero_out_ghost_values();
    level_set_as_heaviside.zero_out_ghost_values();
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::set_flow_field_in_solid_regions_to_zero(
    const DoFHandler<dim> &    flow_dof_handler,
    AffineConstraints<double> &flow_constraints)
  {
    solid.update_ghost_values();

    IndexSet flow_locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(flow_dof_handler, flow_locally_relevant_dofs);

    AffineConstraints<double> solid_constraints;
    solid_constraints.reinit(flow_locally_relevant_dofs);

    FEValues<dim> flow_eval(scratch_data->get_mapping(),
                            flow_dof_handler.get_fe(),
                            Quadrature<dim>(flow_dof_handler.get_fe().get_unit_support_points()),
                            update_quadrature_points);

    FEValues<dim> solid_eval(scratch_data->get_mapping(),
                             scratch_data->get_dof_handler(temp_dof_idx).get_fe(),
                             Quadrature<dim>(flow_dof_handler.get_fe().get_unit_support_points()),
                             update_values);

    const unsigned int dofs_per_cell = flow_dof_handler.get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  solid_at_q(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator solid_cell =
      scratch_data->get_dof_handler(temp_dof_idx).begin_active();

    for (const auto &cell : flow_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            flow_eval.reinit(cell);

            solid_eval.reinit(solid_cell);
            solid_eval.get_function_values(solid, solid_at_q);

            for (const auto q : flow_eval.quadrature_point_indices())
              {
                if (solid_at_q[q] == 1.0)
                  solid_constraints.add_line(local_dof_indices[q]);
              }
          }
        ++solid_cell;
      }

    solid_constraints.close();
    flow_constraints.merge(solid_constraints,
                           AffineConstraints<double>::MergeConflictBehavior::left_object_wins);

    solid.zero_out_ghost_values();
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::remove_the_level_set_from_solid_regions(
    const DoFHandler<dim> &    level_set_dof_handler,
    AffineConstraints<double> &level_set_constraints)
  {
    solid.update_ghost_values();
    AssertThrow(scratch_data->get_degree(temp_dof_idx) == scratch_data->get_degree(ls_dof_idx),
                ExcMessage(
                  "The usage of this function assumes that the temperature field "
                  "is interpolated with the polynomial with the same degree as the level set"));

    const unsigned int dofs_per_cell = level_set_dof_handler.get_fe().n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                         level_set_dof_handler,
                                         support_points);

    AffineConstraints<double> solid_constraints;

    IndexSet ls_locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(level_set_dof_handler, ls_locally_relevant_dofs);

    solid_constraints.reinit(ls_locally_relevant_dofs);
    for (const auto &cell : level_set_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            if (solid[local_dof_indices[i]])
              solid_constraints.add_line(local_dof_indices[i]);
        }
    level_set_constraints.merge(solid_constraints,
                                AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
    level_set_constraints.close();
    solid.zero_out_ghost_values();
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::set_melt_pool_parameters(const Parameters<double> &data_in)
  {
    mp_data = data_in.mp;
  }

  template <int dim>
  bool
  MeltPoolOperation<dim>::is_solid_region(const double phi_liquid, const double temperature)
  {
    if (phi_liquid <= 0.5)
      return false; // point is gas phase
    else if (phi_liquid > 0.5 && temperature >= mp_data.liquid.melting_point)
      return false; // point is melted
    else
      return true;
  }

  template <int dim>
  bool
  MeltPoolOperation<dim>::is_liquid_region(const double phi_liquid, const double temperature)
  {
    if (phi_liquid <= 0.5)
      return false; // point is gas phase
    else if (phi_liquid > 0.5 && temperature >= mp_data.liquid.melting_point)
      return true; // point is melted
    else
      return false; // point is solid
  }

  template class MeltPoolOperation<1>;
  template class MeltPoolOperation<2>;
  template class MeltPoolOperation<3>;
} // namespace MeltPoolDG::MeltPool
