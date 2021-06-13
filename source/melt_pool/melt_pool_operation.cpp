#include <meltpooldg/melt_pool/melt_pool_operation.hpp>
//

#include <meltpooldg/heat/laser_heat_source_gauss.hpp>
#include <meltpooldg/heat/laser_heat_source_gusarov.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG::MeltPool
{
  template <int dim>
  MeltPoolOperation<dim>::MeltPoolOperation(
    const std::shared_ptr<ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &               data_in,
    const unsigned int                       ls_dof_idx_in,
    VectorType *                             temperature,
    const unsigned int                       reinit_dof_idx_in,
    const unsigned int                       flow_vel_dof_idx_in,
    const unsigned int                       flow_vel_quad_idx_in,
    const unsigned int                       temp_dof_idx_in,
    const double                             start_time_in)
    : scratch_data(scratch_data_in)
    , material(data_in.material)
    , do_mushy_zone(data_in.heat.solidification)
    , ls_dof_idx(ls_dof_idx_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , flow_vel_dof_idx(flow_vel_dof_idx_in)
    , flow_vel_quad_idx(flow_vel_quad_idx_in)
    , temp_dof_idx(temp_dof_idx_in)
    , temperature(temperature)
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
     * initialize the recoil pressure operation class
     */
    if (data_in.mp.do_recoil_pressure)
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

    /*
     * Choose the laser heat source model
     */
    if (data_in.laser.heat_source_model == "Gusarov")
      {
        laser_heat_source_operation =
          std::make_shared<Heat::LaserHeatSourceGusarov<dim>>(data_in.laser.gusarov);
      }
    else if (data_in.laser.heat_source_model == "Gauss")
      {
        laser_heat_source_operation =
          std::make_shared<Heat::LaserHeatSourceGauss<dim>>(data_in.laser.gauss);
      }
    else if (data_in.laser.heat_source_model == "Analytical")
      {
        laser_analytical_temperature_field =
          std::make_shared<Heat::LaserAnalyticalTemperatureField<dim>>(*scratch_data_in,
                                                                       data_in.laser.analytical,
                                                                       data_in.material,
                                                                       data_in.laser.scan_speed,
                                                                       temp_dof_idx_in);
      }
    else
      AssertThrow(false,
                  ExcMessage("No requested laser model found. Please specify the "
                             "heat source model in the laser section of the input parameters."))
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::set_initial_condition(const VectorType &level_set_as_heaviside,
                                                VectorType &      level_set)
  {
    /*
     *  Compute analytical temperature field
     */
    if (laser_analytical_temperature_field)
      laser_analytical_temperature_field->compute_temperature_field(
        level_set_as_heaviside,
        *temperature,
        laser_operation->get_laser_power(),
        laser_operation->get_laser_position());
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
  MeltPoolOperation<dim>::compute_heat_source(VectorType &      heat_source,
                                              const VectorType &level_set_as_heaviside,
                                              const double &    dt,
                                              const bool        zero_out)
  {
    // 0) move laser
    laser_operation->move_laser(dt);

    // 1) compute heat source or define analytical temperature field
    if (laser_analytical_temperature_field)
      {
        laser_analytical_temperature_field->compute_temperature_field(
          level_set_as_heaviside,
          *temperature,
          laser_operation->get_laser_power(),
          laser_operation->get_laser_position());
      }
    else
      {
        switch (laser_operation->get_laser_impact_type())
          {
              case Heat::LaserImpactType::volumetric: {
                laser_heat_source_operation->compute_volumetric_heat_source(
                  heat_source,
                  *scratch_data,
                  temp_dof_idx,
                  laser_operation->get_laser_power(),
                  laser_operation->get_laser_position(),
                  zero_out);
                break;
              }
              case Heat::LaserImpactType::interface: {
                laser_heat_source_operation->compute_interfacial_heat_source(
                  heat_source,
                  *scratch_data,
                  temp_dof_idx,
                  laser_operation->get_laser_power(),
                  laser_operation->get_laser_position(),
                  level_set_as_heaviside,
                  ls_dof_idx,
                  zero_out);
                break;
              }
              default: {
                AssertThrow(false, ExcMessage("Laser impact type not implemented here! Abort..."));
                break;
              }
          }
      }
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::compute_melt_front_propagation(const VectorType &level_set_as_heaviside)
  {
    if (!laser_analytical_temperature_field)
      {
        // 1) update phases
        compute_solid_and_liquid_phases(level_set_as_heaviside);

        // 2) update the constraints such that the solid domain is spatially fixed
        make_constraints_in_spatially_fixed_solid_domain();
      }
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::compute_force_flow_rhs(VectorType &      vel_force_rhs,
                                                 const VectorType &level_set_as_heaviside,
                                                 const bool        zero_out) const
  {
    // compute recoil pressure force
    if (recoil_pressure_operation)
      recoil_pressure_operation->compute_recoil_pressure_force(
        vel_force_rhs,
        level_set_as_heaviside,
        *temperature,
        zero_out /*false means add to force vector*/);
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::reinit()
  {
    scratch_data->initialize_dof_vector(solid, temp_dof_idx);
    scratch_data->initialize_dof_vector(liquid, temp_dof_idx);
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solid.update_ghost_values();
    liquid.update_ghost_values();
    vectors.push_back(&solid);
    vectors.push_back(&liquid);
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
#if DEAL_II_VERSION_MAJOR < 10
    MeltPoolDG::VectorTools::update_ghost_values(solid, liquid);
#endif
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
    scratch_data->get_constraint(temp_dof_idx).distribute(solid);
    scratch_data->get_constraint(temp_dof_idx).distribute(liquid);
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
    temperature->update_ghost_values();

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
                compute_solid_fraction(level_set_as_heaviside[local_dof_indices[i]],
                                       (*temperature)[local_dof_indices[i]]);
              liquid[local_dof_indices[i]] =
                compute_liquid_fraction(level_set_as_heaviside[local_dof_indices[i]],
                                        (*temperature)[local_dof_indices[i]]);
            }
        }

    liquid.compress(VectorOperation::insert);
    solid.compress(VectorOperation::insert);

    temperature->zero_out_ghost_values();
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

    UtilityFunctions::check_constraints(flow_dof_handler, solid_constraints);

    flow_constraints.merge(solid_constraints,
                           AffineConstraints<double>::MergeConflictBehavior::left_object_wins);

    UtilityFunctions::check_constraints(flow_dof_handler, flow_constraints);

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

    UtilityFunctions::check_constraints(level_set_dof_handler, solid_constraints);

    level_set_constraints.merge(solid_constraints,
                                AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
    level_set_constraints.close();

    UtilityFunctions::check_constraints(level_set_dof_handler, level_set_constraints);

    solid.zero_out_ghost_values();
  }

  template <int dim>
  void
  MeltPoolOperation<dim>::set_melt_pool_parameters(const Parameters<double> &data_in)
  {
    mp_data = data_in.mp;
  }

  template <int dim>
  double
  MeltPoolOperation<dim>::compute_solid_fraction(const double ls_heaviside, const double T) const
  {
    const auto sf = compute_solid_fraction_no_ls(T);
    if (do_mushy_zone)
      return ls_heaviside * sf;
    else
      return ls_heaviside > 0.5 ? sf : 0.0;
  }

  template <int dim>
  double
  MeltPoolOperation<dim>::compute_liquid_fraction(const double ls_heaviside, const double T) const
  {
    const auto lf = 1.0 - compute_solid_fraction_no_ls(T);
    if (do_mushy_zone)
      return ls_heaviside * lf;
    else
      return ls_heaviside > 0.5 ? lf : 0.0;
  }

  template <int dim>
  double
  MeltPoolOperation<dim>::compute_solid_fraction_no_ls(const double T) const
  {
    if (do_mushy_zone)
      return UtilityFunctions::limit_to_bounds(
        (material.liquidus_temperature - T) * material.inv_mushy_interval, 0.0, 1.0);
    else
      return T < mp_data.liquid.melting_point ? 1.0 : 0.0;
  }

  template class MeltPoolOperation<1>;
  template class MeltPoolOperation<2>;
  template class MeltPoolOperation<3>;
} // namespace MeltPoolDG::MeltPool
