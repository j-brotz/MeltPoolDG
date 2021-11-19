#include <meltpooldg/heat/laser_heat_source_gauss.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceGauss<dim>::LaserHeatSourceGauss(
    const LaserData<double>::GaussData &         data_in,
    const TwoPhaseFluidPropertiesTransitionType &variable_properties_over_interface)
    : data(data_in)
    , variable_properties_over_interface(variable_properties_over_interface)
    , vol_peak_power_density_factor(
        1. / std::pow(data.laser_beam_radius * std::sqrt(numbers::PI / 2), 3))
    , surf_peak_power_density_factor(
        1. / (data.laser_beam_radius * data.laser_beam_radius * numbers::PI / 2))
  {
    AssertThrow(data.laser_beam_radius > 0.0,
                ExcMessage("The laser beam radius must be greater than zero! Abort.."));

    delta_phase_weighted =
      create_phase_weighted_delta_approximation(data.delta_function_type,
                                                data.delta_approximation_phase_weighted);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::local_compute_volumetric_heat_source(const Point<dim> &position,
                                                                  const Point<dim> &laser_position,
                                                                  const double      power) const
  {
    const double distance = position.distance(laser_position);
    return data.absorptivity_liquid * power_density_volumetric(distance, power);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::local_compute_interfacial_heat_source(
    const Point<dim> &            position,
    const Point<dim> &            laser_position,
    const double                  power,
    const Tensor<1, dim, double> &normal_vector,
    const double                  delta_value,
    const double                  heaviside) const
  {
    // only consider distance in x (2D) or x and y (3D) direction. Disregard distance in dim-1
    // direction.
    Point<dim - 1> distance_vector;
    distance_vector[0] = position[0] - laser_position[0];
    if constexpr (dim == 3)
      distance_vector[1] = position[1] - laser_position[1];
    const double distance = distance_vector.norm();

    // assume laser direction coincides with the negative dim-1 direction
    double projection_factor = normal_vector * -Point<dim>::unit_vector(dim - 1);
    if (projection_factor < 0.0)
      projection_factor = 0.0;

    const double weight =
      (variable_properties_over_interface != TwoPhaseFluidPropertiesTransitionType::sharp) ?
        heaviside :
        ((heaviside > 0.5) ? 1.0 : 0.0);

    const double absorptivity =
      UtilityFunctions::interpolate(weight, data.absorptivity_gas, data.absorptivity_liquid);

    return absorptivity * projection_factor * delta_value *
           power_density_interfacial(distance, power);
  }

  template <int dim>
  void
  LaserHeatSourceGauss<dim>::compute_interfacial_heat_source(
    VectorType &            heat_source_vector,
    const ScratchData<dim> &scratch_data,
    const unsigned int      temp_dof_idx,
    const double            laser_power,
    const Point<dim> &      laser_position,
    const VectorType &      level_set_heaviside,
    const unsigned int      ls_dof_idx,
    const bool              zero_out,
    const BlockVectorType * normal_vector,
    const unsigned int      normal_dof_idx) const
  {
    if (zero_out)
      scratch_data.initialize_dof_vector(heat_source_vector, temp_dof_idx);

    level_set_heaviside.update_ghost_values();
    if (normal_vector)
      normal_vector->update_ghost_values();

    const double tolerance_normal_vector =
      UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                            scratch_data.get_mapping());

    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(temp_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
      update_quadrature_points);

    const VectorType *             used_level_set  = &level_set_heaviside;
    unsigned int                   used_ls_dof_idx = ls_dof_idx;
    std::unique_ptr<FEValues<dim>> ls_heaviside_eval;

    ls_heaviside_eval = std::make_unique<FEValues<dim>>(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
      update_values | update_gradients);

    FEValues<dim> normal_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(normal_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
      update_values);

    const unsigned int dofs_per_cell =
      scratch_data.get_dof_handler(temp_dof_idx).get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> ls_heaviside_at_q(ls_heaviside_eval->n_quadrature_points);
    std::vector<dealii::Tensor<1, dim, double>> grad_ls_heaviside_at_q(
      ls_heaviside_eval->n_quadrature_points);
    std::vector<Tensor<1, dim>> normal_at_q(dofs_per_cell, Tensor<1, dim>());

    /*
     * Interpolate the level set DoFs to the temperature DoFs
     */
    VectorType interpolated_vec;

    if (const FE_Q_iso_Q1<dim> *fe_interpolated = dynamic_cast<const FE_Q_iso_Q1<dim> *>(
          &scratch_data.get_dof_handler(ls_dof_idx).get_fe()))
      {
        const auto ls_to_temperature_grad_interpolation_matrix =
          UtilityFunctions::create_dof_interpolation_matrix<dim>(
            scratch_data.get_dof_handler(temp_dof_idx),
            scratch_data.get_dof_handler(ls_dof_idx),
            false);

        ls_heaviside_eval = std::make_unique<FEValues<dim>>(
          scratch_data.get_mapping(),
          scratch_data.get_dof_handler(temp_dof_idx).get_fe(),
          Quadrature<dim>(
            scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
          update_values | update_gradients);

        std::vector<types::global_dof_index> ls_local_dof_indices(
          scratch_data.get_dof_handler(ls_dof_idx).get_fe().n_dofs_per_cell());

        // create vector of interpolated values of level set at DoF points of the temperature field
        used_level_set  = &interpolated_vec;
        used_ls_dof_idx = temp_dof_idx;
        scratch_data.initialize_dof_vector(interpolated_vec, temp_dof_idx);

        for (const auto &cell : scratch_data.get_triangulation().active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
                  &scratch_data.get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &scratch_data.get_dof_handler(ls_dof_idx));
                ls_dof_cell->get_dof_indices(ls_local_dof_indices);

                TriaIterator<DoFCellAccessor<dim, dim, false>> temp_dof_cell(
                  &scratch_data.get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &scratch_data.get_dof_handler(temp_dof_idx));

                temp_dof_cell->get_dof_indices(local_dof_indices);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    double interpolated_value = 0;

                    /* Interpolate the level set Φ from the support points of the level set space j
                     * to the one of the temperature space i, using the interpolation matrix P
                     * _
                     * Φ   = P   · Φ
                     *  i     ij    j
                     */
                    for (unsigned int j = 0;
                         j < scratch_data.get_dof_handler(ls_dof_idx).get_fe().n_dofs_per_cell();
                         ++j)
                      interpolated_value += ls_to_temperature_grad_interpolation_matrix(i, j) *
                                            level_set_heaviside[ls_local_dof_indices[j]];

                    // Store the interpolated values at the support points of the pressure space
                    interpolated_vec[local_dof_indices[i]] = interpolated_value;
                  }
              }
          }

        interpolated_vec.compress(VectorOperation::insert);
        interpolated_vec.update_ghost_values();
      }

    // count the number of nodal assembly entries
    VectorType heat_source_vector_multiplicity;
    heat_source_vector_multiplicity.reinit(heat_source_vector);

    for (const auto &cell : scratch_data.get_triangulation().active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(used_ls_dof_idx));

            TriaIterator<DoFCellAccessor<dim, dim, false>> temp_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(temp_dof_idx));

            TriaIterator<DoFCellAccessor<dim, dim, false>> normal_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(normal_dof_idx));

            temp_dof_cell->get_dof_indices(local_dof_indices);

            // fill multiplicity
            Vector<double> heat_source_vector_multiplicity_local(dofs_per_cell);
            for (auto &val : heat_source_vector_multiplicity_local)
              val = 1.0;
            scratch_data.get_constraint(temp_dof_idx)
              .distribute_local_to_global(heat_source_vector_multiplicity_local,
                                          local_dof_indices,
                                          heat_source_vector_multiplicity);


            heat_source_eval.reinit(temp_dof_cell);
            normal_eval.reinit(normal_dof_cell);
            ls_heaviside_eval->reinit(ls_dof_cell);

            ls_heaviside_eval->get_function_gradients(*used_level_set, grad_ls_heaviside_at_q);
            ls_heaviside_eval->get_function_values(*used_level_set, ls_heaviside_at_q);

            // use filtered normal vector computation ..
            if (normal_vector)
              NormalVector::NormalVectorOperator<dim>::get_unit_normals_at_quadrature(
                normal_eval, *normal_vector, normal_at_q, tolerance_normal_vector);

            Vector<double> heat_source_vector_local(dofs_per_cell);

            for (const auto q : heat_source_eval.quadrature_point_indices())
              {
                const double grad_ls_norm = grad_ls_heaviside_at_q[q].norm();
                const double delta_value =
                  delta_phase_weighted == nullptr ?
                    grad_ls_norm :
                    grad_ls_norm * delta_phase_weighted->compute_weight(ls_heaviside_at_q[q]);

                if (delta_value == 0.0)
                  {
                    heat_source_vector_local[q] = 0;
                    continue;
                  }

                // ... or use (unfiltered) gradient of the level set function
                if (normal_vector == nullptr)
                  normal_at_q[q] = grad_ls_heaviside_at_q[q] / grad_ls_norm;

                heat_source_vector_local[q] =
                  local_compute_interfacial_heat_source(heat_source_eval.quadrature_point(q),
                                                        laser_position,
                                                        laser_power,
                                                        normal_at_q[q],
                                                        delta_value,
                                                        ls_heaviside_at_q[q]);
              }

            scratch_data.get_constraint(temp_dof_idx)
              .distribute_local_to_global(heat_source_vector_local,
                                          local_dof_indices,
                                          heat_source_vector);
          }
      }

    heat_source_vector.compress(VectorOperation::add);
    heat_source_vector_multiplicity.compress(VectorOperation::add);

    /*
     * average the nodally assembled values to smoothen discontinuous gradients of
     * the level set field
     */
    for (unsigned int i = 0; i < heat_source_vector_multiplicity.locally_owned_size(); ++i)
      if (heat_source_vector_multiplicity.local_element(i) > 1.0)
        heat_source_vector.local_element(i) /= heat_source_vector_multiplicity.local_element(i);

    heat_source_vector.zero_out_ghost_values();
    level_set_heaviside.zero_out_ghost_values();
    if (normal_vector)
      normal_vector->zero_out_ghost_values();
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::power_density_volumetric(const double radius, const double power) const
  {
    const double s          = radius / data.laser_beam_radius;
    const double peak_power = power * vol_peak_power_density_factor;
    return peak_power * std::exp(-2. * s * s);
  }

  template <int dim>
  double
  LaserHeatSourceGauss<dim>::power_density_interfacial(const double radius,
                                                       const double power) const
  {
    const double s          = radius / data.laser_beam_radius;
    const double peak_power = power * surf_peak_power_density_factor;
    return peak_power * std::exp(-2. * s * s);
  }

  template class LaserHeatSourceGauss<1>;
  template class LaserHeatSourceGauss<2>;
  template class LaserHeatSourceGauss<3>;
} // namespace MeltPoolDG::Heat
