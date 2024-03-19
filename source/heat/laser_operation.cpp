#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <meltpooldg/heat/laser_intensity_profiles.hpp>
#include <meltpooldg/heat/laser_operation.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserOperation<dim>::LaserOperation(ScratchData<dim>                      &scratch_data_in,
                                      const PeriodicBoundaryConditions<dim> &periodic_bc_in,
                                      const Parameters<double>              &data_in,
                                      const VectorType                      *heaviside_in,
                                      const unsigned int                     hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , periodic_bc(periodic_bc_in)
    , laser_data(data_in.laser)
    , laser_position(laser_data.get_starting_position<dim>())
  {
    AssertThrow(
      laser_data.power_end_time > laser_data.power_start_time,
      ExcMessage(
        "For the temporal ramp distribution of the laser power,"
        " the parameter laser power end time must be larger than laser power start time."));

    if (laser_data.model == LaserModelType::analytical_temperature)
      return;

    /*
     * Factory for the intensity profile
     */
    switch (laser_data.intensity_profile)
      {
          case LaserIntensityProfileType::uniform: {
            intensity_profile = std::make_shared<UniformIntensityProfile<dim, double>>(
              [&]() { return get_laser_power(); });
            break;
          }
          case LaserIntensityProfileType::Gauss: {
            if (laser_data.model == LaserModelType::volumetric)
              intensity_profile = std::make_shared<GaussVolumetricIntensityProfile<dim, double>>(
                laser_data.radius,
                [&]() -> const GaussVolumetricIntensityProfile<dim, double>::State {
                  return {.power = get_laser_power(), .position = get_laser_position()};
                });
            else
              intensity_profile = std::make_shared<GaussProjectionIntensityProfile<dim, double>>(
                laser_data.radius,
                laser_data.get_direction<dim>(),
                [&]() -> const GaussProjectionIntensityProfile<dim, double>::State {
                  return {.power = get_laser_power(), .position = get_laser_position()};
                });
            break;
          }
          case LaserIntensityProfileType::Gusarov: {
            intensity_profile = std::make_shared<GusarovIntensityProfile<dim, double>>(
              laser_data.gusarov,
              laser_data.radius,
              [&]() -> const GusarovIntensityProfile<dim, double>::State {
                return {.power = get_laser_power(), .position = get_laser_position()};
              });
            break;
          }
        default:
          AssertThrow(false, ExcNotImplemented());
      }

    /*
     * Factory for the laser model
     */
    switch (laser_data.model)
      {
          case LaserModelType::volumetric: {
            laser_heat_source_operation_volumetric =
              std::make_unique<Heat::LaserHeatSourceVolumetric<dim>>(intensity_profile);
            break;
          }
        case LaserModelType::interface_projection_regularized:
        case LaserModelType::interface_projection_sharp:
          case LaserModelType::interface_projection_sharp_conforming: {
            laser_heat_source_operation_projection =
              std::make_unique<Heat::LaserHeatSourceProjectionBased<dim>>(
                laser_data,
                intensity_profile,
                data_in.material.two_phase_fluid_properties_transition_type !=
                  TwoPhaseFluidPropertiesTransitionType::sharp,
                laser_data.delta_approximation_phase_weighted);
            break;
          }
          case LaserModelType::RTE: {
            AssertThrow(heaviside_in, ExcMessage("The RTE laser model requires a heaviside!"));

            rte_dof_handler = std::make_unique<DoFHandler<dim>>(
              scratch_data.get_dof_handler(hs_dof_idx_in).get_triangulation());

            scratch_data_in.attach_dof_handler(*rte_dof_handler);
            scratch_data_in.attach_dof_handler(*rte_dof_handler);

            rte_constraints_dirichlet    = std::make_unique<AffineConstraints<double>>();
            rte_hanging_node_constraints = std::make_unique<AffineConstraints<double>>();
            rte_dof_idx = scratch_data_in.attach_constraint_matrix(*rte_constraints_dirichlet);
            rte_hanging_nodes_dof_idx =
              scratch_data_in.attach_constraint_matrix(*rte_hanging_node_constraints);

            rte_dirichlet_boundary_condition = std::make_unique<DirichletBoundaryConditions<dim>>();
            AssertThrow(
              laser_data.rte_boundary_id != numbers::invalid_boundary_id,
              ExcMessage(
                "The RTE laser model requires the RTE boundary id to be set by the simulation!"));
            if (data_in.output.paraview.print_boundary_id)
              Journal::print_line(scratch_data.get_pcout(),
                                  "RTE boundary id = " + std::to_string(laser_data.rte_boundary_id),
                                  "laser");
            rte_dirichlet_boundary_condition->attach(laser_data.rte_boundary_id, intensity_profile);

            if (data_in.base.fe.type == FiniteElementType::FE_SimplexP)
              rte_quad_idx = scratch_data_in.attach_quadrature(
                QGaussSimplex<dim>(data_in.base.fe.get_n_q_points()));
            else
              rte_quad_idx =
                scratch_data_in.attach_quadrature(QGauss<dim>(data_in.base.fe.get_n_q_points()));

            rte_operation = std::make_unique<RadiativeTransport::RadiativeTransportOperation<dim>>(
              scratch_data,
              data_in.rte,
              laser_data.get_direction<dim>(),
              *heaviside_in,
              rte_dof_idx,
              rte_hanging_nodes_dof_idx,
              rte_quad_idx,
              hs_dof_idx_in);
            break;
          }
        default:
          AssertThrow(laser_data.model == LaserModelType::analytical_temperature,
                      ExcMessage(
                        "No requested laser model found. Please specify the "
                        "heat source model in the laser section of the input parameters."));
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::distribute_dofs(const FiniteElementData &fe_data)
  {
    if (laser_data.model == LaserModelType::RTE)
      FiniteElementUtils::distribute_dofs<dim, 1>(fe_data, *rte_dof_handler);
  }

  template <int dim>
  void
  LaserOperation<dim>::setup_constraints()
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        auto mutable_scratch_data = const_cast<ScratchData<dim> &>(scratch_data);
        rte_operation->setup_constraints(mutable_scratch_data,
                                         *rte_dirichlet_boundary_condition,
                                         periodic_bc);
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::distribute_constraints()
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        rte_operation->distribute_constraints();
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::reinit()
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        rte_operation->reinit();
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::attach_vectors(
    std::vector<
      std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>> &data)
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        data.emplace_back(rte_dof_handler.get(), [&](std::vector<VectorType *> &vectors) {
          rte_operation->attach_vectors(vectors);
        });
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    if (laser_data.model == LaserModelType::RTE)
      {
        rte_operation->attach_output_vectors(data_out);
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::reset(const double start_time)
  {
    current_time = start_time;
    compute_laser_intensity();
    laser_position = laser_data.template get_starting_position<dim>();

    // update laser intensity
    if (intensity_profile)
      intensity_profile->set_time(current_time);

    print();
  }

  template <int dim>
  void
  LaserOperation<dim>::move_laser(const double dt)
  {
    bool intensity_or_laser_position_has_changed = false;

    // 0) update current time
    current_time += dt;
    // 1) compute the current center of the laser beam
    if (laser_data.do_move && laser_data.scan_speed != 0.0)
      {
        laser_position[0] += laser_data.scan_speed * dt;
        intensity_or_laser_position_has_changed = true;
      }
    // 2) compute intensity
    intensity_or_laser_position_has_changed =
      compute_laser_intensity() || intensity_or_laser_position_has_changed;
    // 3) update intensity according to the current time if required
    if (intensity_or_laser_position_has_changed)
      {
        if (intensity_profile)
          intensity_profile->set_time(current_time);
        setup_constraints();
      }

    print();
  }

  template <int dim>
  const Point<dim> &
  LaserOperation<dim>::get_laser_position() const
  {
    return laser_position;
  }

  template <int dim>
  double
  LaserOperation<dim>::get_laser_power() const
  {
    return current_power;
  }

  template <int dim>
  bool
  LaserOperation<dim>::compute_laser_intensity()
  {
    const double previous_intensity = laser_intensity;
    if (laser_data.power_over_time == "ramp")
      {
        laser_intensity = (current_time - laser_data.power_start_time) /
                          (laser_data.power_end_time - laser_data.power_start_time);
        laser_intensity = std::min(std::max(0.0, laser_intensity), 1.0);
      }
    else if (laser_data.power_over_time == "constant")
      {
        if (current_time >= laser_data.power_end_time)
          {
            laser_intensity = 0.0;
          }
        else
          laser_intensity = 1.0;
      }
    else
      AssertThrow(false, ExcNotImplemented());

    current_power = laser_data.power * laser_intensity;

    return previous_intensity != laser_intensity;
  }

  /* TODO: add function parameters*/
  template <int dim>
  void
  LaserOperation<dim>::compute_heat_source(VectorType            &heat_source,
                                           VectorType            &heat_user_rhs,
                                           const VectorType      &level_set_as_heaviside,
                                           const unsigned int     ls_dof_idx,
                                           const unsigned int     temp_hanging_nodes_dof_idx,
                                           const unsigned int     temp_quad_idx,
                                           const bool             zero_out,
                                           const BlockVectorType *normal_vector,
                                           const unsigned int     normal_dof_idx) const
  {
    switch (laser_data.model)
      {
          case LaserModelType::volumetric: {
            laser_heat_source_operation_volumetric->compute_volumetric_heat_source(
              heat_source, scratch_data, temp_hanging_nodes_dof_idx, zero_out);
            break;
          }
          case LaserModelType::interface_projection_regularized: {
            laser_heat_source_operation_projection->compute_interfacial_heat_source(
              heat_source,
              scratch_data,
              temp_hanging_nodes_dof_idx,
              level_set_as_heaviside,
              ls_dof_idx,
              zero_out,
              normal_vector,
              normal_dof_idx);
            break;
          }
          case LaserModelType::interface_projection_sharp: {
            laser_heat_source_operation_projection->compute_interfacial_heat_source_sharp(
              heat_user_rhs,
              scratch_data,
              temp_hanging_nodes_dof_idx,
              level_set_as_heaviside,
              ls_dof_idx,
              zero_out,
              normal_vector,
              normal_dof_idx);
            break;
          }
          case LaserModelType::interface_projection_sharp_conforming: {
            laser_heat_source_operation_projection
              ->compute_interfacial_heat_source_sharp_conforming(heat_user_rhs,
                                                                 scratch_data,
                                                                 temp_hanging_nodes_dof_idx,
                                                                 temp_quad_idx,
                                                                 level_set_as_heaviside,
                                                                 ls_dof_idx,
                                                                 zero_out,
                                                                 normal_vector,
                                                                 normal_dof_idx);
            break;
          }
          case LaserModelType::RTE: {
            rte_operation->solve();
            rte_operation->compute_heat_source(heat_source, temp_hanging_nodes_dof_idx, zero_out);
            break;
          }
          default: {
            AssertThrow(false, ExcMessage("Laser impact type not implemented here! Abort..."));
            break;
          }
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::print() const
  {
    std::ostringstream str;
    str << "current laser position: " << laser_position;
    Journal::print_line(scratch_data.get_pcout(), str.str(), "laser");
    str.str("");
    str << "current laser intensity: " << laser_intensity;
    Journal::print_line(scratch_data.get_pcout(), str.str(), "laser");
  }

  template class LaserOperation<1>;
  template class LaserOperation<2>;
  template class LaserOperation<3>;
} // namespace MeltPoolDG::Heat
