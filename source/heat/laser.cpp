#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <meltpooldg/heat/laser.hpp>
#include <meltpooldg/heat/laser_heat_source_gauss.hpp>
#include <meltpooldg/heat/laser_heat_source_gusarov.hpp>
#include <meltpooldg/heat/laser_heat_source_uniform.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserOperation<dim>::LaserOperation(ScratchData<dim>         &scratch_data_in,
                                      const Parameters<double> &data_in,
                                      const VectorType         *heaviside_in,
                                      const unsigned int        hs_dof_idx_in)
    : scratch_data(scratch_data_in)
    , laser_data(data_in.laser)
    , material(data_in.material)
    , laser_position(
        UtilityFunctions::to_point<dim>(laser_data.center.begin(), laser_data.center.end()))
  {
    /*
     * Factory for the laser heat source model
     */
    if (laser_data.heat_source_model == LaserHeatSourceModel::Gusarov)
      {
        laser_heat_source_operation =
          std::make_shared<Heat::LaserHeatSourceGusarov<dim>>(laser_data.gusarov);
      }
    else if (laser_data.heat_source_model == LaserHeatSourceModel::Gauss)
      {
        laser_heat_source_operation = std::make_shared<Heat::LaserHeatSourceGauss<dim>>(
          laser_data.gauss,
          material.two_phase_properties_transition_type,
          laser_data.delta_approximation_phase_weighted);
      }
    else if (laser_data.heat_source_model == LaserHeatSourceModel::uniform)
      {
        laser_heat_source_operation = std::make_shared<Heat::LaserHeatSourceUniform<dim>>(
          laser_data.delta_approximation_phase_weighted);
      }
    else if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
      {
        AssertThrow(heaviside_in, ExcMessage("The RTE laser model requires a heaviside!"));

        rte_dof_handler.reinit(scratch_data.get_dof_handler(hs_dof_idx_in).get_triangulation());

        scratch_data_in.attach_dof_handler(rte_dof_handler);
        scratch_data_in.attach_dof_handler(rte_dof_handler);

        rte_dof_idx = scratch_data_in.attach_constraint_matrix(rte_constraints_dirichlet);
        rte_hanging_nodes_dof_idx =
          scratch_data_in.attach_constraint_matrix(rte_hanging_node_constraints);

        if (data_in.base.do_simplex)
          rte_quad_idx =
            scratch_data_in.attach_quadrature(QGaussSimplex<dim>(data_in.base.n_q_points_1d));
        else
          rte_quad_idx = scratch_data_in.attach_quadrature(QGauss<dim>(data_in.base.n_q_points_1d));

        rte_operation = std::make_shared<RadiativeTransport::RadiativeTransportOperation<dim>>(
          scratch_data,
          data_in.rte,
          *heaviside_in,
          rte_dof_idx,
          rte_hanging_nodes_dof_idx,
          rte_quad_idx,
          hs_dof_idx_in);
      }
    else
      AssertThrow(laser_data.heat_source_model == LaserHeatSourceModel::Analytical,
                  ExcMessage("No requested laser model found. Please specify the "
                             "heat source model in the laser section of the input parameters."));
  }

  template <int dim>
  void
  LaserOperation<dim>::distribute_dofs(const BaseData<double> &base_data)
  {
    if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
      {
        if (base_data.do_simplex)
          rte_dof_handler.distribute_dofs(FE_SimplexP<dim>(base_data.degree));
        else
          rte_dof_handler.distribute_dofs(FE_Q<dim>(base_data.degree));
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::setup_constraints(
    ScratchData<dim> &mutable_scratch_data,
    const std::function<const DirichletBoundaryConditions<dim> &(const std::string &)>
                                          &dirichlet_bc,
    const PeriodicBoundaryConditions<dim> &periodic_bc)
  {
    if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
      {
        rte_operation->setup_constraints(mutable_scratch_data,
                                         dirichlet_bc("intensity"),
                                         periodic_bc,
                                         rte_dof_idx,
                                         rte_hanging_nodes_dof_idx,
                                         true /*set_inhomogeneities*/);
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::distribute_constraints()
  {
    if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
      {
        rte_operation->distribute_constraints();
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::reinit()
  {
    if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
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
    if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
      {
        data.emplace_back(&rte_dof_handler, [&](std::vector<VectorType *> &vectors) {
          rte_operation->attach_vectors(vectors);
        });
      }
  }

  template <int dim>
  void
  LaserOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
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
    print();
  }

  template <int dim>
  void
  LaserOperation<dim>::move_laser(const double dt)
  {
    // 0) update current time
    current_time += dt;
    // 1) compute the current center of the laser beam
    if (laser_data.do_move)
      laser_position[0] += laser_data.scan_speed * dt;
    // 2) update intensity of the laser
    compute_laser_intensity();

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
    return laser_intensity * laser_data.power;
  }

  template <int dim>
  void
  LaserOperation<dim>::compute_laser_intensity()
  {
    if (laser_data.power_over_time == "ramp")
      {
        AssertThrow(
          laser_data.power_end_time > laser_data.power_start_time,
          ExcMessage(
            "For the temporal ramp distribution of the laser power,"
            " the parameter laser power end time must be larger than laser power start time."));
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
    if (laser_data.heat_source_model == LaserHeatSourceModel::RTE)
      {
        rte_operation->solve();
        rte_operation->compute_heat_source(heat_source, temp_hanging_nodes_dof_idx, zero_out);
        return;
      }

    switch (laser_data.impact_type)
      {
          case LaserImpactType::volumetric: {
            laser_heat_source_operation->compute_volumetric_heat_source(heat_source,
                                                                        scratch_data,
                                                                        temp_hanging_nodes_dof_idx,
                                                                        get_laser_power(),
                                                                        get_laser_position(),
                                                                        zero_out);
            break;
          }
          case LaserImpactType::interface: {
            laser_heat_source_operation->compute_interfacial_heat_source(heat_source,
                                                                         scratch_data,
                                                                         temp_hanging_nodes_dof_idx,
                                                                         get_laser_power(),
                                                                         get_laser_position(),
                                                                         level_set_as_heaviside,
                                                                         ls_dof_idx,
                                                                         zero_out,
                                                                         normal_vector,
                                                                         normal_dof_idx);
            break;
          }
          case LaserImpactType::interface_sharp: {
            laser_heat_source_operation->compute_interfacial_heat_source_sharp(
              heat_user_rhs,
              scratch_data,
              temp_hanging_nodes_dof_idx,
              get_laser_power(),
              get_laser_position(),
              level_set_as_heaviside,
              ls_dof_idx,
              zero_out,
              normal_vector,
              normal_dof_idx);
            break;
          }
          case LaserImpactType::interface_sharp_conforming: {
            laser_heat_source_operation->compute_interfacial_heat_source_sharp_conforming(
              heat_user_rhs,
              scratch_data,
              temp_hanging_nodes_dof_idx,
              temp_quad_idx,
              get_laser_power(),
              get_laser_position(),
              level_set_as_heaviside,
              ls_dof_idx,
              zero_out,
              normal_vector,
              normal_dof_idx);
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
