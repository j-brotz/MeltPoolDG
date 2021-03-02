/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, November 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/heat_equation/heat_operation.hpp>
#include <meltpooldg/heat_equation/laser.hpp>
#include <meltpooldg/heat_equation/laser_heat_source_gusarov.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace MeltPool
  {
    using namespace dealii;

    template <int dim>
    class MeltPoolOperation
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

      std::shared_ptr<ScratchData<dim>> scratch_data;
      /**
       *  Parameters
       */
      MeltPoolData<double> mp_data;

      std::shared_ptr<HeatEquation::LaserOperation<dim>>         laser_operation;
      std::shared_ptr<RecoilPressureOperation<dim>>              recoil_pressure_operation;
      std::shared_ptr<HeatEquation::LaserHeatSourceGusarov<dim>> laser_heat_source_operation;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      const unsigned int ls_dof_idx;
      const unsigned int reinit_dof_idx;
      const unsigned int flow_vel_dof_idx;
      const unsigned int flow_vel_quad_idx;
      const unsigned int temp_dof_idx;
      const unsigned int temp_quad_idx;

      /*
       * DoF vectors
       */
      VectorType solid;
      VectorType liquid;
      /*
       *  heat operation
       */
      std::shared_ptr<HeatEquation::HeatOperation<dim>> heat_operation;

    public:
      MeltPoolOperation(const std::shared_ptr<ScratchData<dim>> &scratch_data_in,
                        const Parameters<double> &               data_in,
                        const std::vector<types::boundary_id>    bc_radiation_in,
                        const std::vector<types::boundary_id>    bc_convection_in,
                        const unsigned int                       ls_dof_idx_in,
                        const unsigned int                       reinit_dof_idx_in,
                        const unsigned int                       flow_vel_dof_idx_in,
                        const unsigned int                       flow_vel_quad_idx_in,
                        const unsigned int                       temp_dof_idx_in,
                        const unsigned int                       temp_quad_idx_in,
                        const double                             start_time_in,
                        bool                                     do_recoil_pressure = true)
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
          std::make_shared<HeatEquation::LaserOperation<dim>>(*scratch_data, data_in.laser);
        /*
         *  Initialize the laser operation
         */
        laser_operation->set_initial_condition(start_time_in);
        /*
         *  initialize the heat operation class
         */
        heat_operation =
          std::make_shared<HeatEquation::HeatOperation<dim>>(*scratch_data,
                                                             data_in.heat,
                                                             temp_dof_idx,
                                                             temp_dof_idx, //@todo: hanging nodes
                                                             temp_quad_idx,
                                                             bc_radiation_in,
                                                             bc_convection_in);

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
        scratch_data->initialize_dof_vector(heat_operation->get_temperature(), temp_dof_idx);

        if (mp_data.temperature_formulation == "numerical")
          laser_heat_source_operation =
            std::make_shared<HeatEquation::LaserHeatSourceGusarov<dim>>(*scratch_data,
                                                                        data_in.laser.gusarov,
                                                                        temp_dof_idx);
      }

      void
      set_initial_condition(const VectorType &level_set_as_heaviside,
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

      void
      solve(VectorType &      vel_force_rhs,
            const VectorType &level_set_as_heaviside,
            const VectorType &curvature,
            const double &    surface_tension_coefficient,
            const double &    temperature_dependent_surface_tension_coefficient,
            const double &    surface_tension_reference_temperature,
            const double &    density_gas,
            const double &    density_liquid,
            const double &    dt)
      {
        // 0) move laser
        laser_operation->move_laser(dt);

        // 1) update temperature
        if (mp_data.temperature_formulation == "analytical")
          laser_operation->compute_analytical_temperature_field(
            level_set_as_heaviside,
            heat_operation->get_temperature(),
            temp_dof_idx,
            density_gas,
            density_liquid,
            mp_data,
            -1.0 /* level set value for gas @todo */);

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

        //     ... or evaporative flux
        //@todo

        //     ... temperature dependent surface tension
        if (temperature_dependent_surface_tension_coefficient > 0.0)
          Flow::SurfaceTensionOperation<dim>::compute_temperature_dependent_surface_tension(
            *scratch_data,
            vel_force_rhs,
            level_set_as_heaviside,
            curvature,
            heat_operation->get_temperature(),
            surface_tension_coefficient,
            temperature_dependent_surface_tension_coefficient,
            surface_tension_reference_temperature,
            ls_dof_idx,
            flow_vel_dof_idx,
            flow_vel_quad_idx,
            temp_dof_idx,
            false /*false means add to force vector*/);
      }

      void
      make_constraints_in_spatially_fixed_solid_domain()
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

      void
      reinit()
      {
        heat_operation->reinit();
        scratch_data->initialize_dof_vector(solid, temp_dof_idx);
        scratch_data->initialize_dof_vector(liquid, temp_dof_idx);
      }

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
      {
        heat_operation->attach_vectors(vectors);
        solid.update_ghost_values();
        liquid.update_ghost_values();
        vectors.push_back(&solid);
        vectors.push_back(&liquid);
      }

      void
      attach_output_vectors(DataOut<dim> &data_out) const
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

      void
      distribute_constraints()
      {
        heat_operation->distribute_constraints();
        scratch_data->get_constraint(temp_dof_idx).distribute(solid);
        scratch_data->get_constraint(temp_dof_idx).distribute(liquid);
      }

      const VectorType &
      get_temperature() const
      {
        return heat_operation->get_temperature();
      }

      const VectorType &
      get_solid() const
      {
        return solid;
      }

      const VectorType &
      get_liquid() const
      {
        return liquid;
      }

    private:
      void
      compute_solid_and_liquid_phases(const VectorType &level_set_as_heaviside)
      {
        (void)level_set_as_heaviside; // @todo: the level set will be needed later, when the thermal
                                      // coupling is completed
        heat_operation->get_temperature().update_ghost_values();

        const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(temp_dof_idx);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::map<types::global_dof_index, Point<dim>> support_points;
        DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                             scratch_data->get_dof_handler(temp_dof_idx),
                                             support_points);

        for (const auto &cell : scratch_data->get_dof_handler(temp_dof_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  solid[local_dof_indices[i]] =
                    is_solid_region(support_points[local_dof_indices[i]],
                                    heat_operation->get_temperature()[local_dof_indices[i]]);
                  liquid[local_dof_indices[i]] =
                    is_liquid_region(support_points[local_dof_indices[i]],
                                     heat_operation->get_temperature()[local_dof_indices[i]]);
                }
            }

        liquid.compress(VectorOperation::insert);
        solid.compress(VectorOperation::insert);

        scratch_data->get_constraint(temp_dof_idx).distribute(solid);
        scratch_data->get_constraint(temp_dof_idx).distribute(liquid);

        heat_operation->get_temperature().zero_out_ghosts();
      }

      /**
       *  The constraints of the flow velocity are modified such that they are zero in solid
       *  regions.
       */
      void
      set_flow_field_in_solid_regions_to_zero(const DoFHandler<dim> &    flow_dof_handler,
                                              AffineConstraints<double> &flow_constraints)
      {
        solid.update_ghost_values();

        IndexSet flow_locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(flow_dof_handler, flow_locally_relevant_dofs);

        AffineConstraints<double> solid_constraints;
        solid_constraints.reinit(flow_locally_relevant_dofs);

        FEValues<dim> flow_eval(scratch_data->get_mapping(),
                                flow_dof_handler.get_fe(),
                                Quadrature<dim>(
                                  flow_dof_handler.get_fe().get_unit_support_points()),
                                update_quadrature_points);

        FEValues<dim> solid_eval(scratch_data->get_mapping(),
                                 scratch_data->get_dof_handler(temp_dof_idx).get_fe(),
                                 Quadrature<dim>(
                                   flow_dof_handler.get_fe().get_unit_support_points()),
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

        solid.zero_out_ghosts();
      }
      /**
       *  The level set constraints are modified such that they are zero in solid
       *  regions.
       */
      void
      remove_the_level_set_from_solid_regions(const DoFHandler<dim> &    level_set_dof_handler,
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
        level_set_constraints.merge(
          solid_constraints, AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        level_set_constraints.close();
        solid.zero_out_ghosts();
      }

      void
      set_melt_pool_parameters(const Parameters<double> &data_in)
      {
        mp_data = data_in.mp;
      }

      /**
       *  This function determines for a given point, whether it belongs to the solid domain.
       *
       *  WARNING: All points above (component dim-1) the center point of the laser source are
       *  automatically identified as gaseous parts. Thus, this function has to be modified when the
       *  initial interface between the feedstock and the ambient gas is not planar.
       */
      bool
      is_solid_region(const Point<dim> point, const double temperature)
      {
        if (point[dim - 1] >= laser_operation->get_laser_position()[dim - 1])
          return false;
        else
          return (temperature >= mp_data.liquid.melting_point) ? false : true;
      }

      /*
       * check if a given point is liquid
       */
      bool
      is_liquid_region(const Point<dim> point, const double temperature)
      {
        if (point[dim - 1] >= laser_operation->get_laser_position()[dim - 1])
          return false;
        else
          return (temperature >= mp_data.liquid.melting_point) ? true : false;
      }
    };
  } // namespace MeltPool
} // namespace MeltPoolDG
