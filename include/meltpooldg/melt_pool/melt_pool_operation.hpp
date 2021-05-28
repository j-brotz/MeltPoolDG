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
#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/heat/laser.hpp>
#include <meltpooldg/heat/laser_heat_source.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
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

      std::shared_ptr<Heat::LaserOperation<dim>>    laser_operation;
      std::shared_ptr<RecoilPressureOperation<dim>> recoil_pressure_operation;
      std::shared_ptr<Heat::LaserHeatSource<dim>>   laser_heat_source_operation;
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
      std::shared_ptr<Heat::HeatTransferOperation<dim>> heat_operation;

    public:
      MeltPoolOperation(const std::shared_ptr<ScratchData<dim>> &       scratch_data_in,
                        const std::shared_ptr<BoundaryConditions<dim>> &heat_bc,
                        const Parameters<double> &                      data_in,
                        const unsigned int                              ls_dof_idx_in,
                        const unsigned int                              reinit_dof_idx_in,
                        const unsigned int                              flow_vel_dof_idx_in,
                        const unsigned int                              flow_vel_quad_idx_in,
                        const unsigned int                              temp_dof_idx_in,
                        const unsigned int                              temp_quad_idx_in,
                        const double                                    start_time_in,
                        bool                                            do_recoil_pressure = true);

      void
      set_initial_condition(const VectorType &level_set_as_heaviside,
                            VectorType &      level_set,
                            const double &    density_gas,
                            const double &    density_liquid);

      void
      solve(VectorType &      vel_force_rhs,
            const VectorType &level_set_as_heaviside,
            const double &    density_gas,
            const double &    density_liquid,
            const double &    dt);

      void
      reinit();

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

      void
      attach_output_vectors(GenericDataOut<dim> &data_out) const;

      void
      distribute_constraints();

      const VectorType &
      get_temperature() const;

      const VectorType &
      get_solid() const;

      const VectorType &
      get_liquid() const;

    private:
      void
      make_constraints_in_spatially_fixed_solid_domain();


      void
      compute_solid_and_liquid_phases(const VectorType &level_set_as_heaviside);

      /**
       *  The constraints of the flow velocity are modified such that they are zero in solid
       *  regions.
       */
      void
      set_flow_field_in_solid_regions_to_zero(const DoFHandler<dim> &    flow_dof_handler,
                                              AffineConstraints<double> &flow_constraints);

      /**
       *  The level set constraints are modified such that they are zero in solid
       *  regions.
       */
      void
      remove_the_level_set_from_solid_regions(const DoFHandler<dim> &    level_set_dof_handler,
                                              AffineConstraints<double> &level_set_constraints);

      void
      set_melt_pool_parameters(const Parameters<double> &data_in);

      /**
       *  This function determines for a given pair of level set and temperature values, whether
       * it characterizes a solid state. For this purpose, it is assumed that phi=1 in the liquid
       * AND the solid domain and phi=0 in the gaseous domain. Thus, if phi=1 and the temperature
       * is SMALLER than the fusion point, a solid phase is met.
       */
      bool
      is_solid_region(const double phi_liquid, const double temperature);

      /**
       *  This function determines for a given pair of level set and temperature values, whether
       * it characterizes a liquid state. For this purpose, it is assumed that phi=1 in the liquid
       * AND the solid domain and phi=0 in the gaseous domain. Thus, if phi=1 and the temperature
       * is LARGER than the fusion point, a liquid phase is met.
       */
      bool
      is_liquid_region(const double phi_liquid, const double temperature);
    };
  } // namespace MeltPool
} // namespace MeltPoolDG
