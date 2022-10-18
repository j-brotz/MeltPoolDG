/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, November 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser.hpp>
#include <meltpooldg/heat/laser_analytical_temperature_field.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG
{
  namespace MeltPool
  {
    using namespace dealii;

    template <int dim>
    class MeltPoolOperation
    {
    private:
      using VectorType = LinearAlgebra::distributed::Vector<double>;

      const ScratchData<dim> &scratch_data;
      /**
       *  Parameters
       */
      const MeltPoolData<double> mp_data;
      const MaterialData<double> material;
      const bool                 do_mushy_zone;

      std::shared_ptr<Heat::LaserOperation<dim>>      laser_operation;
      std::shared_ptr<RecoilPressureOperation<dim>>   recoil_pressure_operation;
      std::shared_ptr<Heat::LaserHeatSourceBase<dim>> laser_heat_source_operation;
      std::shared_ptr<Heat::LaserAnalyticalTemperatureField<dim>>
        laser_analytical_temperature_field;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      const unsigned int ls_dof_idx;
      const unsigned int reinit_dof_idx;
      const unsigned int reinit_no_solid_dof_idx;
      const unsigned int flow_vel_dof_idx;
      const unsigned int flow_vel_no_solid_dof_idx;
      const unsigned int flow_vel_quad_idx;
      const unsigned int temp_hanging_nodes_dof_idx;
      VectorType *       temperature;

      /*
       * DoF vectors
       */
      VectorType solid;
      VectorType liquid;

    public:
      MeltPoolOperation(const ScratchData<dim> &  scratch_data_in,
                        const Parameters<double> &data_in,
                        const bool                do_recoil_pressure,
                        const unsigned int        ls_dof_idx_in,
                        VectorType *              temperature,
                        const unsigned int        reinit_dof_idx_in,
                        const unsigned int        reinit_no_solid_dof_idx_in,
                        const unsigned int        flow_vel_dof_idx_in,
                        const unsigned int        flow_vel_no_solid_dof_idx_in,
                        const unsigned int        flow_vel_quad_idx_in,
                        const unsigned int        flow_pressure_hanging_nodes_dof_idx,
                        const unsigned int        temp_dof_idx_in,
                        const unsigned int        temp_hanging_nodes_dof_idx_in,
                        const double              start_time_in);

      void
      set_initial_condition(const VectorType &level_set_as_heaviside, VectorType &level_set);

      void
      compute_heat_source(VectorType &           heat_source,
                          VectorType &           user_rhs,
                          const VectorType &     level_set_as_heaviside,
                          const BlockVectorType &normal_vector,
                          const unsigned int     normal_dof_idx,
                          const double &         dt,
                          const bool             zero_out = true);

      void
      compute_melt_front_propagation(const VectorType &level_set_as_heaviside);

      void
      compute_force_flow_rhs(VectorType &      vel_force_rhs,
                             const VectorType &level_set_as_heaviside,
                             const VectorType &temperature_interface,
                             const VectorType &evaporative_mass_flux,
                             const bool        zero_out = false) const;

      void
      reinit();

      /*
       * attach functions
       */
      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

      void
      attach_output_vectors(GenericDataOut<dim> &data_out) const;

      void
      distribute_constraints();

      /*
       * getter functions
       */
      const VectorType &
      get_solid() const;

      const VectorType &
      get_liquid() const;

      VectorizedArray<double>
      compute_solid_fraction(const VectorizedArray<double> &current_temperature) const;

    private:
      void
      make_constraints_in_spatially_fixed_solid_domain();


      /**
       * This function determines the solid fraction for a given pair of temperature (T) and
       * level-set-heaviside values (phi) as follows:
       *
       *  ______________________________________________________________________
       * |         |                                                           |
       * | phase   |  solid fraction     phi        T                          |
       * |_________|___________________________________________________________|
       * |         |                                                           |
       * | solid   |      1.0             1         T <= T_solidus             |
       * |         |                                                           |
       * |         |     T_l  - T                                              |
       * | mushy   |    ----------        1         T_solidus < T < T_liquidus |
       * |         |     T_l  - T_s                                            |
       * |         |                                                           |
       * | liquid  |       0.0            1         T >= T_liquidus            |
       * |         |                                                           |
       * | gas     |       0.0            0         independent                |
       * |_________|___________________________________________________________|
       *
       *
       * If no mushy zone is prescribed, then T_solidus = T_liquidus = T_melting.
       *
       */
      void
      compute_solid_and_liquid_phases(const VectorType &level_set_as_heaviside);

      /**
       *  The constraints of the flow velocity are modified such that they are zero in solid
       *  regions.
       */
      void
      set_flow_field_in_solid_regions_to_zero(
        const DoFHandler<dim> &          flow_dof_handler,
        const AffineConstraints<double> &flow_constraints_no_solid,
        AffineConstraints<double> &      flow_constraints);

      /**
       *  The reinitialization constraints are modified such that they are zero in solid
       *  regions. This means, the level set field is not modified due to reinitialization
       *  in the solid regions.
       */
      void
      ignore_reinitialization_in_solid_regions(
        const DoFHandler<dim> &          level_set_dof_handler,
        const AffineConstraints<double> &reinit_dirichlet_constraints_no_solid,
        AffineConstraints<double> &      reinit_dirichlet_constraints);

      /**
       * This function returns the solid fraction from a linear interpolation between the solidus
       * and liquidus temperatures, i.e. 0 (T>=T_liquidus) and 1 (T<=T_solidus).
       */
      double
      compute_solid_fraction(double temeprature) const;
    };
  } // namespace MeltPool
} // namespace MeltPoolDG
