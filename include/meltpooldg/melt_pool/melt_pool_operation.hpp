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
#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/heat/laser.hpp>
#include <meltpooldg/heat/laser_analytical_temperature_field.hpp>
#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

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

      std::shared_ptr<ScratchData<dim>> scratch_data;
      /**
       *  Parameters
       */
      MeltPoolData<double> mp_data;
      MaterialData<double> material;
      const bool           do_mushy_zone;

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
      MeltPoolOperation(const std::shared_ptr<ScratchData<dim>> &scratch_data_in,
                        const Parameters<double> &               data_in,
                        const unsigned int                       ls_dof_idx_in,
                        VectorType *                             temperature,
                        const unsigned int                       reinit_dof_idx_in,
                        const unsigned int                       flow_vel_dof_idx_in,
                        const unsigned int                       flow_vel_no_solid_dof_idx_in,
                        const unsigned int                       flow_vel_quad_idx_in,
                        const unsigned int                       temp_hanging_nodes_dof_idx_in,
                        const double                             start_time_in);

      void
      set_initial_condition(const VectorType &level_set_as_heaviside, VectorType &level_set);

      void
      compute_heat_source(VectorType &      heat_source,
                          const VectorType &level_set_as_heaviside,
                          const double &    dt,
                          const bool        zero_out = true);

      void
      compute_melt_front_propagation(const VectorType &level_set_as_heaviside);

      void
      compute_force_flow_rhs(VectorType &      vel_force_rhs,
                             const VectorType &level_set_as_heaviside,
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
      set_flow_field_in_solid_regions_to_zero(
        const DoFHandler<dim> &          flow_dof_handler,
        const AffineConstraints<double> &flow_constraints_no_solid,
        AffineConstraints<double> &      flow_constraints);

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
      double
      compute_solid_fraction(double ls_heaviside, double temperature) const;

      /**
       * This function determines the liquid fraction for a given pair of temperature and
       * level-set-heaviside values.
       *
       * The liquid fraction is determined analogously to compute_solid_fraction().
       */
      double
      compute_liquid_fraction(double ls_heaviside, double temperature) const;

      /**
       * This function return the solid fraction without being informed by the level-set.
       */
      double
      compute_solid_fraction_no_ls(double temeprature) const;
    };
  } // namespace MeltPool
} // namespace MeltPoolDG
