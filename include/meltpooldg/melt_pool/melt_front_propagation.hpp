/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, November 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG
{
  namespace MeltPool
  {
    using namespace dealii;

    template <int dim>
    class MeltFrontPropagation
    {
    private:
      using VectorType = LinearAlgebra::distributed::Vector<double>;

      const ScratchData<dim> &scratch_data;
      /**
       *  Parameters
       */
      const MeltPoolData<double> mp_data;

      // melting/solidification
      Material<double> melting_solidification;
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
      const unsigned int temp_hanging_nodes_dof_idx;
      const VectorType  &temperature;

      /*
       * DoF vectors
       */
      VectorType solid;
      VectorType liquid;

    public:
      MeltFrontPropagation(const ScratchData<dim>   &scratch_data_in,
                           const Parameters<double> &data_in,
                           const unsigned int        ls_dof_idx_in,
                           const VectorType         &temperature,
                           const unsigned int        reinit_dof_idx_in,
                           const unsigned int        reinit_no_solid_dof_idx_in,
                           const unsigned int        flow_vel_dof_idx_in,
                           const unsigned int        flow_vel_no_solid_dof_idx_in,
                           const unsigned int        temp_hanging_nodes_dof_idx_in);

      void
      set_initial_condition(const VectorType &level_set_as_heaviside, VectorType &level_set);

      void
      compute_melt_front_propagation(const VectorType &level_set_as_heaviside);

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

      /**
       * This function returns the solid fraction from a linear interpolation between the solidus
       * and liquidus temperatures, i.e. 0 (T>=T_liquidus) and 1 (T<=T_solidus).
       */
      double
      compute_solid_fraction(const double temperature) const;

      VectorizedArray<double>
      compute_solid_fraction(const VectorizedArray<double> &temperature) const;

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
        const DoFHandler<dim>           &flow_dof_handler,
        const AffineConstraints<double> &flow_constraints_no_solid,
        AffineConstraints<double>       &flow_constraints);

      /**
       *  The reinitialization constraints are modified such that they are zero in solid
       *  regions. This means, the level set field is not modified due to reinitialization
       *  in the solid regions.
       */
      void
      ignore_reinitialization_in_solid_regions(
        const DoFHandler<dim>           &level_set_dof_handler,
        const AffineConstraints<double> &reinit_dirichlet_constraints_no_solid,
        AffineConstraints<double>       &reinit_dirichlet_constraints);
    };
  } // namespace MeltPool
} // namespace MeltPoolDG
