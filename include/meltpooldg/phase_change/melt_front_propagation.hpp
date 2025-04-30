#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/material.hpp>
#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <vector>


namespace MeltPoolDG::MeltPool
{
  template <int dim, typename number>
  class MeltFrontPropagation
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    const ScratchData<dim, dim, number> &scratch_data;
    /**
     *  Parameters
     */
    const MeltPoolData<number> &mp_data;

    // melting/solidification
    Material<number> melting_solidification;
    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData object is selected. This is important when
     * ScratchData holds multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int phase_fraction_dof_idx;
    const unsigned int ls_dof_idx;
    const unsigned int reinit_dof_idx;
    const unsigned int reinit_no_solid_dof_idx;
    const unsigned int flow_vel_dof_idx;
    const unsigned int flow_vel_no_solid_dof_idx;
    const unsigned int heat_hanging_nodes_dof_idx;
    const VectorType  &temperature;

    /*
     * DoF vectors
     */
    VectorType solid;
    VectorType liquid;

  public:
    MeltFrontPropagation(const ScratchData<dim, dim, number> &scratch_data_in,
                         const Parameters<number>            &data_in,
                         const unsigned int                   phase_fraction_dof_idx_in,
                         const unsigned int                   ls_dof_idx_in,
                         const VectorType                    &temperature_in,
                         const unsigned int                   reinit_dof_idx_in,
                         const unsigned int                   reinit_no_solid_dof_idx_in,
                         const unsigned int                   flow_vel_dof_idx_in,
                         const unsigned int                   flow_vel_no_solid_dof_idx_in,
                         const unsigned int                   heat_hanging_nodes_dof_idx_in);

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
    attach_vectors(std::vector<VectorType *> &vectors);

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

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
    number
    compute_solid_fraction(const number temperature) const;

    dealii::VectorizedArray<number>
    compute_solid_fraction(const dealii::VectorizedArray<number> &temperature) const;

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
      const dealii::DoFHandler<dim>           &flow_dof_handler,
      const dealii::AffineConstraints<number> &flow_constraints_no_solid,
      dealii::AffineConstraints<number>       &flow_constraints);

    /**
     *  The reinitialization constraints are modified such that they are zero in solid
     *  regions. This means, the level set field is not modified due to reinitialization
     *  in the solid regions.
     */
    void
    ignore_reinitialization_in_solid_regions(
      const dealii::DoFHandler<dim>           &level_set_dof_handler,
      const dealii::AffineConstraints<number> &reinit_dirichlet_constraints_no_solid,
      dealii::AffineConstraints<number>       &reinit_dirichlet_constraints);
  };
} // namespace MeltPoolDG::MeltPool
