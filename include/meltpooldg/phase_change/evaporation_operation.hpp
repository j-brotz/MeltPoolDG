#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>
#include <meltpooldg/phase_change/evaporation_source_terms_base.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <memory>
#include <tuple>
#include <vector>

namespace MeltPoolDG::Evaporation
{
  /**
   *     This module computes for a given evaporative mass flux $\f\dot{m}\f$ the corresponding
   * interface velocity according to
   *
   *     \f[ \boldsymbol{n}\cfrac{\dot{m}}{\rho} \f]
   *
   *     with the normal vector \f$\boldsymbol{n}\f$, the evaporative mass flux \f$\dot{m}\f$
   *     and the density \f$\rho\f$ as well as the corresponding term in the mass balance
   *     equation of the incompressible Navier-Stokes formulation
   *
   *     \f[ \dot{m}\,(\frac{1}{\rho_l}-\frac{1}{\rho_g})\,\delta \f]
   *
   *     with the delta-function \f$\delta\f$.
   *
   */
  template <int dim, typename number>
  class EvaporationOperation
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const ScratchData<dim, dim, number> &scratch_data;
    /**
     *  parameters controlling the evaporation
     */
    const EvaporationData<number> &evapor_data;

    const MaterialData<number> &material_data;
    /**
     * references to solutions needed for the computation
     */
    const VectorType      &level_set_as_heaviside;
    const BlockVectorType &normal_vector;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int normal_dof_idx;
    const unsigned int evapor_vel_dof_idx;
    const unsigned int evapor_mass_flux_dof_idx;
    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int ls_quad_idx;
    /*
     * cut-off value for normalizing the normal vector field
     */
    const number tolerance_normal_vector;

    /**
     * evaporative mass flux
     */
    VectorType evaporative_mass_flux;
    /**
     * evaporation velocity at quadrature points
     */
    dealii::AlignedVector<dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>
      evaporation_velocities;
    /**
     * evaporation velocity due to evaporation and flow
     */
    VectorType evaporation_velocity;

    bool do_analytical_evaporative_mass_flux = false;

    std::shared_ptr<EvaporationModelBase<number>>                 evapor_model;
    std::shared_ptr<EvaporationMassFluxOperatorBase<dim, number>> evapor_mass_flux_operator;
    std::shared_ptr<EvaporationSourceTermsBase<dim, number>>      evapor_source_terms_operator;

  public:
    EvaporationOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                         const VectorType                    &level_set_as_heaviside_in,
                         const BlockVectorType               &normal_vector_in,
                         const EvaporationData<number>       &evapor_data_in,
                         const MaterialData<number>          &material_data_in,
                         const unsigned int                   normal_dof_idx_in,
                         const unsigned int                   evapor_vel_dof_idx_in,
                         const unsigned int                   evapor_vel_quad_idx_in,
                         const unsigned int                   evapor_mass_flux_dof_idx_in,
                         const unsigned int                   ls_hanging_nodes_dof_idx_in,
                         const unsigned int                   ls_quad_idx_in);

    /**
     * @brief Compute the evaporative mass flux DoF vector for the current evaporation model.
     *
     * This is the unified entry point:
     * - If the model is analytical (`do_analytical_evaporative_mass_flux == true`),
     *   it delegates to compute_analytical_evaporative_mass_flux().
     * - Otherwise, it requires a temperature field and a valid heat DoF index and
     *   delegates to compute_temperature_dependent_evaporative_mass_flux().
     *
     * The resulting `evaporative_mass_flux` vector is written in-place and its ghost
     * values are updated.
     *
     * @param time                Physical time at which to evaluate the mass flux (used by analytical models).
     * @param temperature         Pointer to the temperature field. Must be non-null for non-analytical models;
     *                            ignored for analytical models.
     * @param heat_no_bc_dof_idx  DoF index for the heat field **without BCs** (used by non-analytical models).
     *                            Must differ from `dealii::numbers::invalid_unsigned_int` when
     * required.
     *
     * @see compute_analytical_evaporative_mass_flux()
     * @see compute_temperature_dependent_evaporative_mass_flux()
     */

    void
    compute_evaporative_mass_flux(
      const number      &time,
      const VectorType  *temperature        = nullptr,
      const unsigned int heat_no_bc_dof_idx = dealii::numbers::invalid_unsigned_int);

    /**
     * @brief Compute the evaporative mass flux DoF vector for an analytical evaporation model.
     *
     * Evaluates the model locally at the given time and writes the result into
     * `evaporative_mass_flux`. Also prints a formatted norm to the journal and updates
     * ghost values.
     *
     * @param time  Physical time at which to evaluate the analytical model.
     *
     * @see compute_evaporative_mass_flux()
     */
    void
    compute_analytical_evaporative_mass_flux(const number &time);

    /**
     * @brief Compute the evaporative mass flux DoF vector for a temperature-dependent (non-analytical) model.
     *
     * Uses (or lazily constructs) an EvaporationMassFluxOperatorContinuous to compute the
     * distribution across the interface from the provided temperature field. Writes the
     * result into `evaporative_mass_flux`, prints a formatted norm to the journal, and
     * updates ghost values.
     *
     * @param temperature         Temperature field used to evaluate the evaporation model.
     * @param heat_no_bc_dof_idx  DoF index for the heat field **without BCs** (used for norm/evaluation context).
     *
     * @see compute_evaporative_mass_flux()
     */
    void
    compute_temperature_dependent_evaporative_mass_flux(const VectorType  &temperature,
                                                        const unsigned int heat_no_bc_dof_idx);

    void
    compute_evaporation_velocity();

    void
    compute_level_set_source_term(VectorType        &rhs,
                                  const unsigned int ls_dof_idx,
                                  const VectorType  &level_set,
                                  const unsigned int pressure_dof_idx);

    void
    compute_mass_balance_source_term(VectorType        &mass_balance_rhs,
                                     const unsigned int pressure_dof_idx,
                                     const unsigned int pressure_quad_idx,
                                     bool               zero_out);

    void
    register_surface_mesh(
      const std::vector<
        std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                   std::vector<dealii::Point<dim>> /*quad_points*/,
                   std::vector<number> /*weights*/
                   >> &surface_mesh_info);
    void
    reinit();
    /*
     * attach functions
     */
    void
    attach_dim_vectors(std::vector<VectorType *> &vectors);

    void
    attach_vectors(std::vector<VectorType *> &vectors);

    void
    distribute_constraints();

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;

    /*
     * getter functions
     */
    inline dealii::Tensor<1, dim, dealii::VectorizedArray<number>> *
    begin_evaporation_velocity(const unsigned int macro_cell);

    inline const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &
    begin_evaporation_velocity(const unsigned int macro_cell) const;

    const VectorType &
    get_velocity() const;

    VectorType &
    get_velocity();

    const VectorType &
    get_evaporative_mass_flux() const;

    VectorType &
    get_evaporative_mass_flux();
  };
} // namespace MeltPoolDG::Evaporation
