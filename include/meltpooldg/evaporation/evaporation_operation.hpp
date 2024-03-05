/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/evaporation/evaporation_mass_flux_operator_base.hpp>
#include <meltpooldg/evaporation/evaporation_model_base.hpp>
#include <meltpooldg/evaporation/evaporation_source_terms_base.hpp>
#include <meltpooldg/evaporation/recoil_pressure_data.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;
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
  template <int dim>
  class EvaporationOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim> &scratch_data;
    /**
     *  parameters controlling the evaporation
     */
    const EvaporationData<double> &evaporation_data;

    const MaterialData<double> &material;
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
    const double tolerance_normal_vector;
    /*
     * optional: temperature-dependent evaporation
     */
    mutable const VectorType *temperature;
    unsigned int              temp_dof_idx;

    // only needed if a time-dependent function is given
    mutable double time = numbers::invalid_double;
    /**
     * evaporative mass flux
     */
    VectorType evaporative_mass_flux;
    /**
     * evaporation velocity at quadrature points
     */
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaporation_velocities;
    /**
     * evaporation velocity due to evaporation and flow
     */
    VectorType evaporation_velocity;

    std::shared_ptr<EvaporationModelBase>                 evapor_model;
    std::shared_ptr<EvaporationMassFluxOperatorBase<dim>> evapor_mass_flux_operator;
    std::shared_ptr<EvaporationSourceTermsBase<dim>>      evapor_source_terms_operator;

  public:
    EvaporationOperation(const ScratchData<dim>        &scratch_data_in,
                         const VectorType              &level_set_as_heaviside_in,
                         const BlockVectorType         &normal_vector_in,
                         const EvaporationData<double> &evapor_data,
                         const MaterialData<double>    &material_data,
                         const unsigned int             normal_dof_idx_in,
                         const unsigned int             evapor_vel_dof_idx_in,
                         const unsigned int             evapor_mass_flux_dof_idx_in,
                         const unsigned int             ls_hanging_nodes_dof_idx_in,
                         const unsigned int             ls_quad_idx_in);


    /*
     * Configure the evaporation operation with temperature dependence.
     */
    void
    reinit(const VectorType                             *temperature_in,
           const VectorType                             &distance,
           const RecoilPressureData<double>             &recoil_data,
           const LevelSet::NearestPointData<double>     &nearest_point_data,
           const LevelSet::ReinitializationData<double> &reinit_data,
           const unsigned int                            temp_dof_idx_in);

    /*
     * Compute DoF vector holding evaporative mass flux depending on the given evaporation model
     * and the evaporation mass flux operator for computing the distribution across the interface.
     */
    void
    compute_evaporative_mass_flux();

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
      const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                   std::vector<Point<dim>> /*quad_points*/,
                                   std::vector<double> /*weights*/
                                   >> &surface_mesh_info);
    void
    reinit();

    void
    set_time(const double &time);
    /*
     * attach functions
     */
    void
    attach_dim_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

    void
    distribute_constraints();

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    /*
     * getter functions
     */
    inline Tensor<1, dim, VectorizedArray<double>> *
    begin_evaporation_velocity(const unsigned int macro_cell);

    inline const Tensor<1, dim, VectorizedArray<double>> &
    begin_evaporation_velocity(const unsigned int macro_cell) const;

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const;

    LinearAlgebra::distributed::Vector<double> &
    get_velocity();

    const VectorType &
    get_evaporative_mass_flux() const;

    VectorType &
    get_evaporative_mass_flux();
  };
} // namespace MeltPoolDG::Evaporation
