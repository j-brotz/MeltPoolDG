/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

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

    std::shared_ptr<const ScratchData<dim>> scratch_data;
    /**
     *  parameters controlling the evaporation
     */
    EvaporationData<double> evaporation_data;

    const MaterialData<double> &material;
    /**
     * references to solutions needed for the computation
     */
    const VectorType &     level_set_as_heaviside;
    const BlockVectorType &normal_vector;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int normal_dof_idx;
    const unsigned int evapor_vel_dof_idx;
    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int ls_quad_idx;
    /*
     * cut-off value for normalizing the normal vector field
     */
    const double tolerance_normal_vector;
    /*
     * optional: temperature-dependent evaporation
     */
    const VectorType *temperature;
    const double      temp_dof_idx;
    double            evaporation_mass_transfer_coefficient = 0.0;
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

  public:
    EvaporationOperation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                         const VectorType &                             level_set_as_heaviside_in,
                         const BlockVectorType &                        normal_vector_in,
                         std::shared_ptr<SimulationBase<dim>>           base_in,
                         const unsigned int                             normal_dof_idx_in,
                         const unsigned int                             evapor_vel_dof_idx_in,
                         const unsigned int                             ls_hanging_nodes_dof_idx_in,
                         const unsigned int                             ls_quad_idx_in,
                         const VectorType *                             temperature  = nullptr,
                         const unsigned int                             temp_dof_idx = 0);

    void
    reinit();


    void
    compute_evaporative_mass_flux_from_temperature_const_over_interface(const VectorType &distance);

    void
    compute_evaporative_mass_flux_from_temperature(
      const VectorType & temperature,
      const unsigned int temp_dof_idx,
      const double &     boiling_temperature  = 0.0, //@todo: remove from meltpool
      const double &     pressure_constant    = 0.0,
      const double &     temperature_constant = 0.0);

    void
    compute_evaporation_velocity(
      const std::string &interpolation_type_parameters = "consistent_with_evaporation");

    void
    compute_mass_balance_source_term(VectorType &       mass_balance_rhs,
                                     const unsigned int pressure_dof_idx,
                                     const unsigned int pressure_quad_idx,
                                     bool               zero_out);

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

  private:
    /**
     * @todo
     * !!!!!!!! HARD CODED PARAMETERS !!!!!!!!!!!!! --> this function will be replaced when the heat
     * equation is implemented anyhow
     */
    inline double
    compute_temperature_dependent_mass_flux_rate_from_recoil_pressure(
      const double &T,
      const double &pressure_constant,
      const double &temperature_constant,
      const double &boiling_temperature);

    /**
     *  According to Hardt and Wondra
     */
    inline double
    compute_temperature_dependent_mass_flux_rate(const double &T);
  };
} // namespace MeltPoolDG::Evaporation
