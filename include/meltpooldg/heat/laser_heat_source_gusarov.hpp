/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, March 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/laser_heat_source_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * This class implements the laser heat source model of Gusarov et al. (2009).
   * DOI: 10.1115/1.3109245
   *
   *   ^ dim-1             2*R
   *   |                  <--->
   *   ----> x            | | |
   *                      | | |  laser beam
   *                      | | |     power
   *                      | | |
   *                      | | |
   *
   *    ---------------------------------------^
   *             ++                  ++        |
   *               ++      q       ++          |  layer thickness
   *                 +++        +++            |
   *                     ++++++                v
   *
   * The laser beam is direction is assumed to be in the negative dim-1 direction.
   */
  template <int dim>
  class LaserHeatSourceGusarov : public LaserHeatSourceBase<dim>
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    /*
     *  Parameters for the Gusarov model
     */
    const LaserData<double>::GusarovData gusarov_data;

    const double lambda; // optical thickness (-)

  public:
    LaserHeatSourceGusarov(const LaserData<double>::GusarovData &gusarov_data_in);

    /**
     * volumetric heat source; The z-axis (= axis of the laser beam) is assumed to correspond to
     * negative dim-1 coordinate.
     */
    void
    compute_interfacial_heat_source(VectorType &            heat_source_vector,
                                    const ScratchData<dim> &scratch_data,
                                    const unsigned int      temp_dof_idx,
                                    const double            laser_power,
                                    const Point<dim> &      laser_position,
                                    const VectorType &      level_set_heaviside,
                                    const unsigned int      ls_dof_idx,
                                    const bool              zero_out       = true,
                                    const BlockVectorType * normal_vector  = nullptr,
                                    const unsigned int      normal_dof_idx = 0) const final;


    double
    local_compute_volumetric_heat_source(const Point<dim> &position,
                                         const Point<dim> &laser_position,
                                         const double      power) const final;

  private:
    /**
     * Equation (26) corrected to match Figure 5 of Gusarov et al. (2009)
     */
    double
    power_density(double radius, double power) const;

    /**
     * Analytical derivative of equation (20) of Gusarov et al. (2009)
     */
    double
    dq_dxi(double xi) const;

    const double a;
    const double D;
  };
} // namespace MeltPoolDG::Heat
