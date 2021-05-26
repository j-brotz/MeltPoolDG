/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, March 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /**
   * This class implements the laser heat source model of Gusarov et al. (2009).
   * DOI: 10.1115/1.3109245
   *
   *                       2*R
   *                      <--->
   *                      | | |
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
   */
  template <int dim>
  class LaserHeatSourceGusarov
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

    void
    compute_volumetric_heat_source(VectorType &            heat_source_vector,
                                   const ScratchData<dim> &scratch_data,
                                   unsigned int            temp_dof_idx,
                                   double                  laser_power,
                                   const Point<dim> &      laser_position,
                                   bool                    zero_out = true);

    /**
     * volumetric heat source; The z-axis (= axis of the laser beam) is assumed to correspond to
     * negative dim-1 coordinate.
     */
    double
    get_local_heat_source(const Point<dim> &position,
                          const Point<dim> &laser_position,
                          double            power) const;

  private:
    /**
     * Equation (26) corrected to match Figure 5 of Gusarov et al. (2009)
     */
    double
    power_density(double radius, double power) const;

    /**
     * Analytical derivative of equation (20)
     */
    double
    dq_dxi(double xi) const;
  };
} // namespace MeltPoolDG::Heat
