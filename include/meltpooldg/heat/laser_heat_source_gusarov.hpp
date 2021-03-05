/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, March 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

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

    const ScratchData<dim> &scratch_data;
    /*
     *  Parameters for the Gusarov model
     */
    const LaserData<double>::GusarovData gusarov_data;

    const double lambda; // optical thickness (-)

    const unsigned int temp_dof_idx;

  public:
    LaserHeatSourceGusarov(const ScratchData<dim> &             scratch_data_in,
                           const LaserData<double>::GusarovData gusarov_data_in,
                           unsigned int                         temp_dof_idx_in)
      : scratch_data(scratch_data_in)
      , gusarov_data(gusarov_data_in)
      , lambda(gusarov_data.extinction_coefficient * gusarov_data.layer_thickness)
      , temp_dof_idx(temp_dof_idx_in)
    {}

    void
    compute_volumetric_heat_source(VectorType &      heat_source_vector,
                                   const double      laser_power,
                                   const Point<dim> &laser_position,
                                   bool              zero_out = true)
    {
      if (zero_out)
        scratch_data.initialize_dof_vector(heat_source_vector, temp_dof_idx);

      FEValues<dim> heat_source_eval(
        scratch_data->get_mapping(),
        scratch_data.get_dof_handler(temp_dof_idx).get_fe(),
        Quadrature<dim>(
          scratch_data.get_dof_handler(temp_dof_idx).get_fe().get_unit_support_points()),
        update_quadrature_points);

      const unsigned int dofs_per_cell =
        scratch_data.get_dof_handler(temp_dof_idx).get_fe().n_dofs_per_cell();
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : scratch_data.get_dof_handler(temp_dof_idx).active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);

              heat_source_eval.reinit(cell);

              for (const auto q : heat_source_eval.quadrature_point_indices())
                heat_source_vector[local_dof_indices[q]] =
                  local_heat_source(heat_source_eval.quadrature_point(q),
                                    laser_position,
                                    laser_power);
            }
        }
    }

  private:
    /**
     * Equation (26) corrected to match Figure 5 of Gusarov et al. (2009)
     */
    double
    power_density(const double radius, const double power)
    {
      const double &R = gusarov_data.laser_beam_radius;
      return radius <= R ? 3. * power / (numbers::PI * R * R) * std::pow(1. - radius / R, 2) *
                             std::pow(1 + radius / R, 2) :
                           0.0;
    }

    /**
     * Analytical derivative of equation (20)
     */
    double
    dq_dxi(const double xi)
    {
      const double &rho = gusarov_data.reflectivity;
      const double  a   = std::sqrt(1 - rho);

      const double D = (1 - a) * (1 - a - rho * (1 + a)) * std::exp(-2 * a * lambda) -
                       (1 + a) * (1 + a - rho * (1 - a)) * std::exp(2 * a * lambda);

      // clang-format off
      return xi < lambda ?
               ((3 - 3 * rho) *(std::exp(-xi) + rho * std::exp(xi - 2 * lambda))) 
               / // ------------------------------------------------------
                                       (4 * rho - 3) 
                +
                       2 * a * a * rho 
                / // ------------------
                   (D * (4 * rho - 3)) 
                *
                   (
                    std::exp(-lambda) * (1 - rho * rho) *
                      (std::exp(-2 * a * xi) * (a - 1) + std::exp(2 * a * xi) * (a + 1)) 
                   -
                    (rho * std::exp(-2 * lambda) + 3) *
                      (std::exp(2 * a * (xi - lambda)) * (1 - a - rho * (a + 1)) -
                       std::exp(2 * a * (lambda - xi)) * (a + rho * (a - 1) + 1))
                   ) :
               0.0;
      // clang-format on
    }

    /**
     * volumetric heat source; The z-axis (= axis of the laser beam) is assumed to correspond to
     * negative dim-1 coordinate.
     */

    void
    local_heat_source(const Point<dim> &position,
                      const Point<dim> &laser_position,
                      const double      power)
    {
      const double radius = position.distance(laser_position);
      const double z      = -position[dim - 1];
      const double xi     = z * gusarov_data.extinction_coefficient;

      return (z >= gusarov_data.layer_thickness) || (z < laser_position[dim - 1]) ?
               0. :
               -gusarov_data.extinction_coefficient * power_density(radius, power) * dq_dxi(xi);
    }
  };
} // namespace MeltPoolDG::Heat
