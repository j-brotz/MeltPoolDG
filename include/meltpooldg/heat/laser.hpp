/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class LaserOperation
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const ScratchData<dim> &scratch_data;
    /*
     *  Laser parameters
     */
    const LaserData<double> laser_data;

    const MaterialData<double> material;
    /*
     *  Center of the laser
     */
    Point<dim> laser_position;
    /*
     *  Current time
     */
    double current_time;
    /*
     *  Current intensity of the laser
     */
    double laser_intensity;

  public:
    LaserOperation(const ScratchData<dim> &    scratch_data_in,
                   const LaserData<double> &   laser_data_in,
                   const MaterialData<double> &material_data_in);

    void
    set_initial_condition(double start_time);

    void
    move_laser(double dt);

    Point<dim>
    get_laser_position();

    double
    get_laser_power();

    LaserImpactType
    get_laser_impact_type() const;

  private:
    void
    compute_laser_intensity();
  };
} // namespace MeltPoolDG::Heat
