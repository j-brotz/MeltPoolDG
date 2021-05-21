#include <meltpooldg/heat/laser_heat_source_gusarov.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  LaserHeatSourceGusarov<dim>::LaserHeatSourceGusarov(
    const ScratchData<dim> &             scratch_data_in,
    const LaserData<double>::GusarovData gusarov_data_in,
    unsigned int                         temp_dof_idx_in)
    : scratch_data(scratch_data_in)
    , gusarov_data(gusarov_data_in)
    , lambda(gusarov_data.extinction_coefficient * gusarov_data.layer_thickness)
    , temp_dof_idx(temp_dof_idx_in)
  {}

  template <int dim>
  void
  LaserHeatSourceGusarov<dim>::compute_volumetric_heat_source(VectorType &      heat_source_vector,
                                                              const double      laser_power,
                                                              const Point<dim> &laser_position,
                                                              bool              zero_out)
  {
    if (zero_out)
      scratch_data.initialize_dof_vector(heat_source_vector, temp_dof_idx);

    FEValues<dim> heat_source_eval(
      scratch_data.get_mapping(),
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

  template <int dim>
  double
  LaserHeatSourceGusarov<dim>::power_density(const double radius, const double power)
  {
    const double &R = gusarov_data.laser_beam_radius;
    return radius <= R ? 3. * power / (numbers::PI * R * R) * std::pow(1. - radius / R, 2) *
                           std::pow(1 + radius / R, 2) :
                         0.0;
  }

  template <int dim>
  double
  LaserHeatSourceGusarov<dim>::dq_dxi(const double xi)
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

  template <int dim>
  double
  LaserHeatSourceGusarov<dim>::local_heat_source(const Point<dim> &position,
                                                 const Point<dim> &laser_position,
                                                 const double      power)
  {
    const double radius = position.distance(laser_position);
    const double z      = -(position[dim - 1] - laser_position[dim - 1]);
    const double xi     = z * gusarov_data.extinction_coefficient;

    return (z >= gusarov_data.layer_thickness) || (z < laser_position[dim - 1]) ?
             0. :
             -gusarov_data.extinction_coefficient * power_density(radius, power) * dq_dxi(xi);
  }

  template class LaserHeatSourceGusarov<1>;
  template class LaserHeatSourceGusarov<2>;
  template class LaserHeatSourceGusarov<3>;
} // namespace MeltPoolDG::Heat
