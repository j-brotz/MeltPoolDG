/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::HeatEquation
{
  using namespace dealii;

  template <int dim>
  class HeatOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    HeatOperation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                  const MeltPoolData<double> &                   mp_data_in,
                  const double                                   density_liquid_in,
                  const double                                   density_gas_in,
                  const Point<dim> &                             laser_center_in,
                  const double &                                 time_in,
                  const unsigned int                             temp_dof_idx_in,
                  const unsigned int                             temp_quad_idx_in)
      : scratch_data(scratch_data_in)
      , mp_data(mp_data_in)
      , density_liquid(density_liquid_in)
      , density_gas(density_gas_in)
      , laser_center(laser_center_in)
      , time(time_in)
      , temp_dof_idx(temp_dof_idx_in)
      , temp_quad_idx(temp_quad_idx_in)
    {}

    void
    reinit()
    {
      scratch_data->initialize_dof_vector(temperature, temp_dof_idx);
      scratch_data->initialize_dof_vector(solid, temp_dof_idx);
      scratch_data->initialize_dof_vector(liquid, temp_dof_idx);
    }

    void
    solve(const VectorType &level_set_as_heaviside)
    {
      level_set_as_heaviside.update_ghost_values();

      reinit();
      FEValues<dim> fe_values(scratch_data->get_mapping(),
                              scratch_data->get_dof_handler(temp_dof_idx).get_fe(),
                              scratch_data->get_quadrature(temp_quad_idx),
                              update_values | update_gradients | update_quadrature_points |
                                update_JxW_values);

      const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(temp_dof_idx);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::map<types::global_dof_index, Point<dim>> support_points;
      DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                           scratch_data->get_dof_handler(temp_dof_idx),
                                           support_points);

      for (const auto &cell : scratch_data->get_dof_handler(temp_dof_idx).active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                temperature[local_dof_indices[i]] =
                  analytical_temperature_field(support_points[local_dof_indices[i]],
                                               level_set_as_heaviside[local_dof_indices[i]]);
                solid[local_dof_indices[i]] = is_solid_region(support_points[local_dof_indices[i]]);
                liquid[local_dof_indices[i]] =
                  is_liquid_region(support_points[local_dof_indices[i]]);
              }
          }

      temperature.compress(VectorOperation::insert);

      scratch_data->get_constraint(temp_dof_idx).distribute(temperature);
      scratch_data->get_constraint(temp_dof_idx).distribute(solid);

      level_set_as_heaviside.zero_out_ghosts();
    }

    const VectorType &
    get_temperature() const
    {
      return temperature;
    }

    const VectorType &
    get_solid() const
    {
      return solid;
    }

    const VectorType &
    get_liquid() const
    {
      return liquid;
    }

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      temperature.update_ghost_values();
      solid.update_ghost_values();
      liquid.update_ghost_values();
      vectors.push_back(&temperature);
      vectors.push_back(&solid);
      vectors.push_back(&liquid);
    }

    void
    distribute_constraints()
    {
      scratch_data->get_constraint(temp_dof_idx).distribute(temperature);
      scratch_data->get_constraint(temp_dof_idx).distribute(solid);
      scratch_data->get_constraint(temp_dof_idx).distribute(liquid);
    }

    void
    attach_output_vectors(DataOut<dim> &data_out) const
    {
      MeltPoolDG::VectorTools::update_ghost_values(temperature, solid, liquid);
      /**
       *  temperature
       */
      data_out.add_data_vector(scratch_data->get_dof_handler(temp_dof_idx),
                               temperature,
                               "temperature");
      /**
       *  solid
       */
      data_out.add_data_vector(scratch_data->get_dof_handler(temp_dof_idx), solid, "solid");
      /**
       *  liquid
       */
      data_out.add_data_vector(scratch_data->get_dof_handler(temp_dof_idx), liquid, "liquid");
    }

    /**
     *  This function determines for a given point, whether it belongs to the solid domain.
     *
     *  WARNING: All points above (component dim-1) the center point of the laser source are
     *  automatically identified as gaseous parts. Thus, this function has to be modified when the
     *  initial interface between the feedstock and the ambient gas is not planar.
     */
    bool
    is_solid_region(const Point<dim> point)
    {
      if (mp_data.liquid.melt_pool_radius == 0)
        return false;
      else if (point[dim - 1] >= laser_center[dim - 1])
        return false;
      else
        {
          Point<dim> shifted_center =
            MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
              mp_data.melt_pool_center);

          if (mp_data.melt_pool_shape == "parabola")
            {
              if (dim == 2)
                {
                  const double sign =
                    -2 * mp_data.liquid.melt_pool_radius * (point[1] - shifted_center[1]) +
                    std::pow(point[0] - shifted_center[0], 2);
                  return (sign >= 0) ? true : false;
                }
              else
                AssertThrow(false, ExcMessage("not implemented"));
            }
          else if (mp_data.melt_pool_shape == "ellipse")
            return UtilityFunctions::DistanceFunctions::ellipsoidal_manifold<dim>(
                     point,
                     shifted_center,
                     mp_data.liquid.melt_pool_radius,
                     mp_data.liquid.melt_pool_depth) > 0 ?
                     false :
                     true;
          else if (mp_data.melt_pool_shape == "temperature_dependent")
            return (analytical_temperature_field(point, 1.0 /* is_liquid */) >=
                    mp_data.liquid.melting_point) ?
                     false :
                     true;
          else
            AssertThrow(false, ExcMessage("not implemented"));
        }
    }

  private:
    double
    analytical_temperature_field(Point<dim> point, const double phi)
    {
      if (mp_data.temperature_formulation == "analytical")
        {
          // The temperature function below is derived from the publication on
          // "Heat Source Modeling in Selective Laser Melting" by E. Mirkoohi, D. E. Seivers,
          //  H. Garmestani and S. Y. Liang
          //
          //  In order to capture anisotropic temperature fields, a modification is introduced.
          const double indicator = UtilityFunctions::CharacteristicFunctions::heaviside(phi, 0.5);


          double laser_intensity = 1.0;
          if (mp_data.laser_power_over_time == "ramp")
            {
              AssertThrow(
                mp_data.laser_power_end_time > mp_data.laser_power_start_time,
                ExcMessage(
                  "For the temporal ramp distribution of the laser power,"
                  " the parameter laser power end time must be larger than laser power start time."));
              laser_intensity = (time - mp_data.laser_power_start_time) /
                                (mp_data.laser_power_end_time - mp_data.laser_power_start_time);
              laser_intensity = std::min(std::max(0.0, laser_intensity), 1.0);
            }


          const double &P  = mp_data.laser_power * laser_intensity;
          const double &v  = mp_data.scan_speed;
          const double &T0 = mp_data.ambient_temperature;
          const double &absorptivity =
            (indicator == 1) ? mp_data.liquid.absorptivity : mp_data.gas.absorptivity;
          const double &conductivity =
            (indicator == 1) ? mp_data.liquid.conductivity : mp_data.gas.conductivity;
          const double &capacity =
            (indicator == 1) ? mp_data.liquid.capacity : mp_data.gas.capacity;
          const double density = density_liquid + (density_gas - density_liquid) * indicator;

          const double thermal_diffusivity = conductivity / (density * capacity);

          // modify temperature profile to be anisotropic
          for (int d = 0; d < dim - 1; d++)
            point[d] *= mp_data.temperature_x_to_y_ratio;

          double R = point.distance(laser_center);

          if (R == 0.0)
            R = 1e-16;
          double T = P * absorptivity / (4 * numbers::PI * R * conductivity) *
                       std::exp(-v * (R) / (2. * thermal_diffusivity)) +
                     T0;
          return (T > mp_data.max_temperature) ? mp_data.max_temperature : T;
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }



    /*
     * check if a given point is liquid
     */
    bool
    is_liquid_region(const Point<dim> point)
    {
      if (mp_data.liquid.melt_pool_radius == 0)
        return false;
      else if (point[dim - 1] >= laser_center[dim - 1])
        return false;
      else
        {
          Point<dim> shifted_center =
            MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
              mp_data.melt_pool_center);

          if (mp_data.melt_pool_shape == "parabola")
            {
              if (dim == 2)
                {
                  const double sign =
                    -2 * mp_data.liquid.melt_pool_radius * (point[1] - shifted_center[1]) +
                    std::pow(point[0] - shifted_center[0], 2);
                  return (sign >= 0) ? false : true;
                }
              else
                AssertThrow(false, ExcMessage("not implemented"));
            }
          else if (mp_data.melt_pool_shape == "ellipse")
            return UtilityFunctions::DistanceFunctions::ellipsoidal_manifold<dim>(
                     point,
                     shifted_center,
                     mp_data.liquid.melt_pool_radius,
                     mp_data.liquid.melt_pool_depth) > 0 ?
                     true :
                     false;
          else if (mp_data.melt_pool_shape == "temperature_dependent")
            return (analytical_temperature_field(point, 1.0 /* is_liquid */) >=
                    mp_data.liquid.melting_point) ?
                     true :
                     false;
          else
            AssertThrow(false, ExcMessage("not implemented"));
        }
    }

  private:
    const std::shared_ptr<const ScratchData<dim>> &scratch_data;
    /**
     *  parameters controlling the HeatEquation
     */
    const MeltPoolData<double> &mp_data;

    const double density_liquid;
    const double density_gas;
    /*
     *  Center of the laser
     */
    const Point<dim> &laser_center;
    /**
     *  time
     */
    const double &time;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int temp_dof_idx;
    const unsigned int temp_quad_idx;
    /*
     *    This are the primary solution variables of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType temperature;
    VectorType solid;
    VectorType liquid;
  };
} // namespace MeltPoolDG::HeatEquation
