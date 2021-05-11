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
                         const unsigned int                             temp_dof_idx = 0)

      : scratch_data(scratch_data_in)
      , evaporation_data(base_in->parameters.evapor)
      , material(base_in->parameters.material)
      , level_set_as_heaviside(level_set_as_heaviside_in)
      , normal_vector(normal_vector_in)
      , normal_dof_idx(normal_dof_idx_in)
      , evapor_vel_dof_idx(evapor_vel_dof_idx_in)
      , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx_in)
      , ls_quad_idx(ls_quad_idx_in)
      , tolerance_normal_vector(
          std::min(1e-2,
                   std::max(std::pow(10,
                                     UtilityFunctions::get_exponent_power_ten(std::pow(
                                       GridTools::volume<dim>(scratch_data->get_triangulation(),
                                                              scratch_data->get_mapping()),
                                       1. / dim))) *
                              1e-3,
                            1e-12)))
      , temperature(temperature)
      , temp_dof_idx(temp_dof_idx)

    {
      AssertThrow(material.first.density > 0.0 && material.second.density > 0.0,
                  ExcMessage("The materials' densities must be greater than zero! Abort..."));
      reinit();
      // @todo: provide static function
      if (evaporation_data.formulation_evaporative_mass_flux.find("temperature dependent") !=
          std::string::npos)
        {
          AssertThrow(
            std::abs(evaporation_data.boiling_temperature) > 1e-12,
            ExcMessage(
              "The boiling temperature must not be zero to compte the evaporation mass transfer coefficient"));

          evaporation_mass_transfer_coefficient =
            2. * evaporation_data.coefficient * evaporation_data.latent_heat_of_evaporation *
            material.first.density /
            ((2. - evaporation_data.coefficient) *
             std::sqrt(2. * numbers::PI * PhysicalConstants::universal_gas_constant /
                       evaporation_data.molar_mass) *
             std::pow(evaporation_data.boiling_temperature, 1.5));
        }
    }

    void
    reinit()
    {
      scratch_data->initialize_dof_vector(
        evaporative_mass_flux, ls_hanging_nodes_dof_idx); // @todo: evapor_dof_idx/temp_dof_idx
      scratch_data->initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);
    }


    void
    compute_evaporative_mass_flux_from_temperature_const_over_interface(const VectorType &distance)
    {
      /*
       * collect evaluation points
       */
      distance.update_ghost_values();
      normal_vector.update_ghost_values();

      FEValues<dim, dim> ls_values(
        scratch_data->get_mapping(),
        scratch_data->get_fe(ls_hanging_nodes_dof_idx),
        Quadrature<dim>(
          scratch_data->get_fe(ls_hanging_nodes_dof_idx).base_element(0).get_unit_support_points()),
        update_quadrature_points);

      std::vector<Point<dim>>              evaluation_points;
      std::vector<types::global_dof_index> dof_indices;

      const unsigned int n_q_points = ls_values.get_quadrature().size();

      // temporary values at cell nodes nodes
      Vector<double>                       hs_temp(n_q_points);
      Vector<double>                       distance_temp(n_q_points);
      std::vector<Vector<double>>          normal_temp(dim, Vector<double>(n_q_points));
      std::vector<Point<dim>>              normalized_normal_temp(n_q_points, Point<dim>());
      std::vector<types::global_dof_index> temp_local_dof_indices(n_q_points);

      auto bounding_box    = GridTools::compute_bounding_box(scratch_data->get_triangulation());
      auto boundary_points = bounding_box.get_boundary_points();

      for (const auto &cell :
           scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              ls_values.reinit(cell);

              cell->get_dof_indices(temp_local_dof_indices);

              cell->get_dof_values(level_set_as_heaviside, hs_temp);
              cell->get_dof_values(distance, distance_temp);
              for (unsigned int d = 0; d < dim; ++d)
                cell->get_dof_values(normal_vector.block(d), normal_temp[d]);

              // normalize normal vector values
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int q = 0; q < n_q_points; ++q)
                  normalized_normal_temp[q][d] = normal_temp[d][q];

              for (auto &n : normalized_normal_temp)
                n = n / n.norm();

              for (const auto q : ls_values.quadrature_point_indices())
                {
                  if (hs_temp[q] < 1.0 && hs_temp[q] > 0.0)
                    {
                      // compute corresponding point at level set == 0.0

                      Point<dim> evaluation_point_temp = Point<dim>();
                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          evaluation_point_temp[d] =
                            ls_values.quadrature_point(q)[d] -
                            distance_temp[q] * normalized_normal_temp[q][d];

                          // check if point is outside domain and if so then project it back to the
                          // domain
                          if (evaluation_point_temp[d] < boundary_points.first[d])
                            evaluation_point_temp[d] = boundary_points.first[d];
                          else if (evaluation_point_temp[d] > boundary_points.second[d])
                            evaluation_point_temp[d] = boundary_points.second[d];
                        }

                      evaluation_points.push_back(evaluation_point_temp);
                      dof_indices.push_back(temp_local_dof_indices[q]);
                    }
                }
            }
        }

      /*
       * get temperature values at evaluation points
       */
      Utilities::MPI::RemotePointEvaluation<dim, dim> cache;

      temperature->update_ghost_values();
      const auto temperature_evaluation_values =
        dealii::VectorTools::evaluate_at_points<1>(scratch_data->get_mapping(),
                                                   scratch_data->get_dof_handler(temp_dof_idx),
                                                   *temperature,
                                                   evaluation_points,
                                                   cache,
                                                   dealii::VectorTools::EvaluationFlags::max);
      temperature->zero_out_ghosts();

      Assert(temperature_evaluation_values.size() == evaluation_points.size(),
             ExcMessage("The size of vectors must match."));

      /*
       * compute evaporative mass flux
       */
      evaporative_mass_flux.zero_out_ghosts();
      evaporative_mass_flux = 0.0;

      for (unsigned int i = 0; i < evaluation_points.size(); i++)
        {
          evaporative_mass_flux[dof_indices[i]] =
            (temperature_evaluation_values[i] >= evaporation_data.boiling_temperature) ?
              evaporation_mass_transfer_coefficient *
                (temperature_evaluation_values[i] - evaporation_data.boiling_temperature) :
              0.0;
        }

      evaporative_mass_flux.update_ghost_values();
    }

    void
    compute_evaporative_mass_flux_from_temperature(
      const VectorType & temperature,
      const unsigned int temp_dof_idx,
      const double &     boiling_temperature  = 0.0, //@todo: remove from meltpool
      const double &     pressure_constant    = 0.0,
      const double &     temperature_constant = 0.0)
    {
      temperature.update_ghost_values();
      const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(temp_dof_idx);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : scratch_data->get_dof_handler(temp_dof_idx).active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              if (evaporation_data.formulation_evaporative_mass_flux == "temperature dependent")
                {
                  evaporative_mass_flux[local_dof_indices[i]] =
                    compute_temperature_dependent_mass_flux_rate(temperature[local_dof_indices[i]]);
                }
              else if (evaporation_data.formulation_evaporative_mass_flux ==
                       "recoil pressure dependent")
                {
                  evaporative_mass_flux[local_dof_indices[i]] =
                    compute_temperature_dependent_mass_flux_rate_from_recoil_pressure(
                      temperature[local_dof_indices[i]],
                      pressure_constant,
                      temperature_constant,
                      boiling_temperature);
                }
          }
      temperature.zero_out_ghosts();
    }

    void
    compute_evaporation_velocity(
      const std::string &interpolation_type_parameters = "consistent_with_evaporation")
    {
      level_set_as_heaviside.update_ghost_values();
      normal_vector.update_ghost_values();
      evaporative_mass_flux.update_ghost_values();

      FECellIntegrator<dim, 1, double> ls(scratch_data->get_matrix_free(),
                                          ls_hanging_nodes_dof_idx,
                                          ls_quad_idx);

      FECellIntegrator<dim, dim, double> normal_vec(scratch_data->get_matrix_free(),
                                                    normal_dof_idx,
                                                    ls_quad_idx);

      FECellIntegrator<dim, 1, double> evap_flux(
        scratch_data->get_matrix_free(),
        ls_hanging_nodes_dof_idx, // @todo: generalize --> temp_dof_idx
        ls_quad_idx);

      evaporation_velocities.resize(scratch_data->get_matrix_free().n_cell_batches() *
                                    ls.n_q_points);

      for (unsigned int cell = 0; cell < scratch_data->get_matrix_free().n_cell_batches(); ++cell)
        {
          Tensor<1, dim, VectorizedArray<double>> *evapor_vel = begin_evaporation_velocity(cell);

          ls.reinit(cell);
          ls.read_dof_values(level_set_as_heaviside);
          ls.evaluate(true, true);

          normal_vec.reinit(cell);
          normal_vec.read_dof_values(normal_vector);
          normal_vec.evaluate(true, false);

          evap_flux.reinit(cell);
          evap_flux.read_dof_values(evaporative_mass_flux);
          evap_flux.evaluate(true, false);

          for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
            {
              const auto n_phi =
                MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index),
                                                        tolerance_normal_vector);

              //              ρ
              // evaluate  ------
              //            dρ/dΦ
              VectorizedArray<double> rho_d_rho_d_phi = 1.0;

              if (interpolation_type_parameters == "true")
                {
                  // clang-format off
                  rho_d_rho_d_phi = (ls.get_value(q_index) * material.second.density + (1.-ls.get_value(q_index) * material.first.density)) 
                    / //-----------------------------------------------------------------------------------------------------
                                                    (material.second.density - material.first.density);
                  // clang-format on
                }

              evapor_vel[q_index] = n_phi * evap_flux.get_value(q_index) * rho_d_rho_d_phi *
                                    UtilityFunctions::interpolate(ls.get_value(q_index),
                                                                  1. / material.first.density,
                                                                  1. / material.second.density);

              // The normal vector field is oriented such that the normal vector points from
              // the negative level set value (= default for representing the gas phase) to the
              // positive value (= default for representing the liquid phase). Thus, in case the
              // gas phase corresponds to a level set value of 1, the sign of the normal vector
              // has to be changed.
              if (evaporation_data.ls_value_gas == 1.0)
                AssertThrow(false, ExcNotImplemented());
              // evapor_vel[q_index] *= -1.0;
            }
        }
      level_set_as_heaviside.zero_out_ghosts();
      normal_vector.zero_out_ghosts();
      evaporative_mass_flux.zero_out_ghosts();

      scratch_data->initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);

      /**
       * write interface velocity to dof vector
       */
      UtilityFunctions::fill_dof_vector_from_cell_operation_vec<dim, dim>(
        evaporation_velocity,
        scratch_data->get_matrix_free(),
        evapor_vel_dof_idx,
        ls_quad_idx,
        scratch_data->get_degree(evapor_vel_dof_idx),           // fe_degree of the resulting vector
        scratch_data->get_degree(ls_hanging_nodes_dof_idx) + 1, // n_q_points_1d of cell operation
        [&](const unsigned int cell,
            const unsigned int quad) -> const Tensor<1, dim, VectorizedArray<double>> & {
          return begin_evaporation_velocity(cell)[quad];
        });

      scratch_data->get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);

      scratch_data->get_pcout(1) << "    | evapor: |u|2 = "
                                 << VectorTools::compute_L2_norm<dim>(evaporation_velocity,
                                                                      *scratch_data,
                                                                      evapor_vel_dof_idx,
                                                                      ls_quad_idx)
                                 << std::endl;

      evaporation_velocity.zero_out_ghosts();
    }

    void
    compute_mass_balance_source_term(VectorType &       mass_balance_rhs,
                                     const unsigned int pressure_dof_idx,
                                     const unsigned int pressure_quad_idx,
                                     bool               zero_out)
    {
      evaporative_mass_flux.update_ghost_values();
      if (temperature)
        temperature->update_ghost_values();


      double mass = 0.0;

      scratch_data->get_matrix_free().template cell_loop<VectorType, VectorType>(
        [&](const auto &matrix_free,
            auto &      mass_balance_rhs,
            const auto &level_set_as_heaviside,
            auto        macro_cells) {
          FECellIntegrator<dim, 1, double> heaviside(matrix_free,
                                                     ls_hanging_nodes_dof_idx,
                                                     pressure_quad_idx);

          FECellIntegrator<dim, 1, double> mass_flux(matrix_free,
                                                     pressure_dof_idx,
                                                     pressure_quad_idx);

          FECellIntegrator<dim, 1, double> evap_flux(
            scratch_data->get_matrix_free(),
            ls_hanging_nodes_dof_idx, // @todo: generalize --> temp_dof_idx
            pressure_quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              heaviside.reinit(cell);
              heaviside.read_dof_values_plain(level_set_as_heaviside);
              heaviside.evaluate(false, true);

              mass_flux.reinit(cell);

              evap_flux.reinit(cell);
              evap_flux.read_dof_values_plain(evaporative_mass_flux);
              evap_flux.evaluate(true, false);

              for (unsigned int q_index = 0; q_index < mass_flux.n_q_points; ++q_index)
                {
                  mass_flux.submit_value((1. / material.second.density -
                                          1. / material.first.density) *
                                           heaviside.get_gradient(q_index).norm() *
                                           evap_flux.get_value(q_index),
                                         q_index);
                  // compute overall rhs
                  for (unsigned int v = 0;
                       v < scratch_data->get_matrix_free().n_active_entries_per_cell_batch(cell);
                       ++v)
                    {
                      mass += (1. / material.second.density - 1. / material.first.density) *
                              heaviside.get_gradient(q_index).norm()[v] *
                              evap_flux.get_value(q_index)[v] * mass_flux.JxW(q_index)[v];
                    }
                }

              mass_flux.integrate_scatter(true, false, mass_balance_rhs);
            }
        },
        mass_balance_rhs,
        level_set_as_heaviside,
        zero_out);
      evaporative_mass_flux.zero_out_ghosts();

      scratch_data->get_pcout() << "    | evaporation: jump in the velocity field = "
                                << Utilities::MPI::sum(mass, scratch_data->get_mpi_comm())
                                << std::endl;

      scratch_data->get_pcout() << "    | evapor: |m|2 = " << mass_balance_rhs.l2_norm()
                                << std::endl;
    }

    /*
     * attach functions
     */
    void
    attach_dim_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      evaporation_velocity.update_ghost_values();
      vectors.push_back(&evaporation_velocity);
    }

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      evaporative_mass_flux.update_ghost_values();
      vectors.push_back(&evaporative_mass_flux);
    }

    void
    distribute_constraints()
    {
      scratch_data->get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);
      scratch_data->get_constraint(ls_hanging_nodes_dof_idx).distribute(evaporative_mass_flux);
    }

    void
    attach_output_vectors(DataOut<dim> &data_out) const
    {
      /*
       *  evaporation velocity
       */
      evaporation_velocity.update_ghost_values();
      evaporative_mass_flux.update_ghost_values();

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(scratch_data->get_dof_handler(evapor_vel_dof_idx),
                               evaporation_velocity,
                               std::vector<std::string>(dim, "evaporation_velocity"),
                               vector_component_interpretation);
      /*
       *  evaporation mass flux
       */
      data_out.add_data_vector(scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx),
                               evaporative_mass_flux,
                               "evaporative_mass_flux");
    }

    /*
     * getter functions
     */
    inline Tensor<1, dim, VectorizedArray<double>> *
    begin_evaporation_velocity(const unsigned int macro_cell)
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      return &evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                     macro_cell];
    }

    inline const Tensor<1, dim, VectorizedArray<double>> &
    begin_evaporation_velocity(const unsigned int macro_cell) const
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      return evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                    macro_cell];
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const
    {
      return evaporation_velocity;
    }

    LinearAlgebra::distributed::Vector<double> &
    get_velocity()
    {
      return evaporation_velocity;
    }

    const VectorType &
    get_evaporative_mass_flux() const
    {
      return evaporative_mass_flux;
    }

    VectorType &
    get_evaporative_mass_flux()
    {
      return evaporative_mass_flux;
    }

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
      const double &boiling_temperature)
    {
      // according to Meier 2020
      const double cs = 1.0;  // sticking coefficent
      const double Cm = 1e-3; // molar_mass/(2*pi*molar_gas_constant)
      return (T >= boiling_temperature) ?
               evaporation_data.evaporative_mass_flux_scale_factor * 0.82 * cs *
                 MeltPool::RecoilPressureOperation<dim>::compute_recoil_pressure_coefficient(
                   T, pressure_constant, temperature_constant, boiling_temperature) *
                 std::sqrt(Cm / T) :
               0.0;
    }

    /**
     *  According to Hardt and Wondra
     */
    inline double
    compute_temperature_dependent_mass_flux_rate(const double &T)
    {
      return (T >= evaporation_data.boiling_temperature) ?
               evaporation_mass_transfer_coefficient * (T - evaporation_data.boiling_temperature) :
               0.0;
    }
  };
} // namespace MeltPoolDG::Evaporation
