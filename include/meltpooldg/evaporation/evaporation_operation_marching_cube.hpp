/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/utilities/vector_tools.hpp>

#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <adaflo/sharp_interface_util.h> //@todo: will be replace by the utility function of deal.II as soon the PR is merged
#endif

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  /**                                                        .
   *  This module computes for a given evaporative mass flux m the corresponding
   *  term in the continuity equation in a sharp manner exploiting the Marching
   *  Cube Algorithm.
   *
   *  /           \     /    .    1       1    \
   *  | w , ∇ · u |   = | w, m ( ---  -  --- ) |
   *  \           /     \         ρl      ρg   /
   *              Ω                             Γ
   *                                             lg
   *  with the domain of interest Ω, the test functions w, the liquid-gaseous
   *  interface Γ_lg, the fluid velocity field u, density of the liquid phase ρl
   *  and density of the gaseous phase ρg.
   *
   *  The evaporation velocity is then computed as follows
   *
   *          /  .
   *          |  m / ρl    if phi > 0
   *  u = n   |  .
   *       Γ  |  m / ρg    else
   *          \
   */

  template <int dim>
  class EvaporationOperationMarchingCube
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    EvaporationOperationMarchingCube()
    {}

    static void
    compute_evaporation_velocity(const ScratchData<dim> &scratch_data,
                                 VectorType &            evaporation_velocity,
                                 const VectorType &      evaporative_mass_flux,
                                 const VectorType &      level_set_as_heaviside,
                                 const BlockVectorType & normal_vector,
                                 const double            rho_l,
                                 const double            rho_g,
                                 const unsigned int      evapor_vel_dof_idx,
                                 const unsigned int      ls_hanging_nodes_dof_idx,
                                 const unsigned int      ls_quad_idx,
                                 const unsigned int      normal_dof_idx)
    {
      /**
       * evaporation velocity at quadrature points
       */
      AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaporation_velocities;

      level_set_as_heaviside.update_ghost_values();
      normal_vector.update_ghost_values();
      evaporative_mass_flux.update_ghost_values();

      FECellIntegrator<dim, 1, double> ls(scratch_data.get_matrix_free(),
                                          ls_hanging_nodes_dof_idx,
                                          ls_quad_idx);

      FECellIntegrator<dim, dim, double> normal_vec(scratch_data.get_matrix_free(),
                                                    normal_dof_idx,
                                                    ls_quad_idx);

      FECellIntegrator<dim, 1, double> evap_flux(
        scratch_data.get_matrix_free(),
        ls_hanging_nodes_dof_idx, // @todo: generalize --> temp_dof_idx
        ls_quad_idx);

      evaporation_velocities.resize(scratch_data.get_matrix_free().n_cell_batches() *
                                    ls.n_q_points);

      for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
        {
          Tensor<1, dim, VectorizedArray<double>> *evapor_vel =
            &evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                    cell];

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
                MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index));

              auto is_liquid = compare_and_apply_mask<SIMDComparison::less_than>(
                ls.get_value(q_index),
                VectorizedArray<double>(0.5),
                evap_flux.get_value(q_index) / rho_g,
                evap_flux.get_value(q_index) / rho_l);

              evapor_vel[q_index] = is_liquid * n_phi;
            }
        }
      level_set_as_heaviside.zero_out_ghosts();
      normal_vector.zero_out_ghosts();
      evaporative_mass_flux.zero_out_ghosts();

      scratch_data.initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);

      /**
       * write interface velocity to dof vector
       */
      UtilityFunctions::fill_dof_vector_from_cell_operation_vec<dim, dim>(
        evaporation_velocity,
        scratch_data.get_matrix_free(),
        evapor_vel_dof_idx,
        ls_quad_idx,
        scratch_data.get_degree(evapor_vel_dof_idx),           // fe_degree of the resulting vector
        scratch_data.get_degree(ls_hanging_nodes_dof_idx) + 1, // n_q_points_1d of cell operation
        [&](const unsigned int cell,
            const unsigned int quad) -> const Tensor<1, dim, VectorizedArray<double>> & {
          return const_cast<const Tensor<1, dim, VectorizedArray<double>> &>(
            evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                     cell +
                                   quad]);
        });

      scratch_data.get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);
      scratch_data.get_pcout() << "    | evapor: |u|2 = " << evaporation_velocity.l2_norm()
                               << std::endl;

      evaporation_velocity.zero_out_ghosts();
    }

    static void
    compute_mass_balance_source_term_sharp(const ScratchData<dim> &scratch_data,
                                           VectorType &            mass_balance_rhs,
                                           const VectorType &      evaporative_mass_flux,
                                           const VectorType &      level_set_vector,
                                           const double            rho_l,
                                           const double            rho_g,
                                           const unsigned int      evapor_dof_idx,
                                           const unsigned int      pressure_dof_idx)
    {
      (void)scratch_data;
      (void)mass_balance_rhs;
      (void)evaporative_mass_flux;
      (void)level_set_vector;
      (void)rho_l;
      (void)rho_g;
      (void)evapor_dof_idx;
      (void)pressure_dof_idx;

#ifdef MELT_POOL_DG_WITH_ADAFLO
      if constexpr (dim > 1) // @todo: otherwise i am getting a compiling error atm
        {
          auto surface_quad =
            QGauss<dim - 1>(scratch_data.get_dof_handler(pressure_dof_idx).get_fe().degree + 1);

          evaporative_mass_flux.update_ghost_values();
          const unsigned int n_subdivisions = 3;

          GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(
            scratch_data.get_mapping(),
            scratch_data.get_dof_handler(pressure_dof_idx).get_fe(), // todo
            n_subdivisions);

          FEPointEvaluation<1, dim> mass_flux(
            scratch_data.get_mapping(), scratch_data.get_dof_handler(evapor_dof_idx).get_fe());
          FEPointEvaluation<1, dim> rhs_continuity(
            scratch_data.get_mapping(), scratch_data.get_dof_handler(pressure_dof_idx).get_fe());

          std::vector<double>                  buffer;
          std::vector<types::global_dof_index> local_dof_indices;

          for (const auto &cell :
               scratch_data.get_dof_handler(pressure_dof_idx).active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  // determine if cell is cut by the interface and if yes, determine the quadrature
                  // point location and weight
                  const auto [points, weights] =
                    [&]() -> std::tuple<std::vector<Point<dim>>, std::vector<double>> {
                    // determine points and cells of aux surface triangulation
                    std::vector<Point<dim>>          surface_vertices;
                    std::vector<::CellData<dim - 1>> surface_cells;

                    // run marching cube algorithm
                    mc.process_cell(cell, level_set_vector, 0.0, surface_vertices, surface_cells);

                    if (surface_vertices.size() == 0)
                      return {}; // cell is not cut by interface -> no quadrature points have the be
                                 // determined

                    std::vector<Point<dim>> points;
                    std::vector<double>     weights;

                    // create aux triangulation of subcells
                    Triangulation<dim - 1, dim> surface_triangulation;
                    surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

                    FE_Nothing<dim - 1, dim> fe;
                    FEValues<dim - 1, dim>   fe_eval(fe,
                                                   surface_quad,
                                                   update_quadrature_points | update_JxW_values);

                    // loop over all cells ...
                    for (const auto &sub_cell : surface_triangulation.active_cell_iterators())
                      {
                        fe_eval.reinit(sub_cell);

                        // ... and collect quadrature points and weights
                        for (const auto q : fe_eval.quadrature_point_indices())
                          {
                            points.emplace_back(
                              scratch_data.get_mapping().transform_real_to_unit_cell(
                                cell, fe_eval.quadrature_point(q)));
                            weights.emplace_back(fe_eval.JxW(q));
                          }
                      }
                    return {points, weights};
                  }();


                  if (points.size() == 0)
                    continue; // cell is not cut but the interface -> nothing to do

                  // evaluate rhs term
                  local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
                  buffer.resize(cell->get_fe().n_dofs_per_cell());
                  cell->get_dof_indices(local_dof_indices);

                  const unsigned int n_points = points.size();

                  const ArrayView<const Point<dim>> unit_points(points.data(), n_points);
                  const ArrayView<const double>     JxW(weights.data(), n_points);

                  // gather mass_flux
                  scratch_data.get_constraint(evapor_dof_idx)
                    .get_dof_values(evaporative_mass_flux,
                                    local_dof_indices.begin(),
                                    buffer.begin(),
                                    buffer.end());

                  // evaluate mass_flux
                  mass_flux.evaluate(cell,
                                     unit_points,
                                     make_array_view(buffer),
                                     EvaluationFlags::values);

                  for (unsigned int q = 0; q < n_points; ++q)
                    rhs_continuity.submit_value(mass_flux.get_value(q) * (1. / rho_l - 1. / rho_g) *
                                                  JxW[q],
                                                q);

                  // integrate rhs term of the continuity equation
                  rhs_continuity.integrate(cell, unit_points, buffer, EvaluationFlags::values);

                  scratch_data.get_constraint(evapor_dof_idx)
                    .distribute_local_to_global(buffer, local_dof_indices, mass_balance_rhs);
                }
            }

          mass_balance_rhs.compress(VectorOperation::add);
          mass_balance_rhs.update_ghost_values();
          evaporative_mass_flux.zero_out_ghosts();
        }
      else
        {
          Assert(false, ExcNotImplemented());
        }
#endif
    }
  };
} // namespace MeltPoolDG::Evaporation
