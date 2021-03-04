/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>

namespace MeltPoolDG::HeatEquation
{
  using namespace dealii;

  /*
   *  This operator computes the residual and its consistent tangent of the time-discretized heat
   * equation
   *
   *                 1   /                              \     /                                  \
   *  R(T^(n+1)) =  ---  | w, ρ c_p ( T^(n+1)- T^(n+1)) |  - | ∇w, ρ c_p T^(n+1) u - k ∇T^(n+1)) |
   *                dt   \                              /     \                                  /
   *                                                    Ω                                         Ω
   *                 /    _   \     /    _  \
   *             -   | w, q_s |  -  | w, q  | = 0
   *                 \        /     \       /
   *                           Ω             Γ
   *                                            N
   *
   * with the temperature field T^(n+1), test function w, the density ρ, the specific heat capacity
   * c_p and the conductivity k, source/sink terms q_s and prescribed fluxes q along Neumann
   * boundaries.
   *
   * We assume that the density and the specific heat capacity do not dependent on the temperature.
   *
   *
   */

  template <int dim, typename number = double>
  class HeatOperator : public OperatorBase<number,
                                           LinearAlgebra::distributed::Vector<number>,
                                           LinearAlgebra::distributed::Vector<number>>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using VectorizedArrayType = VectorizedArray<number>;

    const ScratchData<dim> &scratch_data;
    const HeatData<number> &data;
    const unsigned int      temp_dof_idx;
    const unsigned int      temp_quad_idx;

    const double stefan_boltzmann = 5.67e-8; // W/(mK^4) // @todo move -- where?

    const VectorType &temperature;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
                                    neumann_bc; //@todo find a nice way to provide BC
    std::vector<types::boundary_id> bc_radiation_indices;
    std::vector<types::boundary_id> bc_convection_indices;
    unsigned int                    vel_dof_idx = 0; //@todo: fill

    const VectorType &heat_source;
    VectorType        velocity; //@todo: fill

    bool do_velocity = false;

  public:
    HeatOperator(const std::shared_ptr<BoundaryConditions<dim>> &bc,
                 const ScratchData<dim> &                        scratch_data_in,
                 const HeatData<number> &                        data_in,
                 const unsigned int                              temp_dof_idx_in,
                 const unsigned int                              temp_quad_idx_in,
                 const VectorType &                              temperature_in,
                 const VectorType &                              heat_source_in)
      // clang-format off
    : scratch_data         ( scratch_data_in  )
    , data                 ( data_in          )
    , temp_dof_idx         ( temp_dof_idx_in  )
    , temp_quad_idx        ( temp_quad_idx_in )
    , temperature          ( temperature_in   )
    , heat_source          ( heat_source_in   ) 
    {
      if (bc)
      {
        bc_convection_indices = bc->convection_bc;
        bc_radiation_indices = bc->radiation_bc;
        neumann_bc = bc->neumann_bc;
      }
      this->reset_indices(temp_dof_idx_in, temp_quad_idx_in);
    }
    // clang-format on

    void
    assemble_matrixbased([[maybe_unused]] const VectorType &advected_field_old,
                         [[maybe_unused]] SparseMatrixType &matrix,
                         [[maybe_unused]] VectorType &      rhs) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    /*
     *    matrix-free implementation
     */
    void
    vmult(VectorType &dst, const VectorType &src /*solution_update*/) const override
    {
      AssertThrow(this->d_tau > 0.0, ExcMessage("advection diffusion operator: d_tau must be set"));

      scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
        [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
          tangent_cell_loop(matrix_free, dst, src, cell_range);
        }, // cell loop
        [&]([[maybe_unused]] const auto &matrix_free,
            [[maybe_unused]] auto &      dst,
            [[maybe_unused]] const auto &src,
            [[maybe_unused]] auto        face_range) { /*do nothing*/ }, // internal face loop
        [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
          tangent_boundary_loop(matrix_free, dst, src, face_range); // boundary face loop
        },
        dst,
        src,
        true /*zero dst vector*/);
    }

    void
    tangent_cell_loop(const MatrixFree<dim, number> &        matrix_free,
                      VectorType &                           dst,
                      const VectorType &                     src,
                      std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FECellIntegrator<dim, 1, number> temp_vals(matrix_free, this->dof_idx, this->quad_idx);

      // if (do_velocity)
      // FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          temp_vals.reinit(cell);
          temp_vals.gather_evaluate(src, true, true);

          // if (do_velocity)
          //{
          // velocity_vals.reinit(cell);
          // velocity_vals.gather_evaluate(velocity, true, false);
          //}

          for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
            {
              temp_vals.submit_value(data.density * data.capacity / this->d_tau *
                                       temp_vals.get_value(q_index),
                                     q_index);

              auto val_grad = data.conductivity * MeltPoolDG::VectorTools::convert_to_vector<dim>(
                                                    temp_vals.get_gradient(q_index));
              // if (do_velocity)
              //{
              // val_grad += -this->d_tau * data.density * data.capacity *
              // temp_vals.get_value(q_index) * MeltPoolDG::VectorTools::convert_to_vector<dim>(
              // velocity_vals.get_value(q_index));
              //}

              temp_vals.submit_gradient(val_grad, q_index);
            }
          temp_vals.integrate_scatter(true, true, dst);
        }
    }

    /*
     * compute the tangent of Robin-type boundary conditions for convection and radiation
     */
    void
    tangent_boundary_loop(const MatrixFree<dim, number> &        matrix_free,
                          VectorType &                           dst,
                          const VectorType &                     src,
                          std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                             true /*is_interior_face*/,
                                             this->dof_idx,
                                             this->quad_idx);
      FEFaceIntegrator<dim, 1, number> temp_vals(matrix_free,
                                                 true /*is_interior_face*/,
                                                 this->dof_idx,
                                                 this->quad_idx);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          dQ_dT.reinit(face);
          dQ_dT.gather_evaluate(src, true, false);

          temp_vals.reinit(face);
          temp_vals.gather_evaluate(temperature, true, false);

          types::boundary_id bc_index = matrix_free.get_boundary_id(face);

          bool do_radiation =
            (std::find(bc_radiation_indices.begin(), bc_radiation_indices.end(), bc_index) !=
             bc_radiation_indices.end());
          bool do_convection =
            (std::find(bc_convection_indices.begin(), bc_convection_indices.end(), bc_index) !=
             bc_convection_indices.end());

          for (unsigned int q_index = 0; q_index < dQ_dT.n_q_points; ++q_index)
            {
              auto inc_temp_vals_at_q = dQ_dT.get_value(q_index);

              VectorizedArray<double> temp = 0;

              if (do_convection)
                temp += data.convection_coefficient * inc_temp_vals_at_q;
              if (do_radiation)
                temp += 4 * data.emissivity * stefan_boltzmann *
                        pow<double>(temp_vals.get_value(q_index), 3) * inc_temp_vals_at_q;

              dQ_dT.submit_value(-temp, q_index);
            }
          dQ_dT.integrate_scatter(true, false, dst);
        }
    }

    void
    rhs_cell_loop(const MatrixFree<dim, number> &        matrix_free,
                  VectorType &                           dst,
                  const VectorType &                     src, /* temperature_old*/
                  std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FECellIntegrator<dim, 1, number> temp_vals(matrix_free, this->dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> temp_vals_old(matrix_free, this->dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> heat_source_vals(matrix_free, this->dof_idx, this->quad_idx);
      // if (do_velocity)
      // FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);

      MeltPoolDG::VectorTools::update_ghost_values(temperature, src, heat_source);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          temp_vals.reinit(cell);
          temp_vals.read_dof_values_plain(temperature);
          temp_vals.evaluate(true, true);

          temp_vals_old.reinit(cell);
          temp_vals_old.read_dof_values_plain(src);
          temp_vals_old.evaluate(true, false);

          heat_source_vals.reinit(cell);
          heat_source_vals.read_dof_values_plain(heat_source);
          heat_source_vals.evaluate(true, false);

          // if (do_velocity)
          //{
          // velocity_vals.reinit(cell);
          // velocity_vals.gather_evaluate(velocity, true, false);
          //}

          for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
            {
              temp_vals.submit_value(-1. * (data.density * data.capacity *
                                              (temp_vals.get_value(q_index) -
                                               temp_vals_old.get_value(q_index)) /
                                              this->d_tau -
                                            heat_source_vals.get_value(q_index)),
                                     q_index); // negative sign since residual is moved to rhs

              auto val_grad = data.conductivity * MeltPoolDG::VectorTools::convert_to_vector<dim>(
                                                    temp_vals.get_gradient(q_index));

              // if (do_velocity)
              //{
              // val_grad += -data.density * data.capacity *
              // temp_vals.get_value(q_index) * MeltPoolDG::VectorTools::convert_to_vector<dim>(
              // velocity_vals.get_value(q_index));
              //}
              temp_vals.submit_gradient(-1.0 * val_grad, q_index);
            }
          temp_vals.integrate_scatter(true, true, dst);
        }
      MeltPoolDG::VectorTools::zero_out_ghosts(temperature, src, heat_source);
    }
    /*
     * compute the RHS due to Robin-type boundary conditions for convection and radiation
     *
     * @todo: add description
     */
    void
    rhs_boundary_loop(const MatrixFree<dim, number> &        matrix_free,
                      VectorType &                           dst,
                      [[maybe_unused]] const VectorType &    src,
                      std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                             true /* is interior face*/,
                                             this->dof_idx,
                                             this->quad_idx);
      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          dQ_dT.reinit(face);
          dQ_dT.gather_evaluate(temperature, true, false);

          const types::boundary_id bc_index = matrix_free.get_boundary_id(face);

          bool do_neumann = neumann_bc.find(bc_index) != neumann_bc.end();

          bool do_radiation =
            (std::find(bc_radiation_indices.begin(), bc_radiation_indices.end(), bc_index) !=
             bc_radiation_indices.end());
          bool do_convection =
            (std::find(bc_convection_indices.begin(), bc_convection_indices.end(), bc_index) !=
             bc_convection_indices.end());

          AssertThrow(!(do_neumann && (do_radiation || do_convection)),
                      ExcMessage(
                        "It is not allowed to specify both Neumann and radiation and/or convection"
                        " boundary conditions at the same face."));

          if (!do_neumann && !do_radiation && !do_convection)
            continue;

          for (unsigned int q_index = 0; q_index < dQ_dT.n_q_points; ++q_index)
            {
              auto temp_vals = dQ_dT.get_value(q_index);

              VectorizedArray<double> temp = 0;
              if (do_neumann)
                {
                  auto quad_point = dQ_dT.quadrature_point(q_index);
                  for (unsigned int v = 0;
                       v < scratch_data.get_matrix_free().n_active_entries_per_face_batch(face);
                       ++v)
                    {
                      // @todo: This implementation is extremely odd. Is there another way to call a
                      // Function with Point<dim, VectorizedArray>?
                      Point<dim> quad_point_v;

                      for (unsigned int d = 0; d < dim; ++d)
                        quad_point_v[d] = quad_point[d][v];

                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          temp[v] += neumann_bc.at(bc_index)->value(quad_point_v, d) *
                                     dQ_dT.get_normal_vector(q_index)[d][v];
                        }
                    }
                }
              if (do_radiation)
                temp += data.emissivity * stefan_boltzmann *
                        (pow<double>(temp_vals, 4) - std::pow(data.temperature_infinity, 4));
              if (do_convection)
                temp += data.convection_coefficient * (temp_vals - data.temperature_infinity);

              dQ_dT.submit_value(-temp, q_index);
            }
          dQ_dT.integrate_scatter(true, false, dst);
        }
    }

    /**
     * -R(T)
     */
    void
    create_rhs(VectorType &dst, const VectorType &src /*temperature_old*/) const override
    {
      scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
        [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
          rhs_cell_loop(matrix_free, dst, src, cell_range);
        },
        [&]([[maybe_unused]] const auto &matrix_free,
            [[maybe_unused]] auto &      dst,
            [[maybe_unused]] const auto &src,
            [[maybe_unused]] auto        face_range) { /*do nothing*/ }, // face loop
        [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
          rhs_boundary_loop(matrix_free, dst, src, face_range);
        },
        dst,
        src,
        false /*zero dst vector*/); // should not be zeroed out in case of boundary conditions
    }
  };
} // namespace MeltPoolDG::HeatEquation
