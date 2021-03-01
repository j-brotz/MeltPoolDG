/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
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

  template <int dim, typename number = double>
  class HeatOperator : public OperatorBase<number,
                                           LinearAlgebra::distributed::Vector<number>,
                                           LinearAlgebra::distributed::Vector<number>>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using VectorizedArrayType = VectorizedArray<number>;

    const ScratchData<dim> &scratch_data;
    const HeatData<number> &data;
    const unsigned int      temp_dof_idx;
    const unsigned int      temp_quad_idx;
    const unsigned int      vel_dof_idx;

    const VectorType &temperature;
    const VectorType &temperature_old;
    VectorType        heat_source; //@todo: fill
    VectorType        velocity;    //@todo: fill

    std::vector<int> bc_radiation_indices;  //@todo: fill
    std::vector<int> bc_convection_indices; //@todo: fill
    double           rho          = 0.0;    //@todo: fill
    double           c_p          = 0.0;    //@todo: fill
    double           conductivity = 0.0;    //@todo: fill

  public:
    HeatOperator(const ScratchData<dim> &scratch_data_in,
                 const HeatData<number> &data_in,
                 const unsigned int      temp_dof_idx_in,
                 const unsigned int      temp_quad_idx_in,
                 const VectorType &      temperature_in,
                 const VectorType &      temperature_old_in)
      // clang-format off
    : scratch_data        ( scratch_data_in    )
    , data                ( data_in            )
    , temp_dof_idx        ( temp_dof_idx_in    )
    , temp_quad_idx       ( temp_quad_idx_in   )
    , temperature         ( temperature_in     )
    , temperature_old     ( temperature_old_in )
    {
      this->reset_indices(temp_dof_idx_in, temp_quad_idx_in);
    }
    // clang-format on

    /*
     *    this is the matrix-based implementation of the rhs and the matrix
     *    @todo: this could be improved by using the WorkStream functionality of dealii
     */

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
    vmult(VectorType &dst, const VectorType &src) const override
    {
      scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            auto        cell_range) { /* todo */ }, // cell loop
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            auto        face_range) { /*do nothing*/ }, // face loop
        [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
          tangent_boundary_loop(matrix_free, dst, src, face_range);
        },
        dst,
        src,
        true /*zero dst vector*/);
    }

    void
    cell_loop(VectorType &dst, const VectorType &src) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    /*
     * compute the tangent of Robin-type boundary conditions for convection and radiation
     */
    void
    tangent_boundary_loop(const MatrixFree<dim, number> &        matrix_free,
                          VectorType &                           dst,
                          const VectorType &                     src,
                          std::pair<unsigned int, unsigned int> &face_range)
    {
      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free, this->dof_idx, this->quad_idx);
      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          dQ_dT.reinit(face);
          dQ_dT.gather_evaluate(temperature, true, false);

          auto bc_index = matrix_free.get_boundary_id(face);

          for (unsigned int q_index = 0; q_index < dQ_dT.n_q_points; ++q_index)
            {
              auto temp_vals = dQ_dT.get_value(q_index);

              VectorizedArray<double> temp = 0;
              for (unsigned int v = 0;
                   v < scratch_data->get_matrix_free().n_active_entries_per_face_batch(face);
                   ++v)
                {
                  if (std::find(bc_radiation_indices.begin(),
                                bc_radiation_indices.end(),
                                bc_index[v]) != bc_radiation_indices.end())
                    temp[v] += 4 * data.emissivity * std::pow(temp_vals[v], 3);
                  if (std::find(bc_convection_indices.begin(),
                                bc_convection_indices.end(),
                                bc_index[v]) != bc_convection_indices.end())
                    temp[v] += data.convection_coefficient *
                               1.0; // @todo: how to get plain value of shape function ?
                }

              dQ_dT.submit_value(temp, q_index);
            }
          dQ_dT.integrate_scatter(true, false, dst);
        }
    }

    void
    rhs_cell_loop(const MatrixFree<dim, number> &        matrix_free,
                  VectorType &                           dst,
                  const VectorType &                     src,
                  std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FECellIntegrator<dim, 1, number> temp_vals(matrix_free, this->dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> temp_vals_old(matrix_free, this->dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> heat_source_vals(matrix_free, this->dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          temp_vals.reinit(cell);
          temp_vals.gather_evaluate(temperature, true, true);

          temp_vals_old.reinit(cell);
          temp_vals_old.gather_evaluate(temperature_old, true, false);

          heat_source_vals.reinit(cell);
          heat_source_vals.read_dof_values_plain(heat_source);
          heat_source_vals.evaluate(true, false);

          velocity_vals.reinit(cell);
          velocity_vals.gather_evaluate(velocity, true, false);

          for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
            {
              temp_vals.submit_value(
                -1. *
                (rho * c_p * (temp_vals.get_value(q_index) - temp_vals_old.get_value(q_index)) +
                 heat_source_vals.get_value(q_index)));
              temp_vals.submit_gradient(
                -1.0 * (-this->d_tau * (rho * c_p * temp_vals.get_value(q_index) *
                                          velocity_vals.get_value(q_index) -
                                        conductivity * temp_vals.get_gradient(q_index))));
            }
          temp_vals.integrate_scatter(true, true, dst);
        }
    }
    /*
     * compute the RHS due to Robin-type boundary conditions for convection and radiation
     *
     * @todo: add description
     */
    void
    rhs_boundary_loop(const MatrixFree<dim, number> &        matrix_free,
                      VectorType &                           dst,
                      const VectorType &                     src,
                      std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free, this->dof_idx, this->quad_idx);
      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          dQ_dT.reinit(face);
          dQ_dT.gather_evaluate(temperature, true, false);

          auto bc_index = matrix_free.get_boundary_id(face);

          for (unsigned int q_index = 0; q_index < dQ_dT.n_q_points; ++q_index)
            {
              auto temp_vals = dQ_dT.get_value(q_index);

              VectorizedArray<double> temp = 0;
              for (unsigned int v = 0;
                   v < scratch_data->get_matrix_free().n_active_entries_per_face_batch(face);
                   ++v)
                {
                  if (std::find(bc_radiation_indices.begin(),
                                bc_radiation_indices.end(),
                                bc_index[v]) != bc_radiation_indices.end())
                    temp[v] += data.emissivity *
                               (std::pow(temp_vals[v], 4) - std::pow(data.temperature_infinity, 4));
                  if (std::find(bc_convection_indices.begin(),
                                bc_convection_indices.end(),
                                bc_index[v]) != bc_convection_indices.end())
                    temp[v] +=
                      data.convection_coefficient * (temp_vals[v] - data.temperature_infinity);
                }

              dQ_dT.submit_value(-temp, q_index);
            }
          dQ_dT.integrate_scatter(true, false, dst);
        }
    }

    /**
     * -R(T)
     */
    void
    create_rhs(VectorType &dst, const VectorType &src) const override
    {
      scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
        [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
          rhs_cell_loop(matrix_free, dst, src, cell_range);
        },
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            auto        face_range) { /*do nothing*/ }, // face loop
        [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
          rhs_boundary_loop(matrix_free, dst, src, face_range);
        },
        dst,
        src,
        true /*zero dst vector*/); // @todo ?
      AssertThrow(false, ExcNotImplemented());
    }
  };
} // namespace MeltPoolDG::HeatEquation
