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

  /*
   *  This operator computes the residual and its consistent tangent of the time-discretized heat
   * equation
   *
   *               /                              \     /                                  \
   *  R(T^(n+1)) = | w, ρ c_p ( T^(n+1)- T^(n+1)) |  -  | ∇w, ρ c_p T^(n+1) u - k ∇T^(n+1)) |
   *               \                              /     \                                  /
   *                                               Ω                                        Ω
   *               /    _   \     /    _  \
   *             + | w, q_s |  -  | w, q  | = 0
   *               \        /     \       /
   *                         Ω             Γ
   *                                        N
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

    const VectorType &              temperature;
    std::vector<types::boundary_id> bc_radiation_indices;
    std::vector<types::boundary_id> bc_convection_indices;
    unsigned int                    vel_dof_idx = 0; //@todo: fill

    VectorType heat_source; //@todo: fill
    VectorType velocity;    //@todo: fill

    bool do_velocity = false;

  public:
    HeatOperator(const ScratchData<dim> &              scratch_data_in,
                 const HeatData<number> &              data_in,
                 const std::vector<types::boundary_id> bc_radiation_in,
                 const std::vector<types::boundary_id> bc_convection_in,
                 const unsigned int                    temp_dof_idx_in,
                 const unsigned int                    temp_quad_idx_in,
                 const VectorType &                    temperature_in)
      // clang-format off
    : scratch_data         ( scratch_data_in  )
    , data                 ( data_in          )
    , temp_dof_idx         ( temp_dof_idx_in  )
    , temp_quad_idx        ( temp_quad_idx_in )
    , temperature          ( temperature_in   )
    , bc_radiation_indices ( bc_radiation_in  )
    , bc_convection_indices( bc_convection_in )
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
              temp_vals.submit_value(data.density * data.capacity * temp_vals.get_value(q_index),
                                     q_index);


              auto val_grad = -data.conductivity * MeltPoolDG::VectorTools::convert_to_vector<dim>(
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
      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free, this->dof_idx, this->quad_idx);
      FEFaceIntegrator<dim, 1, number> temp_vals(matrix_free, this->dof_idx, this->quad_idx);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          dQ_dT.reinit(face);
          dQ_dT.gather_evaluate(src, true, false);

          temp_vals.reinit(face);
          temp_vals.gather_evaluate(temperature, true, false);

          types::boundary_id bc_index = matrix_free.get_boundary_id(face);

          for (unsigned int q_index = 0; q_index < dQ_dT.n_q_points; ++q_index)
            {
              auto inc_temp_vals_at_q = dQ_dT.get_value(q_index);

              VectorizedArray<double> temp = 0;
              if (std::find(bc_radiation_indices.begin(), bc_radiation_indices.end(), bc_index) !=
                  bc_radiation_indices.end())
                temp += 4 * data.emissivity * stefan_boltzmann *
                        pow<double>(temp_vals.get_value(q_index), 3) * inc_temp_vals_at_q;
              if (std::find(bc_convection_indices.begin(), bc_convection_indices.end(), bc_index) !=
                  bc_convection_indices.end())
                temp += data.convection_coefficient * inc_temp_vals_at_q;

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
      (void)src;
      FECellIntegrator<dim, 1, number> temp_vals(matrix_free, this->dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> temp_vals_old(matrix_free, this->dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> heat_source_vals(matrix_free, this->dof_idx, this->quad_idx);
      // if (do_velocity)
      // FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          temp_vals.reinit(cell);
          temp_vals.gather_evaluate(temperature, true, true);

          temp_vals_old.reinit(cell);
          temp_vals_old.gather_evaluate(src, true, false);

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
                                               temp_vals_old.get_value(q_index)) +
                                            heat_source_vals.get_value(q_index)),
                                     q_index);

              auto val_grad = -data.conductivity * MeltPoolDG::VectorTools::convert_to_vector<dim>(
                                                     temp_vals.get_gradient(q_index));

              // if (do_velocity)
              //{
              // val_grad += -this->d_tau * data.density * data.capacity *
              // temp_vals.get_value(q_index) * MeltPoolDG::VectorTools::convert_to_vector<dim>(
              // velocity_vals.get_value(q_index));
              //}
              temp_vals.submit_gradient(-1.0 * val_grad, q_index);
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
                      [[maybe_unused]] const VectorType &    src,
                      std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free, this->dof_idx, this->quad_idx);
      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          dQ_dT.reinit(face);
          dQ_dT.gather_evaluate(temperature, true, false);

          types::boundary_id bc_index = matrix_free.get_boundary_id(face);

          for (unsigned int q_index = 0; q_index < dQ_dT.n_q_points; ++q_index)
            {
              auto temp_vals = dQ_dT.get_value(q_index);

              VectorizedArray<double> temp = 0;
              if (std::find(bc_radiation_indices.begin(), bc_radiation_indices.end(), bc_index) !=
                  bc_radiation_indices.end())
                temp += data.emissivity * stefan_boltzmann *
                        (pow<double>(temp_vals, 4) - std::pow(data.temperature_infinity, 4));
              if (std::find(bc_convection_indices.begin(), bc_convection_indices.end(), bc_index) !=
                  bc_convection_indices.end())
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
    create_rhs(VectorType &dst, const VectorType &src) const override
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
        true /*zero dst vector*/); // @todo ?
      AssertThrow(false, ExcNotImplemented());
    }
  };
} // namespace MeltPoolDG::HeatEquation
