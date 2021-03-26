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

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  /*
   * This operator computes the residual and its consistent tangent of the discretized heat
   * equation
   *
   *                    1   /                                     \
   *  R(T_b^(n+1)) =   ---  | N_a, ρ c_p ( N_b T_b^(n+1) - T^(n)) |
   *                   dt   \                                     /
   *                                                               Ω
   *                 /                             \
   *               + | N_a, ρ c_p u ∇N_b T_b^(n+1) |
   *                 \                             /
   *                                                Ω
   *                 /                              \
   *               + | N_a, ρ c_p N_b T_b^(n+1) ∇·u |    (this term is not yet considered @todo)
   *                 \                              /
   *                                                 Ω
   *                 /                         \
   *               + | ∇N_a, k ∇N_b T_b^(n+1)) |
   *                 \                         /
   *                                            Ω
   *                 /    _   \     /    _  \
   *               - | w, q_s |  -  | w, q  | = 0
   *                 \        /     \       /
   *                           Ω             Γ
   *                                          N
   *
   *
   *  dR(T^(n+1))    1   /                \     /                              \
   *  ----------- = ---  | N_a, ρ c_p N_b |  +  | N_a, ρ c_p  ( ∇N_b u + ∇·u ) |
   *  dT_b^(n+1)     dt  \                /     \                              /
   *                                       Ω                                    Ω
   *                  /              \
   *              +   | ∇N_a, k ∇N_b |
   *                  \              /
   *                                  Ω
   *                            _
   *                 /        d q       \
   *              -  | N_a, ---------   |
   *                 \       dT_b^(n+1) /
   *                                     Γ
   *                                      N
   *
   * with shape functions N_a and N_b, nodal temperature values T_b^(n+1), the density ρ, the
   * specific heat capacity c_p and the conductivity k, source/sink terms q_s and prescribed
   * fluxes q along Neumann boundaries. The heat flux may result from radiative losses
   *
   *  q = σ ϵ (T^4-T∞^4)
   *
   * with the Stefan-Boltzmann constant σ, the emissivity ϵ and the temperature of the surroundings
   * T∞ as well as convective losses
   *
   *  q = α (T-T∞)
   *
   * with the convection coeficient alpha.
   *
   * We assume that the density and the specific heat capacity do not dependent on the temperature.
   */

  template <int dim, typename number = double>
  class HeatTransferOperator : public OperatorBase<number,
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

    const VectorType &heat_source;

    // optional flow velocity for internal convection
    const unsigned int vel_dof_idx;
    const VectorType * velocity;

    // optional level-set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    const VectorType * level_set_as_heaviside;


  public:
    HeatTransferOperator(const std::shared_ptr<BoundaryConditions<dim>> &bc,
                         const ScratchData<dim> &                        scratch_data_in,
                         const HeatData<number> &                        data_in,
                         const unsigned int                              temp_dof_idx_in,
                         const unsigned int                              temp_quad_idx_in,
                         const VectorType &                              temperature_in,
                         const VectorType &                              heat_source_in,
                         const unsigned int                              vel_dof_idx_in = 0,
                         const VectorType *                              velocity_in    = nullptr,
                         const unsigned int                              ls_dof_idx_in  = 0,
                         const VectorType *level_set_as_heaviside_in                    = nullptr)
      // clang-format off
    : scratch_data           ( scratch_data_in           )
    , data                   ( data_in                   )
    , temp_dof_idx           ( temp_dof_idx_in           )
    , temp_quad_idx          ( temp_quad_idx_in          )
    , temperature            ( temperature_in            )
    , heat_source            ( heat_source_in            ) 
    , vel_dof_idx            ( vel_dof_idx_in            )
    , velocity               ( velocity_in               )
    , ls_dof_idx             ( ls_dof_idx_in             )
    , level_set_as_heaviside ( level_set_as_heaviside_in )
    {
      AssertThrow(!level_set_as_heaviside || (velocity && level_set_as_heaviside) ,
          ExcMessage("Two-phase flow must come with a velocity! Abort..."));

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

      MeltPoolDG::VectorTools::update_ghost_values(temperature);
      if (velocity)
        MeltPoolDG::VectorTools::update_ghost_values(*velocity);
      if (level_set_as_heaviside)
        MeltPoolDG::VectorTools::update_ghost_values(*level_set_as_heaviside);

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
      MeltPoolDG::VectorTools::zero_out_ghosts(temperature);
      if (velocity)
        MeltPoolDG::VectorTools::zero_out_ghosts(*velocity);
      if (level_set_as_heaviside)
        MeltPoolDG::VectorTools::zero_out_ghosts(*level_set_as_heaviside);
    }

    void
    tangent_cell_loop(const MatrixFree<dim, number> &        matrix_free,
                      VectorType &                           dst,
                      const VectorType &                     src,
                      std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FECellIntegrator<dim, 1, number>   temp_vals(matrix_free, temp_dof_idx, this->quad_idx);
      FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, this->quad_idx);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          temp_vals.reinit(cell);
          temp_vals.gather_evaluate(src, true, true);

          if (velocity)
            {
              velocity_vals.reinit(cell);
              velocity_vals.gather_evaluate(*velocity, true, false);
            }

          if (level_set_as_heaviside)
            {
              ls_vals.reinit(cell);
              ls_vals.gather_evaluate(*level_set_as_heaviside, true, false);
            }

          for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
            {
              VectorizedArrayType conductivity, density, capacity;
              if (level_set_as_heaviside)
                std::tie(density, capacity, conductivity) =
                  get_two_phase_material_parameters(ls_vals.get_value(q_index));
              else
                {
                  density      = data.density;
                  capacity     = data.capacity;
                  conductivity = data.conductivity;
                }

              auto val = density * capacity * this->d_tau_inv * temp_vals.get_value(q_index);

              if (velocity)
                {
                  val += density * capacity * temp_vals.get_gradient(q_index) *
                         velocity_vals.get_value(q_index);
                }

              temp_vals.submit_value(val, q_index);

              auto val_grad = conductivity * temp_vals.get_gradient(q_index);

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
                                             temp_dof_idx,
                                             this->quad_idx);
      FEFaceIntegrator<dim, 1, number> temp_vals(matrix_free,
                                                 true /*is_interior_face*/,
                                                 temp_dof_idx,
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
                temp += 4. * data.emissivity * stefan_boltzmann *
                        pow<double>(temp_vals.get_value(q_index), 3) * inc_temp_vals_at_q;

              dQ_dT.submit_value(temp, q_index);
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
      FECellIntegrator<dim, 1, number> temp_vals(matrix_free, temp_dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> temp_vals_old(matrix_free, temp_dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number> heat_source_vals(matrix_free, temp_dof_idx, this->quad_idx);
      FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);
      FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, this->quad_idx);


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

          if (velocity)
            {
              velocity_vals.reinit(cell);
              velocity_vals.read_dof_values_plain(*velocity);
              velocity_vals.evaluate(true, false);
            }

          if (level_set_as_heaviside)
            {
              ls_vals.reinit(cell);
              ls_vals.read_dof_values_plain(*level_set_as_heaviside);
              ls_vals.evaluate(true, false);
            }

          for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
            {
              VectorizedArrayType conductivity, density, capacity;
              if (level_set_as_heaviside)
                std::tie(density, capacity, conductivity) =
                  get_two_phase_material_parameters(ls_vals.get_value(q_index));
              else
                {
                  density      = data.density;
                  capacity     = data.capacity;
                  conductivity = data.conductivity;
                }

              auto val = density * capacity *
                           (temp_vals.get_value(q_index) - temp_vals_old.get_value(q_index)) *
                           this->d_tau_inv -
                         heat_source_vals.get_value(q_index);

              if (velocity)
                {
                  val += density * capacity * temp_vals.get_gradient(q_index) *
                         velocity_vals.get_value(q_index);
                }

              temp_vals.submit_value(-val,
                                     q_index); // negative sign since residual is moved to rhs

              auto val_grad = conductivity * temp_vals.get_gradient(q_index);

              temp_vals.submit_gradient(-1.0 * val_grad,
                                        q_index); // -1 since residual is moved to rhs
            }
          temp_vals.integrate_scatter(true, true, dst);
        }
    }
    /*
     * compute the RHS due to Neumann and Robin-type boundary conditions for convection and
     * radiation
     *
     * @todo: add equations
     */
    void
    rhs_boundary_loop(const MatrixFree<dim, number> &        matrix_free,
                      VectorType &                           dst,
                      [[maybe_unused]] const VectorType &    src,
                      std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                             true /* is interior face*/,
                                             temp_dof_idx,
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

                  temp -= MeltPoolDG::VectorTools::evaluate_function_at_vectorized_points(
                            *neumann_bc.at(bc_index), quad_point) *
                          dQ_dT.get_normal_vector(q_index);
                }
              if (do_radiation)
                temp -= data.emissivity * stefan_boltzmann *
                        (pow<double>(temp_vals, 4) - std::pow(data.temperature_infinity, 4));
              if (do_convection)
                temp -= data.convection_coefficient * (temp_vals - data.temperature_infinity);

              dQ_dT.submit_value(temp, q_index);
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
      MeltPoolDG::VectorTools::update_ghost_values(temperature, heat_source);
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
      MeltPoolDG::VectorTools::zero_out_ghosts(temperature, heat_source);
    }

  private:
    /*
     * @return \c std::tuple with density, capacity and conductivity
     */
    std::tuple<VectorizedArrayType, VectorizedArrayType, VectorizedArrayType>
    get_two_phase_material_parameters(const VectorizedArrayType &ls_heaviside_val) const
    {
      // todo Input material parameters for two phase

      VectorizedArrayType weight;
      if (data.variable_properties_over_interface)
        weight = ls_heaviside_val;
      else
        weight = UtilityFunctions::heaviside(ls_heaviside_val, 0.5);

      const auto density      = UtilityFunctions::interpolate(weight,
                                                         /*other dens*/ data.density / 2,
                                                         data.density);
      const auto capacity     = UtilityFunctions::interpolate(weight,
                                                          /*other cap*/ data.capacity / 2,
                                                          data.capacity);
      const auto conductivity = UtilityFunctions::interpolate(weight,
                                                              /*other cond*/ data.conductivity / 2,
                                                              data.conductivity);
      return std::make_tuple(density, capacity, conductivity);
    }
  };
} // namespace MeltPoolDG::Heat
