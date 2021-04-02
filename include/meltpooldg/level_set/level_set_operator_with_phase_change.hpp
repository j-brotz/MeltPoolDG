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
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  static std::map<std::string, double> get_generalized_theta = {
    {"explicit_euler", 0.0},
    {"implicit_euler", 1.0},
    {"crank_nicolson", 0.5},
  };

  using namespace dealii;

  template <int dim, typename number = double>
  class LevelSetOperatorWithPhaseChange
    : public OperatorBase<number,
                          LinearAlgebra::distributed::Vector<number>,
                          LinearAlgebra::distributed::Vector<number>>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
    using VectorizedArrayType = VectorizedArray<number>;
    using vector              = Tensor<1, dim, VectorizedArray<number>>;
    using scalar              = VectorizedArray<number>;

    const ScratchData<dim> &    scratch_data;
    const VectorType &          advection_velocity;
    const VectorType &          evapor_velocity;
    const LevelSetData<number> &data;
    const unsigned int          ls_hanging_nodes_dof_idx;
    const unsigned int          temp_dof_idx;
    const unsigned int          flow_vel_dof_idx;
    const unsigned int          evapor_vel_dof_idx;
    double                      theta;

  public:
    // clang-format off
    LevelSetOperatorWithPhaseChange( const ScratchData<dim> &scratch_data_in, 
                                     const VectorType                     &advection_velocity_in,
                                     const VectorType                     &evapor_velocity_in,
                                     const LevelSetData<number> &data_in,
                                     const unsigned int                   ls_dof_idx_in,
                                     const unsigned int                   ls_hanging_nodes_dof_idx_in,
                                     const unsigned int                   ls_quad_idx_in, 
                                     const unsigned int                   temp_dof_idx_in,
                                     const unsigned int                   flow_vel_dof_idx_in,
                                     const unsigned int                   evapor_vel_dof_idx_in
                                )
    : scratch_data        ( scratch_data_in       )
    , advection_velocity  ( advection_velocity_in )
    , evapor_velocity     ( evapor_velocity_in    )
    , data                ( data_in               )
    , ls_hanging_nodes_dof_idx( ls_hanging_nodes_dof_idx_in )
    , temp_dof_idx        ( temp_dof_idx_in       )
    , flow_vel_dof_idx    ( flow_vel_dof_idx_in   )
    , evapor_vel_dof_idx  ( evapor_vel_dof_idx_in )
    {
      this->reset_indices(ls_dof_idx_in, ls_quad_idx_in);
      /*
       *  convert the user input to the generalized theta parameter
       */
      if (get_generalized_theta.find(data.time_integration_scheme) != get_generalized_theta.end()) 
        theta = get_generalized_theta[data.time_integration_scheme];
      else
        AssertThrow(false, ExcMessage("Level set operator with mass change: Requested time integration scheme not supported."))
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
      AssertThrow(this->d_tau > 0.0, ExcMessage("advection diffusion operator: d_tau must be set"));

      advection_velocity.update_ghost_values();
      evapor_velocity.update_ghost_values();

      scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
        [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
          FECellIntegrator<dim, 1, number> advected_field(matrix_free,
                                                          this->dof_idx,
                                                          this->quad_idx);

          FECellIntegrator<dim, dim, number> flow_vel(matrix_free,
                                                      flow_vel_dof_idx,
                                                      this->quad_idx);
          FECellIntegrator<dim, dim, number> evapor_vel(matrix_free,
                                                        evapor_vel_dof_idx,
                                                        this->quad_idx);
          if constexpr (dim > 1)
            {
              for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
                {
                  advected_field.reinit(cell);
                  advected_field.gather_evaluate(src, true, true);

                  flow_vel.reinit(cell);
                  flow_vel.read_dof_values_plain(advection_velocity);
                  flow_vel.evaluate(true, true);

                  evapor_vel.reinit(cell);
                  evapor_vel.gather_evaluate(evapor_velocity, true, true);

                  for (unsigned int q_index = 0; q_index < advected_field.n_q_points; ++q_index)
                    {
                      const scalar phi      = advected_field.get_value(q_index);
                      const vector grad_phi = advected_field.get_gradient(q_index);

                      const scalar velocity_grad_phi =
                        scalar_product(flow_vel.get_value(q_index) + evapor_vel.get_value(q_index),
                                       grad_phi);
                      advected_field.submit_value(
                        phi + this->d_tau * theta * velocity_grad_phi
                        // non zero divergence of the fluid field
                        //+ this->d_tau * theta * phi * flow_vel.get_divergence(q_index)
                        //+ this->d_tau * theta * phi * evapor_vel.get_divergence(q_index)
                        ,
                        q_index);
                    }
                  advected_field.integrate_scatter(true, false, dst);
                }
            }
          else
            {
              (void)cell_range;
            }
        },
        dst,
        src,
        true);
      advection_velocity.zero_out_ghosts();
      evapor_velocity.zero_out_ghosts();
    }

    void
    create_rhs(VectorType &dst, const VectorType &src) const override
    {
      advection_velocity.update_ghost_values();
      evapor_velocity.update_ghost_values();
      /*
       * This function creates the rhs of the advection-diffusion problem. When inhomogeneous
       * dirichlet BC are prescribed, the rhs vector is modified including BC terms. Thus the src
       * vector will NOT be zeroed during the cell_loop.
       */
      AssertThrow(this->d_tau > 0.0, ExcMessage("advection diffusion operator: d_tau must be set"));

      scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
        [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
          FECellIntegrator<dim, 1, number, VectorizedArrayType> advected_field(matrix_free,
                                                                               this->dof_idx,
                                                                               this->quad_idx);
          FECellIntegrator<dim, dim, number>                    flow_vel(matrix_free,
                                                      flow_vel_dof_idx,
                                                      this->quad_idx);

          FECellIntegrator<dim, dim, number> evapor_vel(matrix_free,
                                                        evapor_vel_dof_idx,
                                                        this->quad_idx);
          if constexpr (dim > 1)
            {
              for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
                {
                  advected_field.reinit(cell);
                  advected_field.gather_evaluate(src, true, true);

                  flow_vel.reinit(cell);
                  flow_vel.read_dof_values_plain(advection_velocity);
                  flow_vel.evaluate(true, true);

                  evapor_vel.reinit(cell);
                  evapor_vel.gather_evaluate(evapor_velocity, true, true);

                  for (unsigned int q_index = 0; q_index < advected_field.n_q_points; ++q_index)
                    {
                      scalar       phi      = advected_field.get_value(q_index);
                      const vector grad_phi = advected_field.get_gradient(q_index);

                      const scalar velocity_grad_phi =
                        scalar_product(flow_vel.get_value(q_index) + evapor_vel.get_value(q_index),
                                       grad_phi);

                      advected_field.submit_value(
                        phi - this->d_tau * (1. - theta) * velocity_grad_phi
                        // non zero divergence of the fluid field
                        //- this->d_tau * (1-theta) * phi * flow_vel.get_divergence(q_index)
                        //- this->d_tau * (1-theta) * phi * evapor_vel.get_divergence(q_index)
                        ,
                        q_index);
                    }

                  advected_field.integrate_scatter(true, false, dst);
                }
            }
          else
            {
              (void)macro_cells;
            }
        },
        dst,
        src,
        false); // rhs should not be zeroed out in order to consider inhomogeneous dirichlet BC
      advection_velocity.zero_out_ghosts();
      evapor_velocity.zero_out_ghosts();
    }

    void
    solve(const double dt, VectorType &advected_field)
    {
      VectorType src, rhs;
      scratch_data.initialize_dof_vector(rhs, this->dof_idx);
      scratch_data.initialize_dof_vector(src, this->dof_idx);

      int iter = 0;

      this->set_time_increment(dt);

      if (data.do_matrix_free)
        {
          /*
           * apply dirichlet boundary values
           */
          this->create_rhs_and_apply_dirichlet_mf(
            rhs, advected_field, scratch_data, this->dof_idx, ls_hanging_nodes_dof_idx);

          iter =
            LinearSolve<VectorType, SolverGMRES<VectorType>, OperatorBase<double>>::solve(*this,
                                                                                          src,
                                                                                          rhs);
        }
      else
        AssertThrow(false, ExcNotImplemented());

      scratch_data.get_constraint(this->dof_idx).distribute(src);

      advected_field.copy_locally_owned_data_from(src);
      advected_field.update_ghost_values();

      scratch_data.get_pcout(1) << "| GMRES: i=" << std::setw(5) << std::left << iter;
      scratch_data.get_pcout() << "\t |ϕ|2 = " << std::setw(15) << std::left
                               << std::setprecision(10)
                               << VectorTools::compute_L2_norm<dim>(advected_field,
                                                                    scratch_data,
                                                                    this->dof_idx,
                                                                    this->quad_idx)
                               << std::endl;
      advected_field.zero_out_ghosts();
    }
  };
} // namespace MeltPoolDG::LevelSet
