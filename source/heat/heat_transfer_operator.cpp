#include <meltpooldg/heat/heat_transfer_operator.hpp>
//

#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/utilities/physical_constants.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  HeatTransferOperator<dim, number>::HeatTransferOperator(
    const std::shared_ptr<BoundaryConditions<dim>> &bc,
    const ScratchData<dim> &                        scratch_data_in,
    const HeatData<number> &                        data_in,
    const MaterialData<number> &                    material_data_in,
    const unsigned int                              temp_dof_idx_in,
    const unsigned int                              temp_quad_idx_in,
    const VectorType &                              temperature_in,
    const VectorType &                              temperature_old_in,
    const VectorType &                              heat_source_in,
    const unsigned int                              vel_dof_idx_in,
    const VectorType *                              velocity_in,
    const unsigned int                              ls_dof_idx_in,
    const VectorType *                              level_set_as_heaviside_in)
    : scratch_data(scratch_data_in)
    , data(data_in)
    , material(material_data_in)
    , temp_dof_idx(temp_dof_idx_in)
    , temp_quad_idx(temp_quad_idx_in)
    , temperature(temperature_in)
    , temperature_old(temperature_old_in)
    , heat_source(heat_source_in)
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , ls_dof_idx(ls_dof_idx_in)
    , level_set_as_heaviside(level_set_as_heaviside_in)
  {
    AssertThrow(!level_set_as_heaviside || (velocity && level_set_as_heaviside),
                ExcMessage("Two-phase flow must come with a velocity! Abort..."));
    AssertThrow(
      (material.first.conductivity > 0.0 && material.first.density > 0.0) || data.solidification,
      ExcMessage("The material's conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !level_set_as_heaviside ||
        (material.second.conductivity > 0.0 && material.second.density > 0.0),
      ExcMessage(
        "The secondary material's conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !data.solidification || (material.solid.conductivity > 0.0 && material.solid.density > 0.0),
      ExcMessage(
        "In case of solidification the solid's conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !data.solidification || (material.second.conductivity > 0.0 && material.second.density > 0.0),
      ExcMessage(
        "In case of solidification the liquid's (material.second) conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !data.solidification || material.liquidus_temperature > material.solidus_temperature,
      ExcMessage(
        "In case of solidification the liquidus temperature must be greater than the solidus temperature! Abort..."));

    if (bc)
      {
        bc_convection_indices = bc->convection_bc;
        bc_radiation_indices  = bc->radiation_bc;
        neumann_bc            = bc->neumann_bc;
      }

    this->reset_indices(temp_dof_idx_in, temp_quad_idx_in);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::register_evaporative_mass_flux(
    VectorType * evaporative_mass_flux_in,
    const double latent_heat_of_evaporation_in)
  {
    evaporative_mass_flux      = evaporative_mass_flux_in;
    latent_heat_of_evaporation = latent_heat_of_evaporation_in;
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::assemble_matrixbased(
    [[maybe_unused]] const VectorType &                      advected_field_old,
    [[maybe_unused]] HeatTransferOperator::SparseMatrixType &matrix,
    [[maybe_unused]] VectorType &                            rhs) const
  {
    AssertThrow(false, ExcNotImplemented());
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    AssertThrow(this->d_tau > 0.0, ExcMessage("advection diffusion operator: d_tau must be set"));

    MeltPoolDG::VectorTools::update_ghost_values(temperature, heat_source);
    if (velocity)
      MeltPoolDG::VectorTools::update_ghost_values(*velocity);
    if (level_set_as_heaviside)
      MeltPoolDG::VectorTools::update_ghost_values(*level_set_as_heaviside);
    if (data.solidification)
      MeltPoolDG::VectorTools::update_ghost_values(temperature_old);

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

    MeltPoolDG::VectorTools::zero_out_ghost_values(temperature, heat_source);
    if (velocity)
      MeltPoolDG::VectorTools::zero_out_ghost_values(*velocity);
    if (level_set_as_heaviside)
      MeltPoolDG::VectorTools::zero_out_ghost_values(*level_set_as_heaviside);
    if (data.solidification)
      MeltPoolDG::VectorTools::zero_out_ghost_values(temperature_old);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::tangent_cell_loop(
    const MatrixFree<dim, number> &       matrix_free,
    VectorType &                          dst,
    const VectorType &                    src,
    std::pair<unsigned int, unsigned int> cell_range) const
  {
    FECellIntegrator<dim, 1, number>   temp_vals(matrix_free, temp_dof_idx, this->quad_idx);
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   temp_lin_vals(matrix_free, temp_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   temp_old_vals(matrix_free, temp_dof_idx, this->quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        temp_vals.reinit(cell);
        temp_vals.read_dof_values(src);

        tangent_local_cell_operation(
          temp_vals, temp_lin_vals, temp_old_vals, velocity_vals, ls_vals, true);

        temp_vals.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::tangent_boundary_loop(
    const MatrixFree<dim, number> &       matrix_free,
    VectorType &                          dst,
    const VectorType &                    src,
    std::pair<unsigned int, unsigned int> face_range) const
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
        dQ_dT.read_dof_values(src);

        tangent_local_boundary_operation(dQ_dT, temp_vals, true /*do reinit faces*/);

        dQ_dT.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::compute_inverse_diagonal(VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, temp_dof_idx);

    // note: not thread safe!!!
    const auto &                       matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   temp_lin_vals(matrix_free, temp_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   temp_old_vals(matrix_free, temp_dof_idx, this->quad_idx);

    unsigned int old_cell_index = numbers::invalid_unsigned_int;

    // compute diagonal ...
    MatrixFreeTools::template compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
      matrix_free,
      diagonal,
      [&](auto &temp_vals) {
        const unsigned int current_cell_index = temp_vals.get_current_cell_index();

        tangent_local_cell_operation(temp_vals,
                                     temp_lin_vals,
                                     temp_old_vals,
                                     velocity_vals,
                                     ls_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      temp_dof_idx,
      this->quad_idx);

    // ... and invert it
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::compute_system_matrix(
    TrilinosWrappers::SparseMatrix &system_matrix,
    bool                            include_boundary_terms) const
  {
    MeltPoolDG::VectorTools::update_ghost_values(temperature, heat_source);
    if (velocity)
      MeltPoolDG::VectorTools::update_ghost_values(*velocity);
    if (level_set_as_heaviside)
      MeltPoolDG::VectorTools::update_ghost_values(*level_set_as_heaviside);
    if (data.solidification)
      MeltPoolDG::VectorTools::update_ghost_values(temperature_old);
    if (!include_boundary_terms)
      {
        // note: not thread safe!!!
        const auto &                       matrix_free = scratch_data.get_matrix_free();
        FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);
        FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, this->quad_idx);
        FECellIntegrator<dim, 1, number>   temp_lin_vals(matrix_free, temp_dof_idx, this->quad_idx);
        FECellIntegrator<dim, 1, number>   temp_old_vals(matrix_free, temp_dof_idx, this->quad_idx);

        unsigned int old_cell_index = numbers::invalid_unsigned_int;

        // compute matrix (only cell contributions)
        MatrixFreeTools::template compute_matrix<dim, -1, 0, 1, number, VectorizedArray<number>>(
          matrix_free,
          scratch_data.get_constraint(temp_dof_idx),
          system_matrix,
          [&](auto &temp_vals) {
            const unsigned int current_cell_index = temp_vals.get_current_cell_index();

            tangent_local_cell_operation(temp_vals,
                                         temp_lin_vals,
                                         temp_old_vals,
                                         velocity_vals,
                                         ls_vals,
                                         old_cell_index != current_cell_index);

            old_cell_index = current_cell_index;
          },
          temp_dof_idx,
          this->quad_idx);
      }
    else
      {
        {
          const auto &                          matrix_free = scratch_data.get_matrix_free();
          std::pair<unsigned int, unsigned int> cell_range  = {0, matrix_free.n_cell_batches()};

          FECellIntegrator<dim, 1, number>   temp_vals(matrix_free, temp_dof_idx, this->quad_idx);
          FECellIntegrator<dim, dim, number> velocity_vals(matrix_free,
                                                           vel_dof_idx,
                                                           this->quad_idx);
          FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, this->quad_idx);
          FECellIntegrator<dim, 1, number> temp_lin_vals(matrix_free, temp_dof_idx, this->quad_idx);
          FECellIntegrator<dim, 1, number> temp_old_vals(matrix_free, temp_dof_idx, this->quad_idx);

          const unsigned int dofs_per_cell = temp_vals.dofs_per_cell;

          // cell integral
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              const unsigned int n_filled_lanes = matrix_free.n_active_entries_per_cell_batch(cell);

              FullMatrix<TrilinosScalar> matrices[VectorizedArray<number>::size()];
              std::fill_n(matrices,
                          VectorizedArray<number>::size(),
                          FullMatrix<TrilinosScalar>(dofs_per_cell, dofs_per_cell));

              temp_vals.reinit(cell);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    temp_vals.begin_dof_values()[i] = static_cast<number>(i == j);

                  tangent_local_cell_operation(temp_vals,
                                               temp_lin_vals,
                                               temp_old_vals,
                                               velocity_vals,
                                               ls_vals,
                                               j == 0 /*do_reinit_cell*/);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int v = 0; v < n_filled_lanes; ++v)
                      matrices[v](i, j) = temp_vals.begin_dof_values()[i][v];
                }

              for (unsigned int v = 0; v < n_filled_lanes; v++)
                {
                  auto cell_v = matrix_free.get_cell_iterator(cell, v);

                  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
                  cell_v->get_dof_indices(dof_indices);

                  auto temp = dof_indices;
                  for (unsigned int j = 0; j < dof_indices.size(); j++)
                    dof_indices[j] = temp[matrix_free.get_shape_info().lexicographic_numbering[j]];

                  scratch_data.get_constraint(temp_dof_idx)
                    .distribute_local_to_global(matrices[v], dof_indices, system_matrix);
                }
            }
        }
        // boundary_integral
        {
          const auto &                          matrix_free = scratch_data.get_matrix_free();
          std::pair<unsigned int, unsigned int> face_range  = {
            matrix_free.n_inner_face_batches(),
            matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches()};

          FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                                 true /*is_interior_face*/,
                                                 temp_dof_idx,
                                                 this->quad_idx);
          FEFaceIntegrator<dim, 1, number> temp_vals(matrix_free,
                                                     true /*is_interior_face*/,
                                                     temp_dof_idx,
                                                     this->quad_idx);

          const unsigned int dofs_per_cell = dQ_dT.dofs_per_cell;

          for (unsigned int face = face_range.first; face < face_range.second; face++)
            {
              dQ_dT.reinit(face);

              const unsigned int n_filled_lanes = matrix_free.n_active_entries_per_face_batch(face);

              FullMatrix<TrilinosScalar> matrices[VectorizedArray<number>::size()];
              std::fill_n(matrices,
                          VectorizedArray<number>::size(),
                          FullMatrix<TrilinosScalar>(dofs_per_cell, dofs_per_cell));

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    dQ_dT.begin_dof_values()[i] = static_cast<number>(i == j);

                  tangent_local_boundary_operation(dQ_dT, temp_vals, j == 0 /*do_reinit_face*/);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int v = 0; v < n_filled_lanes; ++v)
                      matrices[v](i, j) = dQ_dT.begin_dof_values()[i][v];
                }

              for (unsigned int v = 0; v < n_filled_lanes; v++)
                {
                  unsigned int const cell_number =
                    matrix_free.get_face_info(face).cells_interior[v];

                  auto cell_v =
                    matrix_free.get_cell_iterator(cell_number / VectorizedArray<number>::size(),
                                                  cell_number % VectorizedArray<number>::size());

                  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
                  cell_v->get_dof_indices(dof_indices);

                  auto temp = dof_indices;
                  for (unsigned int j = 0; j < dof_indices.size(); j++)
                    dof_indices[j] = temp[matrix_free.get_shape_info().lexicographic_numbering[j]];

                  scratch_data.get_constraint(temp_dof_idx)
                    .distribute_local_to_global(matrices[v], dof_indices, system_matrix);
                }
            }
        }
      }

    system_matrix.compress(VectorOperation::add);
    MeltPoolDG::VectorTools::zero_out_ghost_values(temperature, heat_source);
    if (velocity)
      MeltPoolDG::VectorTools::zero_out_ghost_values(*velocity);
    if (level_set_as_heaviside)
      MeltPoolDG::VectorTools::zero_out_ghost_values(*level_set_as_heaviside);
    if (data.solidification)
      MeltPoolDG::VectorTools::zero_out_ghost_values(temperature_old);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::rhs_cell_loop(
    const MatrixFree<dim, number> &       matrix_free,
    VectorType &                          dst,
    const VectorType &                    src,
    std::pair<unsigned int, unsigned int> cell_range) const
  {
    FECellIntegrator<dim, 1, number>   temp_vals(matrix_free, temp_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   temp_vals_old(matrix_free, temp_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   heat_source_vals(matrix_free, temp_dof_idx, this->quad_idx);
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, this->quad_idx);
    FECellIntegrator<dim, 1, number>   evapor_vals(matrix_free, ls_dof_idx, this->quad_idx);

    VectorizedArray<number> rho_cp       = material.first.density * material.first.capacity;
    VectorizedArray<number> conductivity = material.first.conductivity;

    q_vapor.resize(scratch_data.get_matrix_free().n_cell_batches(),
                   std::vector<VectorizedArray<double>>(temp_vals.n_q_points));

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        temp_vals.reinit(cell);
        temp_vals.read_dof_values_plain(temperature);
        temp_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        temp_vals_old.reinit(cell);
        temp_vals_old.read_dof_values_plain(src); // = temperature_old
        temp_vals_old.evaluate(EvaluationFlags::values);

        heat_source_vals.reinit(cell);
        heat_source_vals.read_dof_values_plain(heat_source);
        heat_source_vals.evaluate(EvaluationFlags::values);

        if (velocity)
          {
            velocity_vals.reinit(cell);
            velocity_vals.read_dof_values_plain(*velocity);
            velocity_vals.evaluate(EvaluationFlags::values);
          }

        if (level_set_as_heaviside)
          {
            ls_vals.reinit(cell);
            ls_vals.read_dof_values_plain(*level_set_as_heaviside);
            if (evaporative_mass_flux)
              ls_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
            else
              ls_vals.evaluate(EvaluationFlags::values);
          }

        if (evaporative_mass_flux)
          {
            evapor_vals.reinit(cell);
            evapor_vals.read_dof_values_plain(*evaporative_mass_flux);
            evapor_vals.evaluate(EvaluationFlags::values);
          }

        for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
          {
            get_material_parameters(rho_cp,
                                    conductivity,
                                    data.solidification,
                                    level_set_as_heaviside,
                                    temp_vals,
                                    ls_vals,
                                    q_index);

            auto val = this->d_tau_inv * rho_cp *
                         (temp_vals.get_value(q_index) - temp_vals_old.get_value(q_index)) -
                       heat_source_vals.get_value(q_index);

            auto val_grad = conductivity * temp_vals.get_gradient(q_index);

            if (velocity)
              {
                val += rho_cp * temp_vals.get_gradient(q_index) * velocity_vals.get_value(q_index);
              }

            if (evaporative_mass_flux)
              {
                /*
                 * compute the heat sink due to evaporation
                 *             .
                 *    q_s =  - m · h_v · δ
                 *                        Γ
                 */
                const auto temp = evapor_vals.get_value(q_index) * latent_heat_of_evaporation *
                                  ls_vals.get_gradient(q_index).norm();
                q_vapor[cell][q_index] = -temp;
                val += temp;
              }

            temp_vals.submit_value(-1.0 * val,
                                   q_index); // -1 residual is moved to rhs
            temp_vals.submit_gradient(-1.0 * val_grad,
                                      q_index); // -1 since residual is moved to rhs
          }
        temp_vals.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::rhs_boundary_loop(
    const MatrixFree<dim, number> &       matrix_free,
    VectorType &                          dst,
    [[maybe_unused]] const VectorType &   src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                           true /* is interior face*/,
                                           temp_dof_idx,
                                           this->quad_idx);
    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        dQ_dT.reinit(face);
        dQ_dT.read_dof_values_plain(temperature);
        dQ_dT.evaluate(EvaluationFlags::values);

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
              temp -= data.emissivity * PhysicalConstants::stefan_boltzmann_constant *
                      (pow<double>(temp_vals, 4) - std::pow(data.temperature_infinity, 4));
            if (do_convection)
              temp -= data.convection_coefficient * (temp_vals - data.temperature_infinity);

            dQ_dT.submit_value(temp, q_index);
          }
        dQ_dT.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    MeltPoolDG::VectorTools::update_ghost_values(temperature, heat_source);
    if (velocity)
      MeltPoolDG::VectorTools::update_ghost_values(*velocity);
    if (level_set_as_heaviside)
      MeltPoolDG::VectorTools::update_ghost_values(*level_set_as_heaviside);
    if (evaporative_mass_flux)
      MeltPoolDG::VectorTools::update_ghost_values(*evaporative_mass_flux);

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

    MeltPoolDG::VectorTools::zero_out_ghost_values(temperature, heat_source);
    if (velocity)
      MeltPoolDG::VectorTools::zero_out_ghost_values(*velocity);
    if (level_set_as_heaviside)
      MeltPoolDG::VectorTools::zero_out_ghost_values(*level_set_as_heaviside);
    if (evaporative_mass_flux)
      MeltPoolDG::VectorTools::zero_out_ghost_values(*evaporative_mass_flux);

    MeltPoolDG::VectorTools::zero_out_ghost_values(heat_source);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    if (evaporative_mass_flux)
      {
        scratch_data.initialize_dof_vector(evapor_heat_source, temp_dof_idx);
        if (!q_vapor.empty() && scratch_data.is_hex_mesh())
          UtilityFunctions::fill_dof_vector_from_cell_operation<dim, 1>(
            evapor_heat_source,
            scratch_data.get_matrix_free(),
            temp_dof_idx,
            this->quad_idx,
            [&](const unsigned int cell, const unsigned int quad)
              -> const VectorizedArray<double> & { return q_vapor[cell][quad]; });
        evapor_heat_source.update_ghost_values();

        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 evapor_heat_source,
                                 "evporative_heat_source");
      }
    else
      (void)data_out;
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number> &  temp_vals,
    FECellIntegrator<dim, 1, number> &  temp_lin_vals,
    FECellIntegrator<dim, 1, number> &  temp_old_vals,
    FECellIntegrator<dim, dim, number> &velocity_vals,
    FECellIntegrator<dim, 1, number> &  ls_vals,
    const bool                          do_reinit_cells) const
  {
    VectorizedArray<number> rho_cp            = material.first.density * material.first.capacity;
    VectorizedArray<number> conductivity      = material.first.conductivity;
    VectorizedArray<number> d_rho_cp_dT       = 0.0;
    VectorizedArray<number> d_conductivity_dT = 0.0;

    temp_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells)
      {
        if (velocity)
          {
            velocity_vals.reinit(temp_vals.get_current_cell_index());
            velocity_vals.gather_evaluate(*velocity, EvaluationFlags::values);
          }

        if (level_set_as_heaviside)
          {
            ls_vals.reinit(temp_vals.get_current_cell_index());
            if (evaporative_mass_flux)
              ls_vals.gather_evaluate(*level_set_as_heaviside,
                                      EvaluationFlags::values | EvaluationFlags::gradients);
            else
              ls_vals.gather_evaluate(*level_set_as_heaviside, EvaluationFlags::values);
          }

        if (data.solidification)
          {
            temp_old_vals.reinit(temp_vals.get_current_cell_index());
            temp_old_vals.gather_evaluate(temperature_old, EvaluationFlags::values);
          }

        if (data.solidification)
          {
            temp_lin_vals.reinit(temp_vals.get_current_cell_index());
            temp_lin_vals.gather_evaluate(temperature,
                                          EvaluationFlags::values | EvaluationFlags::gradients);
          }
      }

    for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
      {
        get_material_parameters_and_derivatives(rho_cp,
                                                conductivity,
                                                d_rho_cp_dT,
                                                d_conductivity_dT,
                                                data.solidification,
                                                level_set_as_heaviside,
                                                temp_lin_vals,
                                                ls_vals,
                                                q_index);

        auto val = this->d_tau_inv * rho_cp * temp_vals.get_value(q_index);

        auto val_grad = conductivity * temp_vals.get_gradient(q_index);

        if (velocity)
          {
            val += rho_cp * temp_vals.get_gradient(q_index) * velocity_vals.get_value(q_index);
            // todo: add term containing ∇·u  in case of evaporation
          }

        if (data.solidification)
          {
            val += this->d_tau_inv * d_rho_cp_dT *
                   (temp_lin_vals.get_value(q_index) - temp_old_vals.get_value(q_index)) *
                   temp_vals.get_value(q_index);
            val_grad += d_conductivity_dT * temp_lin_vals.get_gradient(q_index) *
                        temp_vals.get_value(q_index);
          }

        if (velocity && data.solidification)
          {
            val += d_rho_cp_dT * temp_lin_vals.get_gradient(q_index) *
                   velocity_vals.get_value(q_index) * temp_vals.get_value(q_index);
            // todo: add term containing ∇·u  in case of evaporation
          }

        temp_vals.submit_value(val, q_index);
        temp_vals.submit_gradient(val_grad, q_index);
      }
    temp_vals.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::tangent_local_boundary_operation(
    FEFaceIntegrator<dim, 1, number> &dQ_dT,
    FEFaceIntegrator<dim, 1, number> &temp_vals,
    const bool                        do_reinit_face) const
  {
    dQ_dT.evaluate(EvaluationFlags::values);

    if (do_reinit_face)
      {
        temp_vals.reinit(dQ_dT.get_current_cell_index());
        temp_vals.gather_evaluate(temperature, EvaluationFlags::values);
      }

    const types::boundary_id bc_index =
      scratch_data.get_matrix_free().get_boundary_id(dQ_dT.get_current_cell_index());

    const bool do_radiation =
      (std::find(bc_radiation_indices.begin(), bc_radiation_indices.end(), bc_index) !=
       bc_radiation_indices.end());

    const bool do_convection =
      (std::find(bc_convection_indices.begin(), bc_convection_indices.end(), bc_index) !=
       bc_convection_indices.end());

    for (unsigned int q_index = 0; q_index < dQ_dT.n_q_points; ++q_index)
      {
        auto inc_temp_vals_at_q = dQ_dT.get_value(q_index);

        VectorizedArray<double> temp = 0;

        if (do_convection)
          temp += data.convection_coefficient * inc_temp_vals_at_q;
        if (do_radiation)
          temp += 4. * data.emissivity * PhysicalConstants::stefan_boltzmann_constant *
                  pow<double>(temp_vals.get_value(q_index), 3) * inc_temp_vals_at_q;

        dQ_dT.submit_value(temp, q_index);
      }
    dQ_dT.integrate(EvaluationFlags::values);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::get_material_parameters(
    VectorizedArray<number> &               rho_cp,
    VectorizedArray<number> &               conductivity,
    const bool                              with_solidification,
    const bool                              with_two_phase,
    const FECellIntegrator<dim, 1, number> &temp_lin_val,
    const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
    const unsigned int                      q_index) const
  {
    if (!with_solidification && !with_two_phase)
      return;

    VectorizedArray<number> liq_capacity     = material.second.capacity;
    VectorizedArray<number> liq_conductivity = material.second.conductivity;
    VectorizedArray<number> liq_density      = material.second.density;

    if (with_two_phase)
      {
        get_material_parameters_with_two_phase_flow(liq_capacity,
                                                    liq_conductivity,
                                                    liq_density,
                                                    ls_heaviside_val.get_value(q_index));
        conductivity = liq_conductivity;
        rho_cp       = liq_density * liq_capacity;
      }

    if (with_solidification)
      {
        VectorizedArray<number> capacity;
        VectorizedArray<number> density;
        const auto solid_fraction = calculate_solid_fraction(temp_lin_val.get_value(q_index));
        get_material_parameters_with_solidification(capacity,
                                                    conductivity,
                                                    density,
                                                    liq_capacity,
                                                    liq_conductivity,
                                                    liq_density,
                                                    solid_fraction);
        rho_cp = density * capacity;
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::get_material_parameters_and_derivatives(
    VectorizedArray<number> &               rho_cp,
    VectorizedArray<number> &               conductivity,
    VectorizedArray<number> &               d_rho_cp_dT,
    VectorizedArray<number> &               d_conductivity_dT,
    const bool                              with_solidification,
    const bool                              with_two_phase,
    const FECellIntegrator<dim, 1, number> &temp_lin_val,
    const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
    const unsigned int                      q_index) const
  {
    if (!with_solidification && !with_two_phase)
      return;

    VectorizedArray<number> liq_capacity     = material.second.capacity;
    VectorizedArray<number> liq_density      = material.second.density;
    VectorizedArray<number> liq_conductivity = material.second.conductivity;

    if (with_two_phase)
      {
        get_material_parameters_with_two_phase_flow(liq_capacity,
                                                    liq_conductivity,
                                                    liq_density,
                                                    ls_heaviside_val.get_value(q_index));
        conductivity = liq_conductivity;
        rho_cp       = liq_density * liq_capacity;
      }

    if (with_solidification)
      {
        VectorizedArray<number> capacity;
        VectorizedArray<number> density;
        const auto solid_fraction = calculate_solid_fraction(temp_lin_val.get_value(q_index));
        get_material_parameters_with_solidification(capacity,
                                                    conductivity,
                                                    density,
                                                    liq_capacity,
                                                    liq_conductivity,
                                                    liq_density,
                                                    solid_fraction);
        rho_cp = density * capacity;
        VectorizedArray<number> d_capacity_dT;
        VectorizedArray<number> d_density_dT;
        get_material_parameter_derivatives_with_solidification(d_capacity_dT,
                                                               d_conductivity_dT,
                                                               d_density_dT,
                                                               liq_capacity,
                                                               liq_conductivity,
                                                               liq_density,
                                                               solid_fraction);
        d_rho_cp_dT = d_capacity_dT * density + d_density_dT * capacity;
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::get_material_parameters_with_solidification(
    VectorizedArray<number> &      capacity,
    VectorizedArray<number> &      conductivity,
    VectorizedArray<number> &      density,
    const VectorizedArray<number> &liq_capacity,
    const VectorizedArray<number> &liq_conductivity,
    const VectorizedArray<number> &liq_density,
    const VectorizedArray<number> &solid_fraction) const
  {
    if (solid_fraction == VectorizedArray<number>(0.0))
      {
        capacity     = liq_capacity;
        conductivity = liq_conductivity;
        density      = liq_density;
        return;
      }

    if (solid_fraction == VectorizedArray<number>(1.0))
      {
        capacity     = material.solid.capacity;
        conductivity = material.solid.conductivity;
        density      = material.solid.density;
        return;
      }

    capacity =
      UtilityFunctions::interpolate_cubic(solid_fraction, liq_capacity, material.solid.capacity);
    conductivity = UtilityFunctions::interpolate_cubic(solid_fraction,
                                                       liq_conductivity,
                                                       material.solid.conductivity);
    density =
      UtilityFunctions::interpolate_cubic(solid_fraction, liq_density, material.solid.density);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::get_material_parameter_derivatives_with_solidification(
    VectorizedArray<number> &      d_capacity_dT,
    VectorizedArray<number> &      d_conductivity_dT,
    VectorizedArray<number> &      d_density_dT,
    const VectorizedArray<number> &liq_capacity,
    const VectorizedArray<number> &liq_conductivity,
    const VectorizedArray<number> &liq_density,
    const VectorizedArray<number> &solid_fraction) const
  {
    if (solid_fraction == VectorizedArray<number>(0.0) ||
        solid_fraction == VectorizedArray<number>(1.0))
      {
        d_capacity_dT     = 0.0;
        d_conductivity_dT = 0.0;
        d_density_dT      = 0.0;
        return;
      }

    d_capacity_dT = -1.0 * material.inv_mushy_interval *
                    UtilityFunctions::interpolate_cubic_derivative(solid_fraction,
                                                                   liq_capacity,
                                                                   material.solid.capacity);
    d_conductivity_dT = -1.0 * material.inv_mushy_interval *
                        UtilityFunctions::interpolate_cubic_derivative(solid_fraction,
                                                                       liq_conductivity,
                                                                       material.solid.conductivity);
    d_density_dT = -1.0 * material.inv_mushy_interval *
                   UtilityFunctions::interpolate_cubic_derivative(solid_fraction,
                                                                  liq_density,
                                                                  material.solid.density);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::get_material_parameters_with_two_phase_flow(
    VectorizedArray<number> &      capacity,
    VectorizedArray<number> &      conductivity,
    VectorizedArray<number> &      density,
    const VectorizedArray<number> &ls_heaviside_val) const
  {
    VectorizedArray<number> weight;
    weight =
      (material.two_phase_properties_transition_type != TwoPhasePropertiesTransitionType::sharp) ?
        ls_heaviside_val :
        UtilityFunctions::heaviside(ls_heaviside_val, 0.5);

    capacity =
      UtilityFunctions::interpolate(weight, material.first.capacity, material.second.capacity);
    conductivity = UtilityFunctions::interpolate(weight,
                                                 material.first.conductivity,
                                                 material.second.conductivity);
    density =
      UtilityFunctions::interpolate(weight, material.first.density, material.second.density);
    // @todo: support density depending on evaporation
  }

  template <int dim, typename number>
  VectorizedArray<number>
  HeatTransferOperator<dim, number>::calculate_solid_fraction(
    const VectorizedArray<number> &current_temperature) const
  {
    return UtilityFunctions::limit_to_bounds((material.liquidus_temperature - current_temperature) *
                                               material.inv_mushy_interval,
                                             0.0,
                                             1.0);
  }

  template class HeatTransferOperator<1, double>;
  template class HeatTransferOperator<2, double>;
  template class HeatTransferOperator<3, double>;
} // namespace MeltPoolDG::Heat
