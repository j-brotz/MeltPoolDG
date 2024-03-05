#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/heat/heat_transfer_operator.hpp>
#include <meltpooldg/interface/exceptions.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/material/material.templates.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>

namespace MeltPoolDG::Heat
{
  template <int dim, typename number>
  HeatTransferOperator<dim, number>::HeatTransferOperator(
    const std::shared_ptr<BoundaryConditions<dim>> &bc,
    const ScratchData<dim>                         &scratch_data_in,
    const HeatData<number>                         &data_in,
    const Material<number>                         &material,
    const unsigned int                              temp_dof_idx_in,
    const unsigned int                              temp_quad_idx_in,
    const unsigned int                              temp_hanging_nodes_dof_idx_in,
    const VectorType                               &temperature_in,
    const VectorType                               &temperature_old_in,
    const VectorType                               &heat_source_in,
    const unsigned int                              vel_dof_idx_in,
    const VectorType                               *velocity_in,
    const unsigned int                              ls_dof_idx_in,
    const VectorType                               *level_set_as_heaviside_in,
    const bool                                      do_solidification_in)
    : scratch_data(scratch_data_in)
    , data(data_in)
    , material(material)
    , temp_dof_idx(temp_dof_idx_in)
    , temp_quad_idx(temp_quad_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , temperature(temperature_in)
    , temperature_old(temperature_old_in)
    , heat_source(heat_source_in)
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , ls_dof_idx(ls_dof_idx_in)
    , level_set_as_heaviside(level_set_as_heaviside_in)
    , do_solidification(do_solidification_in)
    , do_update_ghosts(6)
  {
    // TODO move these asserts to a more central place -> to material perhaps?
    const auto &material_data = material.get_data();
    // TODO move solidification bool to material?
    AssertThrow((material_data.gas.thermal_conductivity > 0.0 && material_data.gas.density > 0.0) ||
                  do_solidification,
                ExcMessage(
                  "The material's conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !level_set_as_heaviside ||
        (material_data.liquid.thermal_conductivity > 0.0 && material_data.liquid.density > 0.0),
      ExcMessage(
        "The secondary material's conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !do_solidification ||
        (material_data.solid.thermal_conductivity > 0.0 && material_data.solid.density > 0.0),
      ExcMessage(
        "In case of solidification the solid's conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !do_solidification ||
        (material_data.liquid.thermal_conductivity > 0.0 && material_data.liquid.density > 0.0),
      ExcMessage(
        "In case of solidification the liquid's (material_data.liquid) conductivity and density must be greater than zero! Abort..."));
    AssertThrow(
      !do_solidification || material_data.liquidus_temperature > material_data.solidus_temperature,
      ExcMessage(
        "In case of solidification the liquidus temperature must be greater than the solidus temperature! Abort..."));

    if (bc)
      {
        bc_convection_indices = bc->convection_bc;
        bc_radiation_indices  = bc->radiation_bc;
        neumann_bc            = bc->neumann_bc;
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::register_surface_mesh(
    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<double> /*weights*/
                                 >> &surface_mesh_info_in)
  {
    surface_mesh_info = &surface_mesh_info_in;
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::register_evaporative_mass_flux(
    VectorType        *evaporative_mass_flux_in,
    const unsigned int evapor_mass_flux_dof_idx_in,
    const double       latent_heat_of_evaporation_in,
    const typename Evaporation::EvaporationData<number>::EvaporativeCooling &evapor_cooling_data)
  {
    delta_phase_weighted = create_phase_weighted_delta_approximation(
      evapor_cooling_data.delta_approximation_phase_weighted);

    evapor_flux_type = evapor_cooling_data.model;
    AssertThrow(surface_mesh_info ||
                  (evapor_flux_type != Evaporation::EvaporCoolingInterfaceFluxType::sharp),
                ExcMessage("If you would like to use a sharp flux model, you first"
                           "need to register the surface mesh."));

    do_level_set_temperature_gradient_interpolation = scratch_data.is_FE_Q_iso_Q_1(ls_dof_idx);

    if (do_level_set_temperature_gradient_interpolation)
      ls_to_temp_grad_interpolation_matrix = UtilityFunctions::create_dof_interpolation_matrix<dim>(
        scratch_data.get_dof_handler(temp_dof_idx),
        scratch_data.get_dof_handler(ls_dof_idx),
        true /* do_matrix_free */);

    evaporative_mass_flux      = evaporative_mass_flux_in;
    evapor_mass_flux_dof_idx   = evapor_mass_flux_dof_idx_in;
    latent_heat_of_evaporation = latent_heat_of_evaporation_in;
    do_phenomenological_recoil_pressure =
      evapor_cooling_data.consider_enthalpy_transport_vapor_mass_flux == "true";

    if (do_phenomenological_recoil_pressure)
      {
        const auto &material_data = material.get_data();
        AssertThrow(!numbers::is_invalid(material_data.specific_enthalpy_reference_temperature),
                    ExcMessage(
                      "For the phenomenological recoil pressure model, the reference temperature "
                      "for computing the specific enthalpy must be specified. Abort..."));

        AssertThrow(!do_solidification || (material_data.solid.specific_heat_capacity ==
                                           material_data.liquid.specific_heat_capacity),
                    ExcMessage("The equation for specific enthalpy for evaporative heat sink "
                               "assumes equality between the solid and liquid "
                               "phase heat capacity! Abort..."));
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::assemble_matrixbased(
    [[maybe_unused]] const VectorType                       &advected_field_old,
    [[maybe_unused]] HeatTransferOperator::SparseMatrixType &matrix,
    [[maybe_unused]] VectorType                             &rhs) const
  {
    AssertThrow(false, ExcNotImplemented());
  }
  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::update_ghost_values() const
  {
    unsigned int i = 0;

    if (do_update_ghosts[i++] = !temperature.has_ghost_elements())
      MeltPoolDG::VectorTools::update_ghost_values(temperature);
    if (do_update_ghosts[i++] = !heat_source.has_ghost_elements())
      MeltPoolDG::VectorTools::update_ghost_values(heat_source);
    if (velocity && (do_update_ghosts[i++] = !velocity->has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(*velocity);
    if (level_set_as_heaviside &&
        (do_update_ghosts[i++] = !level_set_as_heaviside->has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(*level_set_as_heaviside);
    if (do_solidification && (do_update_ghosts[i++] = !temperature_old.has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(temperature_old);
    if (evaporative_mass_flux &&
        (do_update_ghosts[i++] = !evaporative_mass_flux->has_ghost_elements()))
      MeltPoolDG::VectorTools::update_ghost_values(*evaporative_mass_flux);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::zero_out_ghost_values() const
  {
    unsigned int i = 0;

    if (do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(temperature);
    if (do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(heat_source);
    if (velocity && do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(*velocity);
    if (level_set_as_heaviside && do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(*level_set_as_heaviside);
    if (do_solidification && do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(temperature_old);
    if (evaporative_mass_flux && do_update_ghosts[i++])
      MeltPoolDG::VectorTools::zero_out_ghost_values(*evaporative_mass_flux);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        tangent_cell_loop(matrix_free, dst, src, cell_range);
      }, // cell loop
      [&]([[maybe_unused]] const auto &matrix_free,
          [[maybe_unused]] auto       &dst,
          [[maybe_unused]] const auto &src,
          [[maybe_unused]] auto        face_range) { /*do nothing*/ }, // internal face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        tangent_boundary_loop(matrix_free, dst, src, face_range); // boundary face loop
      },
      dst,
      src,
      true /*zero dst vector*/);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::tangent_cell_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> cell_range) const
  {
    FECellIntegrator<dim, 1, number>   temp_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> ls_interpolated_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> temp_lin_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> temp_old_vals(matrix_free,
                                                   temp_hanging_nodes_dof_idx,
                                                   temp_quad_idx);

    std::unique_ptr<FECellIntegrator<dim, 1, number>> evapor_vals;
    if (evaporative_mass_flux &&
        evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
      {
        evapor_vals = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                         evapor_mass_flux_dof_idx,
                                                                         temp_quad_idx);
      }

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        temp_vals.reinit(cell);
        temp_vals.read_dof_values(src);

        tangent_local_cell_operation(temp_vals,
                                     temp_lin_vals,
                                     temp_old_vals,
                                     velocity_vals,
                                     ls_vals,
                                     ls_interpolated_vals,
                                     evapor_vals,
                                     true);

        temp_vals.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::tangent_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                           true /*is_interior_face*/,
                                           temp_dof_idx,
                                           temp_quad_idx);
    FEFaceIntegrator<dim, 1, number> temp_vals(matrix_free,
                                               true /*is_interior_face*/,
                                               temp_dof_idx,
                                               temp_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        dQ_dT.reinit(face);
        dQ_dT.read_dof_values(src);

        tangent_local_boundary_operation(dQ_dT, temp_vals, true /*do reinit faces*/);

        dQ_dT.distribute_local_to_global(dst);
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::compute_inverse_diagonal_from_matrixfree(
    VectorType &diagonal) const
  {
    scratch_data.initialize_dof_vector(diagonal, temp_dof_idx);

    // note: not thread safe!!!
    const auto                        &matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> ls_interpolated_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> temp_lin_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> temp_old_vals(matrix_free,
                                                   temp_hanging_nodes_dof_idx,
                                                   temp_quad_idx);
    std::unique_ptr<FECellIntegrator<dim, 1, number>> evapor_vals;
    if (evaporative_mass_flux &&
        evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
      {
        evapor_vals = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                         evapor_mass_flux_dof_idx,
                                                                         temp_quad_idx);
      }

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
                                     ls_interpolated_vals,
                                     evapor_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      temp_dof_idx,
      temp_quad_idx);

    // ... and invert it
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::compute_system_matrix_from_matrixfree_reduced(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    // note: not thread safe!!!
    const auto                        &matrix_free = scratch_data.get_matrix_free();
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> ls_interpolated_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> temp_lin_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> temp_old_vals(matrix_free,
                                                   temp_hanging_nodes_dof_idx,
                                                   temp_quad_idx);
    std::unique_ptr<FECellIntegrator<dim, 1, number>> evapor_vals;
    if (evaporative_mass_flux &&
        evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
      {
        evapor_vals = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                         evapor_mass_flux_dof_idx,
                                                                         temp_quad_idx);
      }

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
                                     ls_interpolated_vals,
                                     evapor_vals,
                                     old_cell_index != current_cell_index);

        old_cell_index = current_cell_index;
      },
      temp_dof_idx,
      temp_quad_idx);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::compute_system_matrix_from_matrixfree(
    TrilinosWrappers::SparseMatrix &system_matrix) const
  {
    system_matrix = 0.0;

    {
      const auto                           &matrix_free = scratch_data.get_matrix_free();
      std::pair<unsigned int, unsigned int> cell_range  = {0, matrix_free.n_cell_batches()};

      FECellIntegrator<dim, 1, number>   temp_vals(matrix_free, temp_dof_idx, temp_quad_idx);
      FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, temp_quad_idx);
      FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, temp_quad_idx);
      FECellIntegrator<dim, 1, number>   ls_interpolated_vals(matrix_free,
                                                            temp_dof_idx,
                                                            temp_quad_idx);
      FECellIntegrator<dim, 1, number>   temp_lin_vals(matrix_free, temp_dof_idx, temp_quad_idx);
      FECellIntegrator<dim, 1, number>   temp_old_vals(matrix_free,
                                                     temp_hanging_nodes_dof_idx,
                                                     temp_quad_idx);
      std::unique_ptr<FECellIntegrator<dim, 1, number>> evapor_vals;
      if (evaporative_mass_flux &&
          evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
        {
          evapor_vals = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                           evapor_mass_flux_dof_idx,
                                                                           temp_quad_idx);
        }

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
                                           ls_interpolated_vals,
                                           evapor_vals,
                                           j == 0 /*do_reinit_cell*/);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int v = 0; v < n_filled_lanes; ++v)
                  matrices[v](i, j) = temp_vals.begin_dof_values()[i][v];
            }

          for (unsigned int v = 0; v < n_filled_lanes; ++v)
            {
              auto cell_v = matrix_free.get_cell_iterator(cell, v, temp_dof_idx);

              std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
              cell_v->get_dof_indices(dof_indices);

              auto temp = dof_indices;
              for (unsigned int j = 0; j < dof_indices.size(); ++j)
                dof_indices[j] = temp[temp_vals.get_shape_info().lexicographic_numbering[j]];

              scratch_data.get_constraint(temp_dof_idx)
                .distribute_local_to_global(matrices[v], dof_indices, system_matrix);
            }
        }
    }
    // boundary_integral
    {
      const auto                           &matrix_free = scratch_data.get_matrix_free();
      std::pair<unsigned int, unsigned int> face_range  = {matrix_free.n_inner_face_batches(),
                                                           matrix_free.n_inner_face_batches() +
                                                             matrix_free.n_boundary_face_batches()};

      FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                             true /*is_interior_face*/,
                                             temp_dof_idx,
                                             temp_quad_idx);
      FEFaceIntegrator<dim, 1, number> temp_vals(matrix_free,
                                                 true /*is_interior_face*/,
                                                 temp_dof_idx,
                                                 temp_quad_idx);

      const unsigned int dofs_per_cell = dQ_dT.dofs_per_cell;

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
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

          for (unsigned int v = 0; v < n_filled_lanes; ++v)
            {
              unsigned int const cell_number = matrix_free.get_face_info(face).cells_interior[v];

              auto cell_v =
                matrix_free.get_cell_iterator(cell_number / VectorizedArray<number>::size(),
                                              cell_number % VectorizedArray<number>::size(),
                                              temp_dof_idx);

              std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
              cell_v->get_dof_indices(dof_indices);

              auto temp = dof_indices;
              for (unsigned int j = 0; j < dof_indices.size(); ++j)
                dof_indices[j] = temp[dQ_dT.get_shape_info().lexicographic_numbering[j]];

              scratch_data.get_constraint(temp_dof_idx)
                .distribute_local_to_global(matrices[v], dof_indices, system_matrix);
            }
        }
    }

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::rhs_cell_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    const VectorType                     &src,
    std::pair<unsigned int, unsigned int> cell_range) const
  {
    FECellIntegrator<dim, 1, number>   temp_vals(matrix_free, temp_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number>   temp_vals_old(matrix_free,
                                                   temp_hanging_nodes_dof_idx,
                                                   temp_quad_idx);
    FECellIntegrator<dim, 1, number>   heat_source_vals(matrix_free,
                                                      temp_hanging_nodes_dof_idx,
                                                      temp_quad_idx);
    FECellIntegrator<dim, dim, number> velocity_vals(matrix_free, vel_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number>   ls_vals(matrix_free, ls_dof_idx, temp_quad_idx);
    FECellIntegrator<dim, 1, number> ls_interpolated_vals(matrix_free, temp_dof_idx, temp_quad_idx);

    auto &ls_vals_used =
      do_level_set_temperature_gradient_interpolation ? ls_interpolated_vals : ls_vals;

    std::unique_ptr<FECellIntegrator<dim, 1, number>> evapor_vals;

    if (evaporative_mass_flux &&
        evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::regularized)
      {
        evapor_vals = std::make_unique<FECellIntegrator<dim, 1, number>>(matrix_free,
                                                                         evapor_mass_flux_dof_idx,
                                                                         temp_quad_idx);
      }

    //@todo: only in case of certain verbosity level or if variable is requested
    q_vapor.resize_fast(scratch_data.get_matrix_free().n_cell_batches() * temp_vals.n_q_points);
    conductivity_at_q.resize_fast(scratch_data.get_matrix_free().n_cell_batches() *
                                  temp_vals.n_q_points);
    rho_cp_at_q.resize_fast(scratch_data.get_matrix_free().n_cell_batches() * temp_vals.n_q_points);

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
              {
                if (do_level_set_temperature_gradient_interpolation)
                  {
                    ls_interpolated_vals.reinit(cell);

                    UtilityFunctions::compute_gradient_at_interpolated_dof_values<dim>(
                      ls_vals, ls_interpolated_vals, ls_to_temp_grad_interpolation_matrix);
                  }

                ls_vals_used.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
              }
            else
              ls_vals.evaluate(EvaluationFlags::values);
          }

        if (evapor_vals)
          {
            evapor_vals->reinit(cell);
            evapor_vals->read_dof_values_plain(*evaporative_mass_flux);
            evapor_vals->evaluate(EvaluationFlags::values);
          }


        for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
          {
            const auto [rho_cp, conductivity] =
              get_material_parameters(temp_vals, ls_vals_used, q_index);

            conductivity_at_q[cell * temp_vals.n_q_points + q_index] = conductivity;
            rho_cp_at_q[cell * temp_vals.n_q_points + q_index]       = rho_cp;

            auto val = this->time_increment_inv * rho_cp *
                         (temp_vals.get_value(q_index) - temp_vals_old.get_value(q_index)) -
                       heat_source_vals.get_value(q_index);

            auto val_grad = conductivity * temp_vals.get_gradient(q_index);

            if (velocity)
              {
                val += rho_cp * scalar_product(temp_vals.get_gradient(q_index),
                                               velocity_vals.get_value(q_index));
              }

            if (evapor_vals)
              {
                /*
                 * Compute the heat sink due to evaporation
                 *             .
                 *    q_s =  - m · ( h_v + h(T)) · δ
                 *                                  Γ
                 *
                 * with the latent heat of evaporation h_v, the specific enthalpy
                 *
                 *          T
                 *         /
                 *        |
                 * h(T) = | c_p  .  (1)
                 *        |
                 *       /
                 *     T_ref
                 *
                 * where T_ref denotes an artificial reference temperature for
                 * the specific enthalpy.
                 *
                 * h_v + h(T) is the total enthalpy leaving the system with the
                 * evaporative mass flux. h(T) can be typically found in tables.
                 * However, we assume for the computation of the specific
                 * enthalpy a linear temperature-dependence
                 *
                 *    h(T) = h_ref + c_p * T .  (2)
                 *
                 * see also
                 * Meier, Christoph, et al. "A novel smoothed particle hydrodynamics
                 * formulation for thermo-capillary phase change problems with focus
                 * on metal additive manufacturing melt pool modeling." CMAME 381
                 * (2021).
                 *
                 * By setting (1) equal to (2) and assuming the heat capacity to
                 * be temperature-independent, we obtain
                 *
                 *    h_ref = - c_p * T_ref
                 *
                 * Inserting into (2) yields
                 *
                 *    h(T) = c_p * (T - T_ref).
                 *
                 *
                 * @note For the computation of h(T), it is assumed that the
                 *       specific heat capacity c_p corresponds to the value
                 *       for the liquid and solid phase.
                 *
                 * @note Instead of T_ref we could have also introduced directly
                 *       h_ref as an input parameter.
                 */
                VectorizedArray<number> specific_enthalpy(0.0);

                if (do_phenomenological_recoil_pressure)
                  {
                    const auto &material_data = material.get_data();
                    specific_enthalpy         = material_data.liquid.specific_heat_capacity *
                                        (temp_vals.get_value(q_index) -
                                         material_data.specific_enthalpy_reference_temperature);
                  }

                VectorizedArray<double> weight(1.0);

                if (delta_phase_weighted)
                  weight = delta_phase_weighted->compute_weight(ls_vals_used.get_value(q_index));

                const auto temp = evapor_vals->get_value(q_index) *
                                  (latent_heat_of_evaporation + specific_enthalpy) *
                                  ls_vals_used.get_gradient(q_index).norm() * weight;

                q_vapor[cell * temp_vals.n_q_points + q_index] = -temp;
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
  HeatTransferOperator<dim, number>::rhs_cut_cell_loop(VectorType &dst) const
  {
    // evaluate the evaporative heat loss term as surface integral
    if (evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp)
      {
        Assert(evaporative_mass_flux,
               ExcMessage("You need to register the evaporative mass flux."));

        evapor_heat_source = 0;

        FEPointEvaluation<1, dim> evapor_vals_surf(scratch_data.get_mapping(),
                                                   scratch_data.get_fe(evapor_mass_flux_dof_idx),
                                                   update_values);

        FEPointEvaluation<1, dim> temp_vals_surf(scratch_data.get_mapping(),
                                                 scratch_data.get_fe(temp_dof_idx),
                                                 update_values);

        std::vector<double>                  buffer;
        std::vector<types::global_dof_index> local_dof_indices;

        const int n_dofs_evapor = scratch_data.get_fe(evapor_mass_flux_dof_idx).n_dofs_per_cell();
        const int n_dofs_temp   = scratch_data.get_fe(temp_dof_idx).n_dofs_per_cell();

        if (surface_mesh_info->size() > 0)
          {
            for (const auto &[cell, quad_points, weights] : *(surface_mesh_info))
              {
                const unsigned int n_points = quad_points.size();

                const ArrayView<const Point<dim>> unit_points(quad_points.data(), n_points);
                const ArrayView<const double>     JxW(weights.data(), n_points);

                evapor_vals_surf.reinit(cell, unit_points);
                temp_vals_surf.reinit(cell, unit_points);

                const auto temp_dof_cell =
                  TriaIterator<DoFCellAccessor<dim, dim, false>>(&scratch_data.get_triangulation(),
                                                                 cell->level(),
                                                                 cell->index(),
                                                                 &scratch_data.get_dof_handler(
                                                                   temp_dof_idx));

                const auto evapor_dof_cell =
                  TriaIterator<DoFCellAccessor<dim, dim, false>>(&scratch_data.get_triangulation(),
                                                                 cell->level(),
                                                                 cell->index(),
                                                                 &scratch_data.get_dof_handler(
                                                                   evapor_mass_flux_dof_idx));

                // gather evaluate mass_flux
                local_dof_indices.resize(n_dofs_evapor);
                buffer.resize(n_dofs_evapor);
                evapor_dof_cell->get_dof_indices(local_dof_indices);
                scratch_data.get_constraint(evapor_mass_flux_dof_idx)
                  .get_dof_values(*evaporative_mass_flux,
                                  local_dof_indices.begin(),
                                  buffer.begin(),
                                  buffer.end());
                evapor_vals_surf.evaluate(buffer, EvaluationFlags::values);

                // gather evaluate temperature
                buffer.resize(n_dofs_temp);
                local_dof_indices.resize(n_dofs_temp);
                temp_dof_cell->get_dof_indices(local_dof_indices);
                scratch_data.get_constraint(temp_dof_idx)
                  .get_dof_values(temperature,
                                  local_dof_indices.begin(),
                                  buffer.begin(),
                                  buffer.end());
                temp_vals_surf.evaluate(buffer, EvaluationFlags::values);

                for (unsigned int q = 0; q < n_points; ++q)
                  {
                    double specific_enthalpy = 0;

                    if (do_phenomenological_recoil_pressure)
                      {
                        const auto &material_data = material.get_data();
                        specific_enthalpy         = material_data.liquid.specific_heat_capacity *
                                            (temp_vals_surf.get_value(q) -
                                             material_data.specific_enthalpy_reference_temperature);
                      }
                    temp_vals_surf.submit_value(-evapor_vals_surf.get_value(q) *
                                                  (latent_heat_of_evaporation + specific_enthalpy) *
                                                  JxW[q],
                                                q); // *(-1) for the residual
                  }
                temp_vals_surf.test_and_sum(buffer,
                                            EvaluationFlags::values); //

                scratch_data.get_constraint(temp_dof_idx)
                  .distribute_local_to_global(buffer, local_dof_indices, evapor_heat_source);
              }
          }

        evapor_heat_source.compress(VectorOperation::add);
        scratch_data.get_constraint(temp_dof_idx).set_zero(evapor_heat_source);

        dst += evapor_heat_source;
      }
    else if (evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming)
      {
        Assert(evaporative_mass_flux,
               ExcMessage("You need to register the evaporative mass flux."));
        evapor_heat_source = 0;
        // step 1: set material ID of cells
        LevelSet::Tools::set_material_id_from_level_set(scratch_data,
                                                        ls_dof_idx,
                                                        *level_set_as_heaviside);

        // step 2: evaluate and fill rhs
        FEFaceIntegrator<dim, 1, double> temp_eval(scratch_data.get_matrix_free(),
                                                   true /*is_interior_face*/,
                                                   temp_dof_idx,
                                                   temp_quad_idx);
        FEFaceIntegrator<dim, 1, double> evapor_eval(scratch_data.get_matrix_free(),
                                                     true /*is_interior_face*/,
                                                     evapor_mass_flux_dof_idx,
                                                     temp_quad_idx);

        std::pair<unsigned int, unsigned int> face_range = {
          0, scratch_data.get_matrix_free().n_inner_face_batches()};

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            temp_eval.reinit(face);
            if (do_phenomenological_recoil_pressure)
              temp_eval.gather_evaluate(temperature, EvaluationFlags::values);

            evapor_eval.reinit(face);
            evapor_eval.gather_evaluate(*evaporative_mass_flux, EvaluationFlags::values);

            // collect lanes that need to be processed
            std::bitset<VectorizedArray<number>::size()> active_lanes;
            for (unsigned int v = 0;
                 v < scratch_data.get_matrix_free().n_active_entries_per_face_batch(face);
                 ++v)
              {
                const auto face_iter_inner =
                  scratch_data.get_matrix_free().get_face_iterator(face, v, true);
                const auto face_iter_outer =
                  scratch_data.get_matrix_free().get_face_iterator(face, v, false);

                // check if surrounding cells have different materials
                active_lanes[v] = static_cast<bool>(face_iter_inner.first->material_id() !=
                                                    face_iter_outer.first->material_id());
              }

            // loop over quadrature points
            for (unsigned int q = 0; q < temp_eval.n_q_points; ++q)
              {
                VectorizedArray<double> specific_enthalpy = 0;

                if (do_phenomenological_recoil_pressure)
                  {
                    const auto &material_data = material.get_data();
                    specific_enthalpy         = material_data.liquid.specific_heat_capacity *
                                        (temp_eval.get_value(q) -
                                         material_data.specific_enthalpy_reference_temperature);
                  }
                temp_eval.submit_value(-evapor_eval.get_value(q) *
                                         (latent_heat_of_evaporation + specific_enthalpy),
                                       q); // *(-1) for the residual
              }
            temp_eval.integrate(EvaluationFlags::values);
            temp_eval.distribute_local_to_global(evapor_heat_source, 0, active_lanes);
          }
        evapor_heat_source.compress(VectorOperation::add);
        scratch_data.get_constraint(temp_dof_idx).set_zero(evapor_heat_source);
        dst += evapor_heat_source;
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::rhs_boundary_loop(
    const MatrixFree<dim, number>        &matrix_free,
    VectorType                           &dst,
    [[maybe_unused]] const VectorType    &src,
    std::pair<unsigned int, unsigned int> face_range) const
  {
    FEFaceIntegrator<dim, 1, number> dQ_dT(matrix_free,
                                           true /* is interior face*/,
                                           temp_dof_idx,
                                           temp_quad_idx);
    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
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

        dQ_dT.reinit(face);
        dQ_dT.read_dof_values_plain(temperature);
        dQ_dT.evaluate(EvaluationFlags::values);

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
              temp -= data.radiation.emissivity * PhysicalConstants::stefan_boltzmann_constant *
                      (Utilities::fixed_power<4>(temp_vals) -
                       Utilities::fixed_power<4>(data.radiation.temperature_infinity));
            if (do_convection)
              temp -= data.convection.convection_coefficient *
                      (temp_vals - data.convection.temperature_infinity);

            dQ_dT.submit_value(temp, q_index);
          }
        dQ_dT.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::create_rhs(VectorType &dst, const VectorType &src) const
  {
    AssertThrowZeroTimeIncrement(this->time_increment);

    scratch_data.get_matrix_free().template loop<VectorType, VectorType>(
      [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
        rhs_cell_loop(matrix_free, dst, src, cell_range);
      },
      [&]([[maybe_unused]] const auto &matrix_free,
          [[maybe_unused]] auto       &dst,
          [[maybe_unused]] const auto &src,
          [[maybe_unused]] auto        face_range) { /*do nothing*/ }, // face loop
      [&](const auto &matrix_free, auto &dst, const auto &src, auto face_range) {
        rhs_boundary_loop(matrix_free, dst, src, face_range);
      },
      dst,
      src,
      false /*zero dst vector*/); // should not be zeroed out in case of boundary conditions
                                  //
    rhs_cut_cell_loop(dst);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> & /*vectors*/)
  {
    // none
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::distribute_constraints()
  {
    // none
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::reinit()
  {
    // TODO: only if output variable is requested
    if (evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp ||
        evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming)
      scratch_data.initialize_dof_vector(evapor_heat_source, temp_dof_idx);
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    /**
     * write conductivity vector to dof vector
     */
    if (data_out.is_requested("conductivity"))
      {
        scratch_data.initialize_dof_vector(conductivity_vec, temp_hanging_nodes_dof_idx);

        if (!q_vapor.empty() && scratch_data.is_hex_mesh())
          MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
            conductivity_vec,
            scratch_data.get_matrix_free(),
            temp_hanging_nodes_dof_idx,
            temp_quad_idx,
            [&](const unsigned int cell,
                const unsigned int quad) -> const VectorizedArray<double> & {
              return conductivity_at_q[cell * scratch_data.get_n_q_points(temp_quad_idx) + quad];
            });

        scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(conductivity_vec);

        data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                                 conductivity_vec,
                                 "conductivity");
      }

    if (data_out.is_requested("rho_cp"))
      {
        scratch_data.initialize_dof_vector(rho_cp_vec, temp_hanging_nodes_dof_idx);

        if (!q_vapor.empty() && scratch_data.is_hex_mesh())
          MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
            rho_cp_vec,
            scratch_data.get_matrix_free(),
            temp_hanging_nodes_dof_idx,
            temp_quad_idx,
            [&](const unsigned int cell,
                const unsigned int quad) -> const VectorizedArray<double> & {
              return rho_cp_at_q[cell * scratch_data.get_n_q_points(temp_quad_idx) + quad];
            });

        scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(rho_cp_vec);

        data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                                 rho_cp_vec,
                                 "rho_cp");
      }

    /**
     * write evaporative mass flux to dof vector
     */
    if (data_out.is_requested("evaporative_heat_source_projected"))
      {
        if (evaporative_mass_flux)
          {
            scratch_data.initialize_dof_vector(evapor_heat_source_projected, temp_dof_idx);
            if (evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp ||
                evapor_flux_type == Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming)
              {
                VectorTools::project_vector<1, dim>(scratch_data.get_mapping(),
                                                    scratch_data.get_dof_handler(temp_dof_idx),
                                                    scratch_data.get_constraint(temp_dof_idx),
                                                    scratch_data.get_quadrature(temp_quad_idx),
                                                    evapor_heat_source,
                                                    evapor_heat_source_projected);
                data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                         evapor_heat_source,
                                         "evaporative_heat_source");
              }
            else
              {
                if (!q_vapor.empty() && scratch_data.is_hex_mesh())
                  MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
                    evapor_heat_source_projected,
                    scratch_data.get_matrix_free(),
                    temp_hanging_nodes_dof_idx,
                    temp_quad_idx,
                    [&](const unsigned int cell,
                        const unsigned int quad) -> const VectorizedArray<double> & {
                      return q_vapor[cell * scratch_data.get_n_q_points(temp_quad_idx) + quad];
                    });
              }

            Journal::print_formatted_norm(
              scratch_data.get_pcout(2),
              [&]() -> double {
                return VectorTools::compute_norm<dim>(evapor_heat_source_projected,
                                                      scratch_data,
                                                      temp_hanging_nodes_dof_idx,
                                                      temp_quad_idx);
              },
              "int(mDot*hV)",
              "heat_transfer_operator",
              15 /*precision*/
            );

            scratch_data.get_constraint(temp_hanging_nodes_dof_idx)
              .distribute(evapor_heat_source_projected);

            data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                                     evapor_heat_source_projected,
                                     "evaporative_heat_source_projected");
          }
      }
  }

  template <int dim, typename number>
  void
  HeatTransferOperator<dim, number>::tangent_local_cell_operation(
    FECellIntegrator<dim, 1, number>                        &temp_vals,
    FECellIntegrator<dim, 1, number>                        &temp_lin_vals,
    FECellIntegrator<dim, 1, number>                        &temp_old_vals,
    FECellIntegrator<dim, dim, number>                      &velocity_vals,
    FECellIntegrator<dim, 1, number>                        &ls_vals,
    FECellIntegrator<dim, 1, number>                        &ls_interpolated_vals,
    const std::unique_ptr<FECellIntegrator<dim, 1, number>> &evapor_vals,
    const bool                                               do_reinit_cells) const
  {
    auto &ls_vals_used =
      do_level_set_temperature_gradient_interpolation ? ls_interpolated_vals : ls_vals;

    temp_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    if (do_reinit_cells)
      {
        if (velocity)
          {
            velocity_vals.reinit(temp_vals.get_current_cell_index());
            velocity_vals.read_dof_values_plain(*velocity);
            velocity_vals.evaluate(EvaluationFlags::values);
          }

        if (level_set_as_heaviside)
          {
            ls_vals.reinit(temp_vals.get_current_cell_index());
            ls_vals.read_dof_values_plain(*level_set_as_heaviside);

            if (evaporative_mass_flux)
              {
                if (do_level_set_temperature_gradient_interpolation)
                  {
                    ls_interpolated_vals.reinit(temp_vals.get_current_cell_index());

                    UtilityFunctions::compute_gradient_at_interpolated_dof_values<dim>(
                      ls_vals, ls_interpolated_vals, ls_to_temp_grad_interpolation_matrix);

                    ls_interpolated_vals.evaluate(EvaluationFlags::values |
                                                  EvaluationFlags::gradients);
                  }
                else
                  ls_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
              }
            else
              ls_vals.evaluate(EvaluationFlags::values);
          }

        if (do_solidification)
          {
            temp_old_vals.reinit(temp_vals.get_current_cell_index());
            temp_old_vals.read_dof_values_plain(temperature_old);
            temp_old_vals.evaluate(EvaluationFlags::values);

            temp_lin_vals.reinit(temp_vals.get_current_cell_index());
            temp_lin_vals.read_dof_values_plain(temperature);
            temp_lin_vals.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
          }
        if (evapor_vals)
          {
            evapor_vals->reinit(temp_vals.get_current_cell_index());
            evapor_vals->read_dof_values_plain(*evaporative_mass_flux);
            evapor_vals->evaluate(EvaluationFlags::values);
          }
      }

    for (unsigned int q_index = 0; q_index < temp_vals.n_q_points; ++q_index)
      {
        const auto [rho_cp, conductivity, d_rho_cp_d_T, d_conductivity_d_T] =
          get_material_parameters_and_derivatives(temp_lin_vals, ls_vals_used, q_index);

        auto val = this->time_increment_inv * rho_cp * temp_vals.get_value(q_index);

        auto val_grad = conductivity * temp_vals.get_gradient(q_index);

        if (velocity)
          {
            val += rho_cp * scalar_product(temp_vals.get_gradient(q_index),
                                           velocity_vals.get_value(q_index));
            // todo: add term containing ∇·u  in case of evaporation
          }

        if (do_solidification)
          {
            val += this->time_increment_inv * d_rho_cp_d_T *
                   (temp_lin_vals.get_value(q_index) - temp_old_vals.get_value(q_index)) *
                   temp_vals.get_value(q_index);
            val_grad += d_conductivity_d_T * temp_lin_vals.get_gradient(q_index) *
                        temp_vals.get_value(q_index);
          }

        if (velocity && do_solidification)
          {
            val += d_rho_cp_d_T * temp_lin_vals.get_gradient(q_index) *
                   velocity_vals.get_value(q_index) * temp_vals.get_value(q_index);
            // todo: add term containing ∇·u  in case of evaporation
          }

        if (evapor_vals && do_phenomenological_recoil_pressure)
          {
            /*
             * derivative of specific enthalpy h(T) with respect to the temperature:
             *
             *  d h(T)
             * -------- = c_p^sl
             *    dT
             *
             * tangent of heat sink due to evaporation:
             *
             *  d q_s      .
             * ------- = - m * c_p^sl * δ
             *    dT                      Γ
             *
             * For the details regarding h(t)/c_p, see the documentation in
             * rhs_cell_loop().
             */

            VectorizedArray<double> weight(1.0);
            if (delta_phase_weighted)
              weight = delta_phase_weighted->compute_weight(ls_vals_used.get_value(q_index));

            const auto &material_data = material.get_data();

            val += evapor_vals->get_value(q_index) * temp_vals.get_value(q_index) *
                   material_data.liquid.specific_heat_capacity *
                   ls_vals_used.get_gradient(q_index).norm() * weight;
          }

        temp_vals.submit_value(val, q_index);
        temp_vals.submit_gradient(val_grad, q_index);
      }
    temp_vals.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

    // evaluate the evaporative heat loss term as surface integral
    if (evaporative_mass_flux && surface_mesh_info && do_phenomenological_recoil_pressure)
      {
        // TODO
      }
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
          temp += data.convection.convection_coefficient * inc_temp_vals_at_q;
        if (do_radiation)
          temp += 4. * data.radiation.emissivity * PhysicalConstants::stefan_boltzmann_constant *
                  pow<double>(temp_vals.get_value(q_index), 3) * inc_temp_vals_at_q;

        dQ_dT.submit_value(temp, q_index);
      }
    dQ_dT.integrate(EvaluationFlags::values);
  }



  template <int dim, typename number>
  std::tuple<VectorizedArray<number>, VectorizedArray<number>>
  HeatTransferOperator<dim, number>::get_material_parameters(
    const FECellIntegrator<dim, 1, number> &temp_lin_val,
    const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
    const unsigned int                      q_index) const
  {
    if (!data.use_volume_specific_thermal_capacity_for_phase_interpolation)
      {
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          ls_heaviside_val,
          temp_lin_val,
          MaterialUpdateFlags::thermal_conductivity | MaterialUpdateFlags::specific_heat_capacity |
            MaterialUpdateFlags::density,
          q_index);
        return {material_values.specific_heat_capacity * material_values.density,
                material_values.thermal_conductivity};
      }
    // special interpolation of volumetric heat capacity
    else
      {
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          ls_heaviside_val,
          temp_lin_val,
          MaterialUpdateFlags::thermal_conductivity |
            MaterialUpdateFlags::volume_specific_heat_capacity,
          q_index);
        return {material_values.volume_specific_heat_capacity,
                material_values.thermal_conductivity};
      }
  }



  template <int dim, typename number>
  std::tuple<VectorizedArray<number>,
             VectorizedArray<number>,
             VectorizedArray<number>,
             VectorizedArray<number>>
  HeatTransferOperator<dim, number>::get_material_parameters_and_derivatives(
    const FECellIntegrator<dim, 1, number> &temp_lin_val,
    const FECellIntegrator<dim, 1, number> &ls_heaviside_val,
    const unsigned int                      q_index) const
  {
    if (!data.use_volume_specific_thermal_capacity_for_phase_interpolation)
      {
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          ls_heaviside_val,
          temp_lin_val,
          MaterialUpdateFlags::thermal_conductivity | MaterialUpdateFlags::specific_heat_capacity |
            MaterialUpdateFlags::density | MaterialUpdateFlags::d_thermal_conductivity_d_T |
            MaterialUpdateFlags::d_specific_heat_capacity_d_T | MaterialUpdateFlags::d_density_d_T,
          q_index);

        return {/* rho cp */ material_values.density * material_values.specific_heat_capacity,
                /* conductivity */ material_values.thermal_conductivity,
                /* d_rho_cp_d_T */ material_values.d_specific_heat_capacity_d_T *
                    material_values.density +
                  material_values.d_density_d_T * material_values.specific_heat_capacity,
                /* d_conductivity_d_T */ material_values.d_thermal_conductivity_d_T};
      }
    else
      {
        // special interpolation of volumetric heat capacity
        const auto material_values = material.template compute_parameters<VectorizedArray<number>>(
          ls_heaviside_val,
          temp_lin_val,
          MaterialUpdateFlags::thermal_conductivity |
            MaterialUpdateFlags::volume_specific_heat_capacity |
            MaterialUpdateFlags::d_thermal_conductivity_d_T |
            MaterialUpdateFlags::d_volume_specific_heat_capacity_d_T,
          q_index);
        return {/* rho cp */ material_values.volume_specific_heat_capacity,
                /* conductivity */ material_values.thermal_conductivity,
                /* d_rho_cp_d_T */ material_values.d_volume_specific_heat_capacity_d_T,
                /* d_conductivity_d_T */ material_values.d_thermal_conductivity_d_T};
      }
  }



  template class HeatTransferOperator<1, double>;
  template class HeatTransferOperator<2, double>;
  template class HeatTransferOperator<3, double>;
} // namespace MeltPoolDG::Heat
