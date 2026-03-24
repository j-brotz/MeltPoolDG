#include <meltpooldg/utilities/vector_tools.hpp>
//
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <algorithm>
#include <limits>
#include <memory>


namespace MeltPoolDG::VectorTools
{
  template <int dim, int spacedim, typename number>
  void
  convert_fe_system_vector_to_block_vector(
    const dealii::LinearAlgebra::distributed::Vector<number> &in,
    const dealii::DoFHandler<dim, spacedim>                  &dof_handler_fe_system,
    dealii::LinearAlgebra::distributed::BlockVector<number>  &out,
    const dealii::DoFHandler<dim, spacedim>                  &dof_handler)
  {
    const bool update_ghosts = not in.has_ghost_elements();
    if (update_ghosts)
      in.update_ghost_values();

    for (const auto &cell_fe_system : dof_handler_fe_system.active_cell_iterators())
      {
        if (not cell_fe_system->is_locally_owned())
          continue;

        dealii::Vector<number> local(dof_handler_fe_system.get_fe().n_dofs_per_cell());
        cell_fe_system->get_dof_values(in, local);


        auto cell = dealii::DoFCellAccessor<dim, dim, false>(&dof_handler.get_triangulation(),
                                                             cell_fe_system->level(),
                                                             cell_fe_system->index(),
                                                             &dof_handler);

        for (unsigned int d = 0; d < dim; ++d)
          {
            const unsigned int     n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
            dealii::Vector<number> local_component(n_dofs_per_component);

            for (unsigned int c = 0; c < n_dofs_per_component; ++c)
              local_component[c] =
                local[dof_handler_fe_system.get_fe().component_to_system_index(d, c)];

            cell.set_dof_values(local_component, out.block(d));
          }
      }

    if (update_ghosts)
      in.zero_out_ghost_values();
  }

  template <int dim, int n_components, typename number>
  void
  project_function_to_grid_points(dealii::LinearAlgebra::distributed::Vector<number> &out,
                                  const dealii::Function<dim>                        &function,
                                  const dealii::MatrixFree<dim, number>              &matrix_free,
                                  const unsigned int                                  dof_idx,
                                  const unsigned int                                  quad_idx)
  {
    FECellIntegrator<dim, n_components, number> phi(matrix_free, dof_idx, quad_idx);
    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
      phi);
    out.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
          phi.submit_dof_value(evaluate_function_at_vectorized_points<dim, number, n_components>(
                                 function, phi.quadrature_point(q)),
                               q);
        inverse.transform_from_q_points_to_basis(n_components,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(out);
      }
  }

  template <int dim, int spacedim, typename number>
  void
  convert_block_vector_to_fe_system_vector(
    const dealii::LinearAlgebra::distributed::BlockVector<number> &in,
    const dealii::DoFHandler<dim, spacedim>                       &dof_handler,
    dealii::LinearAlgebra::distributed::Vector<number>            &out,
    const dealii::DoFHandler<dim, spacedim>                       &dof_handler_fe_system)
  {
    const bool update_ghosts = not in.has_ghost_elements();
    if (update_ghosts)
      in.update_ghost_values();

    for (const auto &cell_fe_system : dof_handler_fe_system.active_cell_iterators())
      {
        if (not cell_fe_system->is_locally_owned())
          continue;

        auto cell =
          dealii::DoFCellAccessor<dim, dim, false>(&dof_handler_fe_system.get_triangulation(),
                                                   cell_fe_system->level(),
                                                   cell_fe_system->index(),
                                                   &dof_handler);

        dealii::Vector<number> local(dof_handler_fe_system.get_fe().n_dofs_per_cell());

        for (unsigned int d = 0; d < dim; ++d)
          {
            const unsigned int     n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
            dealii::Vector<number> local_component(n_dofs_per_component);

            cell.get_dof_values(in.block(d), local_component);

            for (unsigned int c = 0; c < n_dofs_per_component; ++c)
              local[dof_handler_fe_system.get_fe().component_to_system_index(d, c)] =
                local_component[c];
          }
        cell_fe_system->set_dof_values(local, out);
      }

    if (update_ghosts)
      in.zero_out_ghost_values();
  }

  template <int dim, typename number, typename VectorType>
  number
  compute_global_error_norm(const VectorType                    &solution,
                            const dealii::Triangulation<dim>    &triangulation,
                            const dealii::Mapping<dim>          &mapping,
                            const dealii::DoFHandler<dim>       &dof_handler,
                            const dealii::Quadrature<dim>       &quadrature,
                            const dealii::VectorTools::NormType  norm_type,
                            const dealii::Function<dim, number> &reference_solution,
                            const dealii::Function<dim, number> *weight)
  {
    const bool is_ghosted = solution.has_ghost_elements();

    if (not is_ghosted)
      solution.update_ghost_values();

    dealii::Vector<float> difference_per_cell(triangulation.n_active_cells());

    dealii::VectorTools::integrate_difference(mapping,
                                              dof_handler,
                                              solution,
                                              reference_solution,
                                              difference_per_cell,
                                              quadrature,
                                              norm_type,
                                              weight);

    if (not is_ghosted)
      solution.zero_out_ghost_values();

    return dealii::VectorTools::compute_global_error(triangulation, difference_per_cell, norm_type);
  }

  template <int dim, typename number, typename VectorType>
  number
  compute_norm(const VectorType                   &solution,
               const dealii::Triangulation<dim>   &triangulation,
               const dealii::Mapping<dim>         &mapping,
               const dealii::DoFHandler<dim>      &dof_handler,
               const dealii::Quadrature<dim>      &quadrature,
               const dealii::VectorTools::NormType norm_type)
  {
    return compute_global_error_norm(solution,
                                     triangulation,
                                     mapping,
                                     dof_handler,
                                     quadrature,
                                     norm_type,
                                     dealii::Functions::ZeroFunction<dim>(
                                       dof_handler.get_fe().n_components()));
  }

  template <int dim, typename number, typename VectorType>
  number
  compute_norm(const VectorType                    &solution,
               const ScratchData<dim, dim, number> &scratch_data,
               const unsigned int                   dof_idx,
               const unsigned int                   quad_idx,
               const dealii::VectorTools::NormType  norm_type)
  {
    return compute_norm<dim, number, VectorType>(solution,
                                                 scratch_data.get_triangulation(dof_idx),
                                                 scratch_data.get_mapping(),
                                                 scratch_data.get_dof_handler(dof_idx),
                                                 scratch_data.get_quadrature(quad_idx),
                                                 norm_type);
  }

  template <int n_components, int dim, typename VectorType>
  void
  project_vector(const dealii::Mapping<dim>                                       &mapping,
                 const dealii::DoFHandler<dim>                                    &dof,
                 const dealii::AffineConstraints<typename VectorType::value_type> &constraints,
                 const dealii::Quadrature<dim>                                    &quadrature,
                 const VectorType                                                 &vec_in,
                 VectorType                                                       &vec_out)
  {
    using number = typename VectorType::value_type;

    typename dealii::MatrixFree<dim, number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, number>::AdditionalData::partition_color;
    additional_data.mapping_update_flags = dealii::update_values | dealii::update_JxW_values;

    const auto matrix_free = std::make_shared<dealii::MatrixFree<dim, number>>();
    matrix_free->reinit(mapping, dof, constraints, quadrature, additional_data);

    using MatrixType =
      dealii::MatrixFreeOperators::MassOperator<dim, -1, 0, n_components, VectorType>;
    MatrixType mass_matrix;
    mass_matrix.initialize(matrix_free);

    mass_matrix.compute_diagonal();

    dealii::ReductionControl                  control(6 * vec_in.size(), 0., 1e-12, false, false);
    dealii::SolverCG<VectorType>              cg(control);
    const dealii::DiagonalMatrix<VectorType> &preconditioner =
      *mass_matrix.get_matrix_diagonal_inverse();

    cg.solve(mass_matrix, vec_out, vec_in, preconditioner);
  }

  template <typename number>
  number
  max_element(const dealii::LinearAlgebra::distributed::Vector<number> &vec,
              const MPI_Comm                                           &mpi_comm)
  {
    number max = std::numeric_limits<number>::lowest();
    for (unsigned int i = 0; i < vec.locally_owned_size(); ++i)
      max = std::max(max, vec.local_element(i));

    return dealii::Utilities::MPI::max(max, mpi_comm);
  }

  template <typename number>
  number
  min_element(const dealii::LinearAlgebra::distributed::Vector<number> &vec,
              const MPI_Comm                                           &mpi_comm)
  {
    number min = std::numeric_limits<number>::max();
    for (unsigned int i = 0; i < vec.locally_owned_size(); ++i)
      min = std::min(min, vec.local_element(i));

    return dealii::Utilities::MPI::min(min, mpi_comm);
  }

  // Explicit instantiations

  template void
  convert_fe_system_vector_to_block_vector(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::DoFHandler<1, 1> &,
    dealii::LinearAlgebra::distributed::BlockVector<double> &,
    const dealii::DoFHandler<1, 1> &);
  template void
  convert_fe_system_vector_to_block_vector(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::DoFHandler<2, 2> &,
    dealii::LinearAlgebra::distributed::BlockVector<double> &,
    const dealii::DoFHandler<2, 2> &);
  template void
  convert_fe_system_vector_to_block_vector(
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::DoFHandler<3, 3> &,
    dealii::LinearAlgebra::distributed::BlockVector<double> &,
    const dealii::DoFHandler<3, 3> &);

  template void
  project_function_to_grid_points<2, 2, double>(
    dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::Function<2> &,
    const dealii::MatrixFree<2, double> &,
    const unsigned int,
    const unsigned int);

  template void
  convert_block_vector_to_fe_system_vector(
    const dealii::LinearAlgebra::distributed::BlockVector<double> &,
    const dealii::DoFHandler<1, 1> &,
    dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::DoFHandler<1, 1> &);
  template void
  convert_block_vector_to_fe_system_vector(
    const dealii::LinearAlgebra::distributed::BlockVector<double> &,
    const dealii::DoFHandler<2, 2> &,
    dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::DoFHandler<2, 2> &);
  template void
  convert_block_vector_to_fe_system_vector(
    const dealii::LinearAlgebra::distributed::BlockVector<double> &,
    const dealii::DoFHandler<3, 3> &,
    dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::DoFHandler<3, 3> &);

  template double
  compute_global_error_norm(const dealii::LinearAlgebra::distributed::Vector<double> &solution,
                            const dealii::Triangulation<1>                           &triangulation,
                            const dealii::Mapping<1>                                 &mapping,
                            const dealii::DoFHandler<1>                              &dof_handler,
                            const dealii::Quadrature<1>                              &quadrature,
                            const dealii::VectorTools::NormType                       norm_type,
                            const dealii::Function<1, double> &reference_solution,
                            const dealii::Function<1, double> *weight);

  template double
  compute_global_error_norm(const dealii::LinearAlgebra::distributed::Vector<double> &solution,
                            const dealii::Triangulation<2>                           &triangulation,
                            const dealii::Mapping<2>                                 &mapping,
                            const dealii::DoFHandler<2>                              &dof_handler,
                            const dealii::Quadrature<2>                              &quadrature,
                            const dealii::VectorTools::NormType                       norm_type,
                            const dealii::Function<2, double> &reference_solution,
                            const dealii::Function<2, double> *weight);

  template double
  compute_global_error_norm(const dealii::LinearAlgebra::distributed::Vector<double> &solution,
                            const dealii::Triangulation<3>                           &triangulation,
                            const dealii::Mapping<3>                                 &mapping,
                            const dealii::DoFHandler<3>                              &dof_handler,
                            const dealii::Quadrature<3>                              &quadrature,
                            const dealii::VectorTools::NormType                       norm_type,
                            const dealii::Function<3, double> &reference_solution,
                            const dealii::Function<3, double> *weight);

  template double
  compute_norm(const dealii::LinearAlgebra::distributed::Vector<double> &,
               const dealii::Triangulation<1> &,
               const dealii::Mapping<1> &,
               const dealii::DoFHandler<1> &,
               const dealii::Quadrature<1> &,
               const dealii::VectorTools::NormType);
  template double
  compute_norm(const dealii::LinearAlgebra::distributed::Vector<double> &,
               const dealii::Triangulation<2> &,
               const dealii::Mapping<2> &,
               const dealii::DoFHandler<2> &,
               const dealii::Quadrature<2> &,
               const dealii::VectorTools::NormType);
  template double
  compute_norm(const dealii::LinearAlgebra::distributed::Vector<double> &,
               const dealii::Triangulation<3> &,
               const dealii::Mapping<3> &,
               const dealii::DoFHandler<3> &,
               const dealii::Quadrature<3> &,
               const dealii::VectorTools::NormType);

  template double
  compute_norm(const dealii::LinearAlgebra::distributed::Vector<double> &,
               const ScratchData<1, 1, double> &,
               const unsigned int,
               const unsigned int,
               const dealii::VectorTools::NormType);
  template double
  compute_norm(const dealii::LinearAlgebra::distributed::Vector<double> &,
               const ScratchData<2, 2, double> &,
               const unsigned int,
               const unsigned int,
               const dealii::VectorTools::NormType);
  template double
  compute_norm(const dealii::LinearAlgebra::distributed::Vector<double> &,
               const ScratchData<3, 3, double> &,
               const unsigned int,
               const unsigned int,
               const dealii::VectorTools::NormType);

  template void
  project_vector<1, 1, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<1> &,
    const dealii::DoFHandler<1> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<1, 2, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<2> &,
    const dealii::DoFHandler<2> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<1, 3, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<3> &,
    const dealii::DoFHandler<3> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<2, 1, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<1> &,
    const dealii::DoFHandler<1> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<2, 2, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<2> &,
    const dealii::DoFHandler<2> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<2, 3, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<3> &,
    const dealii::DoFHandler<3> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<3, 1, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<1> &,
    const dealii::DoFHandler<1> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<3, 2, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<2> &,
    const dealii::DoFHandler<2> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);
  template void
  project_vector<3, 3, dealii::LinearAlgebra::distributed::Vector<double>>(
    const dealii::Mapping<3> &,
    const dealii::DoFHandler<3> &,
    const dealii::AffineConstraints<
      typename dealii::LinearAlgebra::distributed::Vector<double>::value_type> &,
    const dealii::Quadrature<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    dealii::LinearAlgebra::distributed::Vector<double> &);

  template double
  max_element(const dealii::LinearAlgebra::distributed::Vector<double> &, const MPI_Comm &);

  template double
  min_element(const dealii::LinearAlgebra::distributed::Vector<double> &, const MPI_Comm &);
} // namespace MeltPoolDG::VectorTools