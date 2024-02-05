#pragma once
#include <deal.II/base/mpi.h>

#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/scratch_data.hpp>

namespace dealii
{
  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator+=(const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec,
             const dealii::VectorizedArray<Number, N>                       &scalar)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator+=(const dealii::VectorizedArray<Number, N>                       &scalar,
             const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator-=(const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec,
             const dealii::VectorizedArray<Number, N>                       &scalar)
  {
    auto temp = vec;
    temp[0] -= scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator-=(const dealii::VectorizedArray<Number, N>                       &scalar,
             const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec)
  {
    auto temp = -vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::VectorizedArray<Number, N>
  scalar_product(const dealii::VectorizedArray<Number, N>                       &scalar,
                 const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec)
  {
    return vec[0] * scalar;
  }

  template <typename Number, std::size_t N>
  dealii::VectorizedArray<Number, N>
  scalar_product(const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec,
                 const dealii::VectorizedArray<Number, N>                       &scalar)
  {
    return vec[0] * scalar;
  }
} // namespace dealii


namespace MeltPoolDG
{
  using namespace dealii;

  namespace VectorTools
  {
    template <int dim, typename number = double>
    VectorizedArray<number>
    compute_mask_narrow_band(const VectorizedArray<number> &val, const double narrow_band_threshold)
    {
      VectorizedArray<number> indicator = 1.0;
      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        if (std::abs(val[v]) >= narrow_band_threshold)
          indicator[v] = 0.0;

      return indicator;
    }

    template <int dim, int spacedim, typename Number>
    void
    convert_fe_system_vector_to_block_vector(const LinearAlgebra::distributed::Vector<Number> &in,
                                             const DoFHandler<dim, spacedim> &dof_handler_fe_system,
                                             LinearAlgebra::distributed::BlockVector<Number> &out,
                                             const DoFHandler<dim, spacedim> &dof_handler)
    {
      const bool update_ghosts = !in.has_ghost_elements();
      if (update_ghosts)
        in.update_ghost_values();

      for (const auto &cell_fe_system : dof_handler_fe_system.active_cell_iterators())
        if (cell_fe_system->is_locally_owned())
          {
            Vector<double> local(dof_handler_fe_system.get_fe().n_dofs_per_cell());
            cell_fe_system->get_dof_values(in, local);


            auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler.get_triangulation(),
                                                         cell_fe_system->level(),
                                                         cell_fe_system->index(),
                                                         &dof_handler);

            for (unsigned int d = 0; d < dim; ++d)
              {
                const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                Vector<double>     local_component(n_dofs_per_component);

                for (unsigned int c = 0; c < n_dofs_per_component; ++c)
                  local_component[c] =
                    local[dof_handler_fe_system.get_fe().component_to_system_index(d, c)];

                cell.set_dof_values(local_component, out.block(d));
              }
          }

      if (update_ghosts)
        in.zero_out_ghost_values();
    }

    template <int dim, int spacedim, typename Number>
    void
    convert_block_vector_to_fe_system_vector(
      const LinearAlgebra::distributed::BlockVector<Number> &in,
      const DoFHandler<dim, spacedim>                       &dof_handler,
      LinearAlgebra::distributed::Vector<Number>            &out,
      const DoFHandler<dim, spacedim>                       &dof_handler_fe_system)
    {
      const bool update_ghosts = !in.has_ghost_elements();
      if (update_ghosts)
        in.update_ghost_values();

      for (const auto &cell_fe_system : dof_handler_fe_system.active_cell_iterators())
        if (cell_fe_system->is_locally_owned())
          {
            auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler_fe_system.get_triangulation(),
                                                         cell_fe_system->level(),
                                                         cell_fe_system->index(),
                                                         &dof_handler);

            Vector<double> local(dof_handler_fe_system.get_fe().n_dofs_per_cell());

            for (unsigned int d = 0; d < dim; ++d)
              {
                const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                Vector<double>     local_component(n_dofs_per_component);

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

    template <typename... T>
    void
    update_ghost_values(const T &...args)
    {
      ((args.update_ghost_values()), ...);
    }

    template <typename... T>
    void
    zero_out_ghost_values(const T &...args)
    {
      ((args.zero_out_ghost_values()), ...);
    }

    template <int dim, typename number>
    Tensor<1, dim, number>
    to_vector(const Tensor<1, dim, number> &in)
    {
      return in;
    }

    template <int dim, typename number>
    Tensor<1, dim, number>
    to_vector(const number &in)
    {
      Tensor<1, dim, number> vec;

      vec[0] = in;

      return vec;
    }


    template <int dim, typename number>
    Tensor<1, dim, VectorizedArray<number>>
    normalize(const VectorizedArray<number> &in, const double zero = 1e-16)
    {
      Tensor<1, dim, VectorizedArray<number>> vec;

      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        vec[0][v] = in[v] >= zero ? 1.0 : -1.0;

      return vec;
    }

    template <int dim, typename number>
    Tensor<1, dim, VectorizedArray<number>>
    normalize(const Tensor<1, dim, VectorizedArray<number>> &in, const double zero = 1e-16)
    {
      Tensor<1, dim, VectorizedArray<number>> vec;

      const auto n_norm = in.norm();

      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        if (n_norm[v] > zero)
          for (unsigned int d = 0; d < dim; ++d)
            vec[d][v] = in[d][v] / n_norm[v];
        else
          for (unsigned int d = 0; d < dim; ++d)
            vec[d][v] = 0.0;

      return vec;
    }

    template <int dim, typename number = double>
    Tensor<1, dim, VectorizedArray<number>>
    evaluate_function_at_vectorized_points(const Function<dim>                       &func,
                                           const Point<dim, VectorizedArray<double>> &points)
    {
      AssertThrow(func.n_components == dim, ExcNotImplemented());

      Tensor<1, dim, VectorizedArray<number>> vec;

      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        {
          Point<dim> point_v;

          for (unsigned int d = 0; d < dim; ++d)
            point_v[d] = points[d][v];

          for (unsigned int d = 0; d < dim; ++d)
            vec[d][v] = func.value(point_v, d);
        }
      return vec;
    }

    template <int dim, typename VectorType>
    double
    compute_norm(const VectorType                   &solution,
                 const Triangulation<dim>           &triangulation,
                 const Mapping<dim>                 &mapping,
                 const DoFHandler<dim>              &dof_handler,
                 const Quadrature<dim>              &quadrature,
                 const dealii::VectorTools::NormType norm_type)
    {
      const bool is_ghosted = solution.has_ghost_elements();

      if (!is_ghosted)
        solution.update_ghost_values();

      Vector<float> difference_per_cell(triangulation.n_active_cells());

      dealii::VectorTools::integrate_difference(mapping,
                                                dof_handler,
                                                solution,
                                                Functions::ZeroFunction<dim>(
                                                  dof_handler.get_fe().n_components()),
                                                difference_per_cell,
                                                quadrature,
                                                norm_type);

      if (!is_ghosted)
        solution.zero_out_ghost_values();

      return dealii::VectorTools::compute_global_error(triangulation,
                                                       difference_per_cell,
                                                       norm_type);
    }

    template <int dim, typename VectorType>
    double
    compute_norm(const VectorType                   &solution,
                 const ScratchData<dim>             &scratch_data,
                 const unsigned int                  dof_idx,
                 const unsigned int                  quad_idx,
                 const dealii::VectorTools::NormType norm_type = dealii::VectorTools::L2_norm)
    {
      return compute_norm<dim, VectorType>(solution,
                                           scratch_data.get_triangulation(dof_idx),
                                           scratch_data.get_mapping(),
                                           scratch_data.get_dof_handler(dof_idx),
                                           scratch_data.get_quadrature(quad_idx),
                                           norm_type);
    }

    template <int n_components, int dim, typename VectorType>
    void
    project_vector(const Mapping<dim>                                       &mapping,
                   const DoFHandler<dim>                                    &dof,
                   const AffineConstraints<typename VectorType::value_type> &constraints,
                   const Quadrature<dim>                                    &quadrature,
                   const VectorType                                         &vec_in,
                   VectorType                                               &vec_out)
    {
      using Number = typename VectorType::value_type;

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::partition_color;
      additional_data.mapping_update_flags = (update_values | update_JxW_values);

      const auto matrix_free = std::make_shared<MatrixFree<dim, Number>>();
      matrix_free->reinit(mapping, dof, constraints, quadrature, additional_data);

      using MatrixType = MatrixFreeOperators::MassOperator<dim, -1, 0, n_components, VectorType>;
      MatrixType mass_matrix;
      mass_matrix.initialize(matrix_free);

      mass_matrix.compute_diagonal();

      ReductionControl                  control(6 * vec_in.size(), 0., 1e-12, false, false);
      SolverCG<VectorType>              cg(control);
      const DiagonalMatrix<VectorType> &preconditioner = *mass_matrix.get_matrix_diagonal_inverse();

      cg.solve(mass_matrix, vec_out, vec_in, preconditioner);
    }

    /**
     * For a given @p matrix_free object, execute scalar- or vector-valued @p cell_operation
     * on each quadrature point  defined by @p quad_idx and fill them into a
     * DoF-vector @p vec defined by @p dof_idx.
     */
    template <int dim, int n_components, typename T, typename VectorType>
    void
    fill_dof_vector_from_cell_operation(
      VectorType                                             &vec,
      const MatrixFree<dim, double, VectorizedArray<double>> &matrix_free,
      unsigned int                                            dof_idx,
      unsigned int                                            quad_idx,
      const T                                                &cell_operation)
    {
      FECellIntegrator<dim, n_components, double> fe_eval(matrix_free, dof_idx, quad_idx);

      MatrixFreeOperators::
        CellwiseInverseMassMatrix<dim, -1, n_components, double, VectorizedArray<double>>
          inverse_mass_matrix(fe_eval);

      const auto &lexicographic_numbering =
        matrix_free.get_shape_info(dof_idx, quad_idx).lexicographic_numbering;

      VectorType weights;
      weights.reinit(vec);
      std::vector<double> ones(fe_eval.dofs_per_cell, 1.0);

      std::vector<types::global_dof_index> dof_indices(fe_eval.dofs_per_cell);
      std::vector<types::global_dof_index> dof_indices_mf(fe_eval.dofs_per_cell);
      std::vector<double>                  dof_values(fe_eval.dofs_per_cell);

      AffineConstraints<double> dummy;

      vec = 0.0;

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          fe_eval.reinit(cell);

          for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
            {
              const auto temp = cell_operation(cell, q);
              for (int c = 0; c < n_components; ++c)
                if constexpr (std::is_same<typename std::remove_const<decltype(temp)>::type,
                                           VectorizedArray<double>>::value)
                  {
                    static_assert(n_components == 1,
                                  "The path should be only accessed for a single component.");
                    fe_eval.begin_values()[q] = temp;
                  }
                else if constexpr (std::is_same<
                                     typename std::remove_const<decltype(temp)>::type,
                                     Tensor<1, n_components, VectorizedArray<double>>>::value)
                  {
                    fe_eval.begin_values()[c * fe_eval.n_q_points + q] = temp[c];
                  }
                else
                  {
                    Assert(false, ExcNotImplemented());
                  }
            }
          inverse_mass_matrix.transform_from_q_points_to_basis(n_components,
                                                               fe_eval.begin_values(),
                                                               fe_eval.begin_dof_values());

          for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
            {
              matrix_free.get_cell_iterator(cell, v, dof_idx)->get_dof_indices(dof_indices);

              for (unsigned int j = 0; j < dof_indices.size(); ++j)
                {
                  dof_indices_mf[j] = dof_indices[lexicographic_numbering[j]];
                  dof_values[j]     = fe_eval.begin_dof_values()[j][v];
                }

              dummy.distribute_local_to_global(dof_values, dof_indices_mf, vec);
              dummy.distribute_local_to_global(ones, dof_indices_mf, weights);
            }
        }

      vec.compress(VectorOperation::add);
      weights.compress(VectorOperation::add);

      for (unsigned int i = 0; i < vec.locally_owned_size(); ++i)
        if (weights.local_element(i) != 0.0)
          vec.local_element(i) /= weights.local_element(i);
    }

    /**
     * Calculate the overall maximum element of a distributed vector @p vec.
     */
    template <typename number>
    number
    max_element(const LinearAlgebra::distributed::Vector<number> &vec, const MPI_Comm &mpi_comm)
    {
      number max = std::numeric_limits<number>::lowest();
      for (unsigned int i = 0; i < vec.locally_owned_size(); ++i)
        max = std::max(max, vec.local_element(i));

      return dealii::Utilities::MPI::max(max, mpi_comm);
    }

    /**
     * Calculate the overall minimum element of a distributed vector @p vec.
     */
    template <typename number>
    number
    min_element(const LinearAlgebra::distributed::Vector<number> &vec, const MPI_Comm &mpi_comm)
    {
      number min = std::numeric_limits<number>::max();
      for (unsigned int i = 0; i < vec.locally_owned_size(); ++i)
        min = std::min(min, vec.local_element(i));

      return dealii::Utilities::MPI::min(min, mpi_comm);
    }

  } // namespace VectorTools
} // namespace MeltPoolDG
