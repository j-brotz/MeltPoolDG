#pragma once
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/scratch_data.hpp>

namespace dealii
{
  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator+=(const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec,
             const dealii::VectorizedArray<Number, N> &                      scalar)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator+=(const dealii::VectorizedArray<Number, N> &                      scalar,
             const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec)
  {
    auto temp = vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator-=(const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec,
             const dealii::VectorizedArray<Number, N> &                      scalar)
  {
    auto temp = vec;
    temp[0] -= scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>>
  operator-=(const dealii::VectorizedArray<Number, N> &                      scalar,
             const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec)
  {
    auto temp = -vec;
    temp[0] += scalar;

    return temp;
  }

  template <typename Number, std::size_t N>
  dealii::VectorizedArray<Number, N>
  scalar_product(const dealii::VectorizedArray<Number, N> &                      scalar,
                 const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec)
  {
    return vec[0] * scalar;
  }

  template <typename Number, std::size_t N>
  dealii::VectorizedArray<Number, N>
  scalar_product(const dealii::Tensor<1, 1, dealii::VectorizedArray<Number, N>> &vec,
                 const dealii::VectorizedArray<Number, N> &                      scalar)
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
    void
    filter_narrow_band(const VectorizedArray<number> &                  val,
                       Tensor<1, dim, dealii::VectorizedArray<number>> &grad,
                       const double                                     narrow_band_threshold)
    {
      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        {
          if (std::abs(val[v]) >= narrow_band_threshold)
            {
              for (unsigned int d = 0; d < dim; ++d)
                grad[d][v] = 0.0;
            }
        }
    }

    template <int dim, typename number = double>
    void
    filter_narrow_band(const VectorizedArray<number> &                  val,
                       Tensor<2, dim, dealii::VectorizedArray<number>> &grad,
                       const double                                     narrow_band_threshold)
    {
      for (unsigned int d = 0; d < dim; ++d)
        filter_narrow_band<dim, number>(val, grad[d], narrow_band_threshold);
    }

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

    template <int dim, typename number = double>
    void
    filter_narrow_band(const VectorizedArray<number> &val1,
                       VectorizedArray<number> &      val,
                       const double                   narrow_band_threshold)
    {
      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        if (std::abs(val1[v]) >= narrow_band_threshold)
          val[v] = 0.0;
    }

    template <int dim, int spacedim, typename Number>
    void
    convert_fe_system_vector_to_block_vector(const LinearAlgebra::distributed::Vector<Number> &in,
                                             const DoFHandler<dim, spacedim> &dof_handler_fe_system,
                                             LinearAlgebra::distributed::BlockVector<Number> &out,
                                             const DoFHandler<dim, spacedim> &dof_handler)
    {
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

      in.zero_out_ghost_values();
    }

    template <int dim, int spacedim, typename Number>
    void
    convert_block_vector_to_fe_system_vector(
      const LinearAlgebra::distributed::BlockVector<Number> &in,
      const DoFHandler<dim, spacedim> &                      dof_handler,
      LinearAlgebra::distributed::Vector<Number> &           out,
      const DoFHandler<dim, spacedim> &                      dof_handler_fe_system)
    {
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
    static Tensor<1, dim, VectorizedArray<number>>
    normalize(const VectorizedArray<number> &in, const double zero = 1e-16)
    {
      Tensor<1, dim, VectorizedArray<number>> vec;

      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        vec[0][v] = in[v] >= zero ? 1.0 : -1.0;

      return vec;
    }

    template <int dim, typename number>
    static Tensor<1, dim, VectorizedArray<number>>
    normalize(const Tensor<1, dim, VectorizedArray<number>> &in, const double zero = 1e-16)
    {
      Tensor<1, dim, VectorizedArray<number>> vec;

      auto n_norm = in.norm();
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
    static Tensor<1, dim, VectorizedArray<number>>
    evaluate_function_at_vectorized_points(const Function<dim> &                      func,
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
    compute_L2_norm(const VectorType &        solution,
                    const Triangulation<dim> &triangulation,
                    const Mapping<dim> &      mapping,
                    const DoFHandler<dim> &   dof_handler,
                    const Quadrature<dim> &   quadrature)
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
                                                dealii::VectorTools::L2_norm);

      if (!is_ghosted)
        solution.zero_out_ghost_values();

      return dealii::VectorTools::compute_global_error(triangulation,
                                                       difference_per_cell,
                                                       dealii::VectorTools::L2_norm);
    }

    template <int dim, typename VectorType>
    double
    compute_L2_norm(const VectorType &      solution,
                    const ScratchData<dim> &scratch_data,
                    const unsigned int      dof_idx,
                    const unsigned int      quad_idx)
    {
      return compute_L2_norm<dim, VectorType>(solution,
                                              scratch_data.get_triangulation(dof_idx),
                                              scratch_data.get_mapping(),
                                              scratch_data.get_dof_handler(dof_idx),
                                              scratch_data.get_quadrature(quad_idx));
    }
  } // namespace VectorTools
} // namespace MeltPoolDG
