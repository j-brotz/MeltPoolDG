#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/tools.hpp>

#include <boost/tuple/detail/tuple_basic.hpp>

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

namespace MeltPoolDG
{
  template <int dim,
            int n_solution_components,
            typename number,
            typename VectorizedArrayType = dealii::VectorizedArray<number>,
            typename VectorType          = dealii::LinearAlgebra::distributed::Vector<number>>
  class GenericStaggeredLoopOperator
  {
    using ValueType = dealii::Tensor<1, n_solution_components, VectorizedArrayType>;
    using GradientType =
      dealii::Tensor<1, n_solution_components, dealii::Tensor<1, dim, VectorizedArrayType>>;

    using CellQuadOpType = std::function<std::tuple<ValueType, GradientType>(
      const unsigned int                             cell_batch_id,
      const ValueType                               &w_q,
      const GradientType                            &grad_w_q,
      const dealii::Point<dim, VectorizedArrayType> &quadrature_point)>;

    using InnerFaceQuadOpType =
      std::function<std::tuple<ValueType, GradientType, ValueType, GradientType>(
        const std::array<unsigned int, VectorizedArrayType::size()> &cell_ids_m,
        const std::array<unsigned int, VectorizedArrayType::size()> &cell_ids_p,
        const ValueType                                             &w_m,
        const GradientType                                          &grad_w_m,
        const ValueType                                             &w_p,
        const GradientType                                          &grad_w_p,
        const dealii::Tensor<1, dim, VectorizedArrayType>           &normal,
        const dealii::Point<dim, VectorizedArrayType>               &quadrature_point)>;

    using BoundaryFaceQuadOpType = std::function<std::tuple<ValueType, GradientType>(
      const std::array<unsigned int, VectorizedArrayType::size()> &cell_ids_m,
      const ValueType                                             &w_m,
      const GradientType                                          &grad_w_m,
      const dealii::Tensor<1, dim, VectorizedArrayType>           &normal,
      const dealii::Point<dim, VectorizedArrayType>               &quadrature_point,
      const dealii::types::boundary_id                             boundary_id)>;

  public:
    /**
     * Constructor.
     *
     * @param mf_context The matrix-free context containing the necessary information for
     * the matrix-free loops, such as the matrix-free object, the dof and quadrature indices.
     */
    explicit GenericStaggeredLoopOperator(const MatrixFreeContext<dim, number> mf_context)
      : mf_context(mf_context)
    {}

    /**
     * Adds a operation performed at each quadrature point in the cell loop.
     */
    template <typename F>
    void
    add_cell_quadrature_operation(const F &operation)
    {
      cell_quadrature_operations.push_back(operation);
    }

    /**
     * Adds a operation performed at each quadrature point in the inner face loop.
     */
    template <typename F>
    void
    add_inner_face_quadrature_operation(const F &operation)
    {
      inner_face_quadrature_operations.push_back(operation);
    }

    /**
     * Adds a operation performed at each quadrature point in the boundary face loop.
     */
    template <typename F>
    void
    add_boundary_face_quadrature_operation(const F &operation)
    {
      boundary_face_quadrature_operations.push_back(operation);
    }

    /**
     * Adds evaluation and integration flags for the face loops. These flags determine which data is
     * gathered at the quadrature points and which data is scattered back to the degrees of freedom
     * after the quadrature point operations are performed. The flags are added in a cumulative way,
     * i.e., if this function is called multiple times, the provided flags will be added to the
     * already existing ones.
     *
     * @param evaluation_flags The evaluation flags to be added. These flags determine which data is
     * gathered at the quadrature points.
     * @param integration_flags The integration flags to be added. These flags determine which data
     * is scattered back to the degrees of freedom after the quadrature point operations are
     * performed.
     */
    void
    add_face_eval_flags(const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
                        const dealii::EvaluationFlags::EvaluationFlags integration_flags)
    {
      eval_flags.face.evaluation  = evaluation_flags;
      eval_flags.face.integration = integration_flags;
    }

    /**
     * Adds evaluation and integration flags for the cell loops. These flags determine which data is
     * gathered at the quadrature points and which data is scattered back to the degrees of freedom
     * after the quadrature point operations are performed. The flags are added in a cumulative way,
     * i.e., if this function is called multiple times, the provided flags will be added to the
     * already existing ones.
     *
     * @param evaluation_flags The evaluation flags to be added. These flags determine which data is
     * gathered at the quadrature points.
     * @param integration_flags The integration flags to be added. These flags determine which data
     * is scattered back to the degrees of freedom after the quadrature point operations are
     * performed.
     */
    void
    add_cell_eval_flags(const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
                        const dealii::EvaluationFlags::EvaluationFlags integration_flags)
    {
      eval_flags.cell.evaluation  = evaluation_flags;
      eval_flags.cell.integration = integration_flags;
    }

    /**
     * Performs the matrix-free loop evaluating the provided quadrature point operations and using
     * the provided evaluation and integration flags.
     *
     * @param dst The destination vector.
     * @param src The source vector.
     * @param zero_dst_vector Whether to zero the destination vector before the loop.
     */
    void
    matrix_free_loop(VectorType &dst, const VectorType &src, const bool zero_dst_vector) const
    {
      using local_applier_type = std::function<void(const dealii::MatrixFree<dim, number> &,
                                                    VectorType       &dst,
                                                    const VectorType &src,
                                                    const std::pair<unsigned int, unsigned int> &)>;

      local_applier_type cell          = MPDG_LAMBDA_WRAPPER(this->local_apply_cell);
      local_applier_type face          = MPDG_LAMBDA_WRAPPER(this->local_apply_face);
      local_applier_type boundary_face = MPDG_LAMBDA_WRAPPER(this->local_apply_boundary_face);
      mf_context.mf.loop(cell, face, boundary_face, dst, src, zero_dst_vector);
    }

  private:
    struct
    {
      struct EvalFlags
      {
        dealii::EvaluationFlags::EvaluationFlags evaluation;
        dealii::EvaluationFlags::EvaluationFlags integration;
      };

      EvalFlags cell;
      EvalFlags face;
    } eval_flags;

    std::vector<CellQuadOpType>         cell_quadrature_operations;
    std::vector<InnerFaceQuadOpType>    inner_face_quadrature_operations;
    std::vector<BoundaryFaceQuadOpType> boundary_face_quadrature_operations;

    MatrixFreeContext<dim, number> mf_context;

     void
    local_apply_cell(const dealii::MatrixFree<dim, number> &mf,
                     VectorType                            &dst,
                     const VectorType                      &src,
                     const std::pair<unsigned, unsigned>   &cell_range) const
    {
      FECellIntegrator<dim, n_solution_components, number> phi(mf,
                                                               mf_context.dof_idx,
                                                               mf_context.quad_idx);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src, eval_flags.cell.evaluation);

          for (const unsigned int q : phi.quadrature_point_indices())
            {
              ValueType    value;
              GradientType gradient;

              for (const auto &op : cell_quadrature_operations)
                {
                  tie_add(std::tie(value, gradient),
                          op(cell, phi.get_value(q), phi.get_gradient(q), phi.quadrature_point(q)));
                }

              phi.submit_value(value, q);
              phi.submit_gradient(gradient, q);
            }

          phi.integrate_scatter(eval_flags.cell.integration, dst);
        }
    }

    void
    local_apply_face(const dealii::MatrixFree<dim, number>       &mf,
                     VectorType                                  &dst,
                     const VectorType                            &src,
                     const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceIntegrator<dim, n_solution_components, number> phi_m(mf,
                                                                 true,
                                                                 mf_context.dof_idx,
                                                                 mf_context.quad_idx);
      FEFaceIntegrator<dim, n_solution_components, number> phi_p(mf,
                                                                 false,
                                                                 mf_context.dof_idx,
                                                                 mf_context.quad_idx);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          phi_p.reinit(face);
          phi_p.gather_evaluate(src, eval_flags.face.evaluation);

          phi_m.reinit(face);
          phi_m.gather_evaluate(src, eval_flags.face.evaluation);

          for (const unsigned int q : phi_m.quadrature_point_indices())
            {
              ValueType value_m;
              ValueType value_p;

              GradientType gradient_m;
              GradientType gradient_p;

              for (const auto &op : inner_face_quadrature_operations)
                {
                  tie_add(std::tie(value_m, gradient_m, value_p, gradient_p),
                          op(phi_m.get_cell_ids(),
                             phi_p.get_cell_ids(),
                             phi_m.get_value(q),
                             phi_m.get_gradient(q),
                             phi_p.get_value(q),
                             phi_p.get_gradient(q),
                             phi_m.normal_vector(q),
                             phi_m.quadrature_point(q)));
                }
              phi_m.submit_value(value_m, q);
              phi_p.submit_value(value_p, q);

              phi_m.submit_gradient(gradient_m, q);
              phi_p.submit_gradient(gradient_p, q);
            }

          phi_m.integrate_scatter(eval_flags.face.integration, dst);
          phi_p.integrate_scatter(eval_flags.face.integration, dst);
        }
    }

    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number> &mf,
                              VectorType                            &dst,
                              const VectorType                      &src,
                              const std::pair<unsigned, unsigned>   &face_range) const
    {
      FEFaceIntegrator<dim, n_solution_components, number> phi_m(mf,
                                                                 true,
                                                                 mf_context.dof_idx,
                                                                 mf_context.quad_idx);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          phi_m.reinit(face);
          phi_m.gather_evaluate(src, eval_flags.face.evaluation);

          for (const unsigned int q : phi_m.quadrature_point_indices())
            {
              ValueType    value_m;
              GradientType gradient_m;

              for (const auto &op : boundary_face_quadrature_operations)
                {
                  tie_add(std::tie(value_m, gradient_m),
                          op(phi_m.get_cell_ids(),
                             phi_m.get_value(q),
                             phi_m.get_gradient(q),
                             phi_m.normal_vector(q),
                             phi_m.quadrature_point(q),
                             phi_m.boundary_id()));
                }

              phi_m.submit_value(value_m, q);
              phi_m.submit_gradient(gradient_m, q);
            }

          phi_m.integrate_scatter(eval_flags.face.integration, dst);
        }
    }
  };
} // namespace MeltPoolDG
