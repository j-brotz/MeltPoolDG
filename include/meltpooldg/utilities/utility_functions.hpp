#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/utilities/fe_integrator.hpp>

#include <algorithm>
#include <cmath>
#include <ios>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace MeltPoolDG
{
  enum BooleanType
  {
    Union,
    Intersection,
    Subtraction
  };

  namespace UtilityFunctions
  {
    template <typename number1, typename number2>
    void
    remove_duplicates(std::vector<number1> &a, std::vector<number2> &b)
    {
      std::unordered_set<int> seen;
      auto                    it_a = a.begin();
      auto                    it_b = b.begin();

      while (it_a != a.end())
        {
          if (seen.find(*it_a) != seen.end())
            {
              // If element is duplicate, remove it from both vectors
              it_a = a.erase(it_a);
              it_b = b.erase(it_b);
            }
          else
            {
              // If element is not duplicate, mark it as seen and move iterators forward
              seen.insert(*it_a);
              ++it_a;
              ++it_b;
            }
        }
    }

    template <typename T>
    std::string
    to_string_with_precision(const T a_value, const int n = 6)
    {
      std::ostringstream out;
      out.precision(n);
      out << std::scientific << a_value;
      return out.str();
    }

    /**
     * This function returns heaviside values for a given VectorizedArray. The limit to
     * distinguish between 0 and 1 can be adjusted by the argument "limit". This function is
     * particularly suited in the context of MatrixFree routines.
     */
    template <typename number>
    number
    heaviside(const number in, const number limit = 0.0)
    {
      return in > limit ? 1.0 : 0.0;
    }

    template <typename number>
    std::vector<number>
    heaviside(const std::vector<number> in, const number limit = 0.0)
    {
      std::vector<number> out(in.size());
      for (unsigned int i = 0; i < in.size(); ++i)
        out[i] = heaviside(in[i], limit);

      return out;
    }

    template <typename number>
    dealii::VectorizedArray<number>
    heaviside(const dealii::VectorizedArray<number> &in, const number limit = 0.0)
    {
      return compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
        in, dealii::VectorizedArray<number>(limit), 1.0, 0.0);
    }

    namespace CharacteristicFunctions
    {
      template <typename number>
      inline number
      tanh_characteristic_function(const number &distance, const number &eps)
      {
        return std::tanh(distance / (2. * eps));
      }

      template <typename number>
      inline number
      heaviside(const number &distance, const number &eps)
      {
        if (distance > eps)
          return 1;
        else if (distance <= -eps)
          return 0;
        else
          return (distance + eps) / (2. * eps) +
                 1. / (2. * dealii::numbers::PI) * std::sin(dealii::numbers::PI * distance / eps);
      }

      template <typename number>
      inline int
      sgn(const number &x)
      {
        return (x < 0) ? -1 : 1;
      }

      template <typename number>
      inline number
      normalize(const number &x, const number &x_min, const number &x_max)
      {
        return (x - x_min) / (x_max - x_min);
      }

    } // namespace CharacteristicFunctions

    template <typename number>
    inline dealii::VectorizedArray<number>
    limit_to_bounds(const dealii::VectorizedArray<number> &in,
                    const number                           lower_limit,
                    const number                           upper_limit)
    {
      const auto ub = compare_and_apply_mask<dealii::SIMDComparison::greater_than>(in,
                                                                                   upper_limit,
                                                                                   upper_limit,
                                                                                   in);
      return compare_and_apply_mask<dealii::SIMDComparison::less_than>(ub,
                                                                       lower_limit,
                                                                       lower_limit,
                                                                       ub);
    }

    template <typename number>
    inline number
    limit_to_bounds(const number in, const number lower_limit, const number upper_limit)
    {
      const auto ub = in > upper_limit ? upper_limit : in;
      return ub < lower_limit ? lower_limit : ub;
    }

    /**
     * This function returns 1.0 if the \p in value is between (excluding) the \p lower_limit and
     * \p upper_limit. Otherwise, this function returns 0.0.
     */
    template <typename number>
    dealii::VectorizedArray<number>
    is_between(const dealii::VectorizedArray<number> &in,
               const number                           lower_limit,
               const number                           upper_limit)
    {
      return compare_and_apply_mask<dealii::SIMDComparison::less_than>(
        in,
        upper_limit,
        compare_and_apply_mask<dealii::SIMDComparison::greater_than>(in, lower_limit, 1.0, 0.0),
        0.0);
    }

    /**
     * Return the exponent to the power of ten of an expression like 5*10^5 --> return 5
     */
    template <typename number>
    inline int
    get_exponent_power_ten(const number x)
    {
      if (x >= 1e-16) // positive number
        return std::floor(std::log10(x));
      else if (x <= 1e-16) // negative number
        return std::floor(std::log10(std::abs(x)));
      else // number close to 0
        return 0;
    }

    /**
     * Starting from a @p starting_point, generate points along the given @p
     * unit_vec with a given number of steps @p n_inc_per_side and ranging up to a
     * @p max_distance_per_side. If @p bidirectional is set to true, points on both
     * sides of the @p starting_point are generated.
     */
    template <int dim, typename vector, typename number>
    void
    generate_points_along_vector(std::vector<dealii::Point<dim>> &points_normal_to_interface,
                                 const dealii::Point<dim>        &starting_point,
                                 const vector                    &unit_vec,
                                 const number                     max_distance_per_side,
                                 const unsigned int               n_inc_per_side,
                                 const bool                       bidirectional = true)
    {
      const number step = max_distance_per_side / n_inc_per_side;
      for (int n = n_inc_per_side; n >= 0; --n)
        points_normal_to_interface.emplace_back(starting_point + unit_vec * n * step);
      if (bidirectional)
        {
          for (unsigned int n = 1; n <= n_inc_per_side; ++n)
            points_normal_to_interface.emplace_back(starting_point - unit_vec * n * step);
        }
    }


    /*
     * 1d numerical integration of @p vals given at @p points using a trapezoidal rule.
     */
    template <int dim, typename number>
    number
    integrate_over_line(const std::vector<number>             &vals,
                        const std::vector<dealii::Point<dim>> &points)
    {
      number result = 0;
      for (unsigned int i = 0; i < vals.size() - 1; ++i)
        result += (vals[i] + vals[i + 1]) / 2. * (points[i + 1].distance(points[i]));

      return result;
    }

    template <int dim, typename number>
    number
    compute_numerical_zero_of_norm(const dealii::Triangulation<dim> &triangulation,
                                   const dealii::Mapping<dim>       &mapping)
    {
      return std::min(1e-2,
                      std::max(std::pow(10,
                                        UtilityFunctions::get_exponent_power_ten(std::pow(
                                          dealii::GridTools::volume<dim>(triangulation, mapping),
                                          1. / dim))) *
                                 1e-3,
                               1e-12));
    }

    /**
     * This function creates a n x m interpolation matrix P, to interpolate DoF values
     * per cell from one space (n) given by @param dof_handler_1 to another space (m)
     * given py @param dof_handler_2.
     *
     * The interpolation of cell-wise DoF values x, using the matrix P can be done
     * as follows
     *   _
     *   x   = P   x
     *    i     ij  j
     *
     * with i=0...n-1 and j=0...m-1.
     *
     * @note The row/column indices are sorted in lexicographic order.
     *
     * @note Enable do_sort_lexicographically if the interpolation matrix
     *   should be used in matrix-free loops.
     *
     *
     * ---------------------------------------------------------------------------------
     * Copied from adaflo:
     *
     * https://github.com/kronbichler/adaflo/blob/f873472c43798304bbdb7f0cbeb556061c489020/source/level_set_base.cc#L68-L137
     *
     * @note semantics slightly modified
     * ---------------------------------------------------------------------------------
     */
    template <int dim, typename number>
    dealii::FullMatrix<number>
    create_dof_interpolation_matrix(const dealii::DoFHandler<dim> &dof_handler_1, // e.g. pressure
                                    const dealii::DoFHandler<dim> &dof_handler_2, // e.g. level set
                                    const bool                     do_sort_lexicographically)
    {
      dealii::FullMatrix<number> dof_interpolation_matrix(dof_handler_1.get_fe().n_dofs_per_cell(),
                                                          dof_handler_2.get_fe().n_dofs_per_cell());

      const dealii::FE_Q_iso_Q1<dim> *fe_2 =
        dynamic_cast<const dealii::FE_Q_iso_Q1<dim> *>(&dof_handler_2.get_fe());

      AssertThrow(fe_2,
                  dealii::ExcMessage(
                    "dof_handler_2 must contain finite elements of type FE_Q_iso_Q1."));

      const std::vector<unsigned int> lexicographic_ls = fe_2->get_poly_space_numbering_inverse();


      //@todo: get rid of base_element
      if (const dealii::FE_Q<dim> *fe_1 =
            dynamic_cast<const dealii::FE_Q<dim> *>(&dof_handler_1.get_fe().base_element(0)))
        {
          const std::vector<unsigned int> lexicographic_p =
            fe_1->get_poly_space_numbering_inverse();
          for (unsigned int j = 0; j < fe_1->dofs_per_cell; ++j)
            {
              const dealii::Point<dim> p =
                fe_1->get_unit_support_points()[do_sort_lexicographically ? lexicographic_p[j] : j];
              for (unsigned int i = 0; i < fe_2->dofs_per_cell; ++i)
                dof_interpolation_matrix(j, i) = dof_handler_2.get_fe().shape_value(
                  do_sort_lexicographically ? lexicographic_ls[i] : i, p);
            }
        }
      else if (const dealii::FE_Q_DG0<dim> *fe_1 = dynamic_cast<const dealii::FE_Q_DG0<dim> *>(
                 &dof_handler_1.get_fe().base_element(0)))
        {
          const std::vector<unsigned int> lexicographic_p =
            fe_1->get_poly_space_numbering_inverse();

          // Loop over all support points except the one for the discontinuous
          // shape function in the middle of the cell (dofs_per_cell - 1).
          for (unsigned int j = 0; j < fe_1->dofs_per_cell - 1; ++j)
            {
              const dealii::Point<dim> p =
                fe_1->get_unit_support_points()[do_sort_lexicographically ? lexicographic_p[j] : j];
              for (unsigned int i = 0; i < fe_2->dofs_per_cell; ++i)
                dof_interpolation_matrix(j, i) = dof_handler_2.get_fe().shape_value(
                  do_sort_lexicographically ? lexicographic_ls[i] : i, p);
            }
        }
      else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "The operation for the requested pair of DoFHandler "
                      "is not supported. Types must be: dof_handler_1 = FE_Q_iso_Q1;"
                      " dof_handler_2 = FE_Q || FE_Q_DG0."));

      return dof_interpolation_matrix;
    }


    /**
     * Compute from @param values of a given field, @param interpolated_values by
     * means of a given @param interpolation_matrix. Finally, from the interpolated_values
     * the gradients are evaluated.
     *
     * @note The interpolation_matrix should be computed using
     *       UtilityFunctions::create_dof_interpolation_matrix().
     *
     */
    template <int dim, typename number>
    void
    compute_gradient_at_interpolated_dof_values(
      FECellIntegrator<dim, 1, number> &values,
      FECellIntegrator<dim, 1, number> &interpolated_values,
      const dealii::FullMatrix<number> &interpolation_matrix)
    {
      // Evaluate the field Φ at the support points of its space j
      values.evaluate(dealii::EvaluationFlags::values);

      // Loop over the support points of the to be interpolated space i
      for (unsigned int i = 0; i < interpolated_values.dofs_per_cell; ++i)
        {
          dealii::VectorizedArray<number> interpolated_value = 0;

          // Interpolate the field Φ from the support points of the space j
          // of the original field to the one of the interpolated field i,
          // using the interpolation matrix P
          // _
          // Φ   = P   · Φ
          //  i     ij    j
          for (unsigned int j = 0; j < values.dofs_per_cell; ++j)
            interpolated_value += interpolation_matrix(i, j) * values.get_dof_value(j);

          // Store the interpolated values at the support points of the pressure space
          interpolated_values.submit_dof_value(interpolated_value, i);
        }

      // Evaluate the gradient from the interpolated field
      //                       _
      //                     ∇ Φ
      //
      interpolated_values.evaluate(dealii::EvaluationFlags::gradients);
    }

    template <int dim, typename ForwardIterator>
    dealii::Point<dim>
    to_point(const ForwardIterator begin, const ForwardIterator end)
    {
      (void)end;
      AssertIndexRange(dim, std::distance(begin, end) + 1);

      dealii::Point<dim> point;

      auto it = begin;
      for (int i = 0; i < dim; ++i)
        point[i] = *it++;

      return point;
    }


    // TODO: Check performance when the functions below are inlined.
    /**
     * Compute the dyadic product of two rank-1 tensors passing a pointer to the start of the values
     * of the two tensors.
     */
    template <int T1_dim, int T2_dim, typename number>
    dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>>
    dyadic_product(const number *a_start, const number *b_start)
    {
      dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>> c;
      for (unsigned int i = 0; i < T1_dim; ++i)
        {
          for (unsigned int j = 0; j < T2_dim; ++j)
            {
              c[i][j] = *(a_start + i) * *(b_start + j);
            }
        }
      return c;
    }

    /**
     * Compute the dyadic product of two rank-1 tensors
     */
    template <int T1_dim, int T2_dim, typename number>
    dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>>
    dyadic_product(const dealii::Tensor<1, T1_dim, number> &a,
                   const dealii::Tensor<1, T2_dim, number> &b)
    {
      return dyadic_product<T1_dim, T2_dim, number>(&a[0], &b[0]);
    }

    /**
     * Return the transpose of a dealii::Tensor<dealii::Tensor>
     */
    template <int T1_dim, int T2_dim, typename number>
    dealii::Tensor<1, T2_dim, dealii::Tensor<1, T1_dim, number>>
    transpose(const dealii::Tensor<1, T1_dim, dealii::Tensor<1, T2_dim, number>> &in)
    {
      dealii::Tensor<1, T2_dim, dealii::Tensor<1, T1_dim, number>> out;
      for (unsigned int i = 0; i < T1_dim; ++i)
        for (unsigned int j = 0; j < T2_dim; ++j)
          out[j][i] = in[i][j];
      return out;
    }

    /**
     * Return the trace of a dealii::Tensor<dealii::Tensor>
     */
    template <int dim, typename number>
    number
    trace(const dealii::Tensor<1, dim, dealii::Tensor<1, dim, number>> &in)
    {
      number tr(0.0);
      for (unsigned int i = 0; i < dim; ++i)
        tr += in[i][i];
      return tr;
    }

    template <int dim, typename number>
    number
    trace(const dealii::Tensor<2, dim, number> &in)
    {
      number tr(0.0);
      for (unsigned int i = 0; i < dim; ++i)
        tr += in[i][i];
      return tr;
    }

    /**
     * Return identity matrix
     */
    template <int dim, typename number>
    dealii::SymmetricTensor<2, dim, number>
    identity()
    {
      dealii::Tensor<1, dim, dealii::Tensor<1, dim, number>> out;
      for (unsigned int i = 0; i < dim; ++i)
        out[i][i] = number(1.0);
      return dealii::SymmetricTensor<2, dim, number>(std::array<number, dim>(1.));
    }

    /**
     * Helper functions for matrix-vector and matrix-matrix computations when both matrix and vector
     * are implemented as dealii::Tensor.
     */
    template <int n_rows, int n_columns, typename number>
    dealii::Tensor<1, n_rows, number>
    matrix_vector_product(
      const dealii::Tensor<1, n_rows, dealii::Tensor<1, n_columns, number>> &matrix,
      const dealii::Tensor<1, n_columns, number>                            &vector)
    {
      dealii::Tensor<1, n_rows, number> result;
      for (unsigned int i = 0; i < n_rows; ++i)
        for (unsigned int j = 0; j < n_columns; ++j)
          result[i] += matrix[i][j] * vector[j];
      return result;
    }

    template <int a, int b, int c, typename number>
    dealii::Tensor<1, a, dealii::Tensor<1, c, number>>
    matrix_matrix_product(const dealii::Tensor<1, a, dealii::Tensor<1, b, number>> &matrix1,
                          const dealii::Tensor<1, b, dealii::Tensor<1, c, number>> &matrix2)
    {
      dealii::Tensor<1, a, dealii::Tensor<1, c, number>> result;
      for (unsigned int i = 0; i < a; ++i)
        for (unsigned int j = 0; j < c; ++j)
          for (unsigned int k = 0; k < b; ++k)
            result[i][j] += matrix1[i][k] * matrix2[k][j];
      return result;
    }

    /**
     * Helper function for the computation of a weighted average:
     * weight_a * term_a + weight_b * term_b
     */
    template <typename TypeWeight, typename TypeTerm>
    TypeTerm
    calculate_arithmetic_phase_weighted_average(const TypeWeight &weight_a,
                                                const TypeTerm   &term_a,
                                                const TypeWeight &weight_b,
                                                const TypeTerm   &term_b)
    {
      return weight_a * term_a + weight_b * term_b;
    }

    /**
     * Contracts the given second order tensor with a vector, which results in a vector. Note that
     * the second order tensor is provided as two nested first order tensors in this function.
     *
     * @param tensor Second order tensor of type
     * 'dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>>'
     * @param vector Vector of type 'dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>'
     *
     * @return Result of the contraction, i.e. result_i = tensor_ij * vector_j. The result has vector
     * type 'dealii::Tensor<1, dim_1, dealii::VectorizedArray<number>>'.
     *
     * @ note The dimensions of the provided tensor and the vector have to match, i.e. the second
     * basis of the tensor and the vector must have the same dimension @p dim_2.
     */
    template <int dim_1, int dim_2, typename number>
    dealii::Tensor<1, dim_1, dealii::VectorizedArray<number>>
    contract_tensor_with_vector(
      const dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>>
                                                                      &tensor,
      const dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>> &vector)
    {
      dealii::Tensor<1, dim_1, dealii::VectorizedArray<number>> result;

      for (unsigned int i = 0; i < dim_1; ++i)
        result[i] = tensor[i] * vector;

      return result;
    }

    /**
     * Contracts the average of two given second order tensors with a vector, which results in a
     * vector. Note that the second order tensors are provided as two nested first order tensors in
     * this function.
     *
     * @param tensor_1 First second order tensor of type
     * 'dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>>'
     * @param tensor_2 Second second order tensor of type
     * 'dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>>'
     * @param vector Vector of type 'dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>'
     *
     * @return Result of the contraction, i.e. result_i = 0.5 * (tensor_1_ij + tensor_2_ij) * vector_j.
     * The result has vector type 'dealii::Tensor<1, dim_1, dealii::VectorizedArray<number>>'.
     *
     * @ note The dimensions of the provided tensors and the vector have to match, i.e. the second
     * basis of the tensors and the vector must have the same dimension @p dim_2.
     */
    template <int dim_1, int dim_2, typename number>
    dealii::Tensor<1, dim_1, dealii::VectorizedArray<number>>
    contract_average_tensor_with_vector(
      const dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>>
        &tensor_1,
      const dealii::Tensor<1, dim_1, dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>>>
                                                                      &tensor_2,
      const dealii::Tensor<1, dim_2, dealii::VectorizedArray<number>> &vector)
    {
      dealii::Tensor<1, dim_1, dealii::VectorizedArray<number>> result;

      for (unsigned int i = 0; i < dim_1; ++i)
        result[i] = (tensor_1[i] + tensor_2[i]) * vector;

      return 0.5 * result;
    }

  } // namespace UtilityFunctions
} // namespace MeltPoolDG
