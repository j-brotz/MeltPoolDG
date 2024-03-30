#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include <meltpooldg/utilities/fe_integrator.hpp>

#include <algorithm>
#include <cmath>
#include <ios>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace dealii
{
  enum TriangulationType
  {
    shared,
    parallel_distributed,
    parallel_fullydistributed,
    serial
  };

  template <int dim, int spacedim = dim>
  TriangulationType
  get_triangulation_type(const Triangulation<dim, spacedim> &tria)
  {
    if (dynamic_cast<parallel::shared::Triangulation<dim, spacedim> *>(
          const_cast<Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::shared;
    else if (dynamic_cast<parallel::distributed::Triangulation<dim, spacedim> *>(
               const_cast<Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::parallel_distributed;
    else if (dynamic_cast<parallel::fullydistributed::Triangulation<dim, spacedim> *>(
               const_cast<Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::parallel_fullydistributed;
    else
      return TriangulationType::serial;
  }

  // note: the content of this namespace will be part of deal.II
  namespace internal
  {
    template <typename VectorType,
              std::enable_if_t<!IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    unsigned int
    n_blocks(const VectorType &)
    {
      return 1;
    }



    template <typename VectorType,
              std::enable_if_t<IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    unsigned int
    n_blocks(const VectorType &vector)
    {
      return vector.n_blocks();
    }



    template <typename VectorType,
              std::enable_if_t<!IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    VectorType &
    block(VectorType &vector, const unsigned int b)
    {
      AssertDimension(b, 0);
      (void)b;
      return vector;
    }



    template <typename VectorType,
              std::enable_if_t<IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    typename VectorType::BlockType &
    block(VectorType &vector, const unsigned int b)
    {
      return vector.block(b);
    }



    template <typename VectorType,
              std::enable_if_t<!IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    const VectorType &
    block(const VectorType &vector, const unsigned int b)
    {
      AssertDimension(b, 0);
      (void)b;
      return vector;
    }



    template <typename VectorType,
              std::enable_if_t<IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    const typename VectorType::BlockType &
    block(const VectorType &vector, const unsigned int b)
    {
      return vector.block(b);
    }
  } // namespace internal
} // namespace dealii

namespace MeltPoolDG
{
  using namespace dealii;
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

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
    VectorizedArray<number>
    heaviside(const VectorizedArray<number> &in, const number limit = 0.0)
    {
      return compare_and_apply_mask<SIMDComparison::greater_than>(in,
                                                                  VectorizedArray<double>(limit),
                                                                  1.0,
                                                                  0.0);
    }

    namespace CharacteristicFunctions
    {
      inline double
      tanh_characteristic_function(const double &distance, const double &eps)
      {
        return std::tanh(distance / (2. * eps));
      }

      inline double
      heaviside(const double &distance, const double &eps)
      {
        if (distance > eps)
          return 1;
        else if (distance <= -eps)
          return 0;
        else
          return (distance + eps) / (2. * eps) +
                 1. / (2. * numbers::PI) * std::sin(numbers::PI * distance / eps);
      }

      inline int
      sgn(const double &x)
      {
        return (x < 0) ? -1 : 1;
      }

      inline double
      normalize(const double &x, const double &x_min, const double &x_max)
      {
        return (x - x_min) / (x_max - x_min);
      }

    } // namespace CharacteristicFunctions

    template <typename number>
    inline VectorizedArray<number>
    limit_to_bounds(const VectorizedArray<number> &in,
                    const number                   lower_limit,
                    const number                   upper_limit)
    {
      const auto ub =
        compare_and_apply_mask<SIMDComparison::greater_than>(in, upper_limit, upper_limit, in);
      return compare_and_apply_mask<SIMDComparison::less_than>(ub, lower_limit, lower_limit, ub);
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
    VectorizedArray<number>
    is_between(const VectorizedArray<number> &in,
               const number                   lower_limit,
               const number                   upper_limit)
    {
      return compare_and_apply_mask<SIMDComparison::less_than>(
        in,
        upper_limit,
        compare_and_apply_mask<SIMDComparison::greater_than>(in, lower_limit, 1.0, 0.0),
        0.0);
    }

    /**
     * Return the exponent to the power of ten of an expression like 5*10^5 --> return 5
     */
    inline int
    get_exponent_power_ten(const double x)
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
    template <int dim, typename vector>
    void
    generate_points_along_vector(std::vector<Point<dim>> &points_normal_to_interface,
                                 const Point<dim>        &starting_point,
                                 const vector            &unit_vec,
                                 const double             max_distance_per_side,
                                 const unsigned int       n_inc_per_side,
                                 const bool               bidirectional = true)
    {
      const double step = max_distance_per_side / n_inc_per_side;
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
    template <int dim>
    double
    integrate_over_line(const std::vector<double> &vals, const std::vector<Point<dim>> &points)
    {
      double result = 0;
      for (unsigned int i = 0; i < vals.size() - 1; ++i)
        result += (vals[i] + vals[i + 1]) / 2. * (points[i + 1].distance(points[i]));

      return result;
    }

    template <int dim>
    double
    compute_numerical_zero_of_norm(const Triangulation<dim> &triangulation,
                                   const Mapping<dim>       &mapping)
    {
      return std::min(1e-2,
                      std::max(std::pow(10,
                                        UtilityFunctions::get_exponent_power_ten(
                                          std::pow(GridTools::volume<dim>(triangulation, mapping),
                                                   1. / dim))) *
                                 1e-3,
                               1e-12));
    }

    /*
     * Compute a linearly extrapolated initial guess value (predictor) for the
     * Newton-Raphson solver at time "t_n+1" from the old solution @param old_vec
     * computed at time "t_n" and the old old solution @param old_old_vec computed
     * at time "t_n-1". In addition the @param current_time_increment
     * dt_n+1 = t_n+1 - t_n and the @param old_time_incrementdt_n = t_n-t_n-1 have
     * to be provided.
     *
     * The predictor is computed as follows
     *
     *                    dt_n+1
     *  q_n+1^(0) = q_n + ------ * (q_n-q_n-1)
     *                    dt_n
     *
     */
    template <typename VectorType>
    void
    compute_linear_predictor(const VectorType &old_vec,
                             const VectorType &old_old_vec,
                             VectorType       &predictor,
                             const double      current_time_increment,
                             const double      old_time_increment)
    {
      if (std::abs(old_time_increment) < 1e-12)
        {
          predictor = old_vec;
          return;
        }

      const double fraction = current_time_increment / old_time_increment;

      for (unsigned int c = 0; c < internal::n_blocks(predictor); ++c)
        DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < internal::block(predictor, c).locally_owned_size(); ++i)
        internal::block(predictor, c).local_element(i) =
          (fraction + 1.0) * internal::block(old_vec, c).local_element(i) -
          fraction * internal::block(old_old_vec, c).local_element(i);
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
    template <int dim>
    FullMatrix<double>
    create_dof_interpolation_matrix(const DoFHandler<dim> &dof_handler_1, // e.g. pressure
                                    const DoFHandler<dim> &dof_handler_2, // e.g. level set
                                    const bool             do_sort_lexicographically)
    {
      FullMatrix<double> dof_interpolation_matrix(dof_handler_1.get_fe().n_dofs_per_cell(),
                                                  dof_handler_2.get_fe().n_dofs_per_cell());

      const FE_Q_iso_Q1<dim> *fe_2 =
        dynamic_cast<const FE_Q_iso_Q1<dim> *>(&dof_handler_2.get_fe());

      AssertThrow(fe_2,
                  ExcMessage("dof_handler_2 must contain finite elements of type FE_Q_iso_Q1."));

      const std::vector<unsigned int> lexicographic_ls = fe_2->get_poly_space_numbering_inverse();


      //@todo: get rid of base_element
      if (const FE_Q<dim> *fe_1 =
            dynamic_cast<const FE_Q<dim> *>(&dof_handler_1.get_fe().base_element(0)))
        {
          const std::vector<unsigned int> lexicographic_p =
            fe_1->get_poly_space_numbering_inverse();
          for (unsigned int j = 0; j < fe_1->dofs_per_cell; ++j)
            {
              const Point<dim> p =
                fe_1->get_unit_support_points()[do_sort_lexicographically ? lexicographic_p[j] : j];
              for (unsigned int i = 0; i < fe_2->dofs_per_cell; ++i)
                dof_interpolation_matrix(j, i) = dof_handler_2.get_fe().shape_value(
                  do_sort_lexicographically ? lexicographic_ls[i] : i, p);
            }
        }
      else if (const FE_Q_DG0<dim> *fe_1 =
                 dynamic_cast<const FE_Q_DG0<dim> *>(&dof_handler_1.get_fe().base_element(0)))
        {
          const std::vector<unsigned int> lexicographic_p =
            fe_1->get_poly_space_numbering_inverse();

          // Loop over all support points except the one for the discontinuous
          // shape function in the middle of the cell (dofs_per_cell - 1).
          for (unsigned int j = 0; j < fe_1->dofs_per_cell - 1; ++j)
            {
              const Point<dim> p =
                fe_1->get_unit_support_points()[do_sort_lexicographically ? lexicographic_p[j] : j];
              for (unsigned int i = 0; i < fe_2->dofs_per_cell; ++i)
                dof_interpolation_matrix(j, i) = dof_handler_2.get_fe().shape_value(
                  do_sort_lexicographically ? lexicographic_ls[i] : i, p);
            }
        }
      else
        AssertThrow(
          false,
          ExcMessage(
            "The operation for the requested pair of DoFHandler "
            "is not supported. Types must be: dof_handler_1 = FE_Q_iso_Q1; dof_handler_2 = FE_Q || FE_Q_DG0."));

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
    template <int dim>
    void
    compute_gradient_at_interpolated_dof_values(
      FECellIntegrator<dim, 1, double> &values,
      FECellIntegrator<dim, 1, double> &interpolated_values,
      const FullMatrix<double>         &interpolation_matrix)
    {
      // Evaluate the field Φ at the support points of its space j
      values.evaluate(EvaluationFlags::values);

      // Loop over the support points of the to be interpolated space i
      for (unsigned int i = 0; i < interpolated_values.dofs_per_cell; ++i)
        {
          VectorizedArray<double> interpolated_value = 0;

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
      interpolated_values.evaluate(EvaluationFlags::gradients);
    }

    template <int dim, typename ForwardIterator>
    Point<dim>
    to_point(const ForwardIterator begin, const ForwardIterator end)
    {
      (void)end;
      AssertIndexRange(dim, std::distance(begin, end) + 1);

      Point<dim> point;

      auto it = begin;
      for (int i = 0; i < dim; ++i)
        point[i] = *it++;

      return point;
    }
  } // namespace UtilityFunctions
} // namespace MeltPoolDG
