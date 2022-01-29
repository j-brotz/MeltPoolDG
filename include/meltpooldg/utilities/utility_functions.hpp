#pragma once
// dealii
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

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

  namespace TypeDefs
  {
    enum class VerbosityType
    {
      silent,
      major,
      detailed
    };
  } // namespace TypeDefs

  namespace UtilityFunctions
  {
    /**
     * For a given @p matrix_free object, execute scalar- or vector-valued @p cell_operation
     * on each quadrature point  defined by @p quad_idx and fill them into a
     * DoF-vector @p vec defined by @p dof_idx.
     */
    template <int dim, int n_components, typename T>
    void
    fill_dof_vector_from_cell_operation(
      VectorType &                                            vec,
      const MatrixFree<dim, double, VectorizedArray<double>> &matrix_free,
      unsigned int                                            dof_idx,
      unsigned int                                            quad_idx,
      const T &                                               cell_operation)
    {
      FECellIntegrator<dim, n_components, double> fe_eval(matrix_free, dof_idx, quad_idx);

      MatrixFreeOperators::
        CellwiseInverseMassMatrix<dim, -1, n_components, double, VectorizedArray<double>>
          inverse_mass_matrix(fe_eval);

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

          // write values back into global vector
          fe_eval.set_dof_values(vec);
        }
    }

    /*
     * This function converts a string of coordinates given as e.g. "5,10,5" to a Point<dim>
     * object.
     */
    template <int dim>
    Point<dim>
    convert_string_coords_to_point(const std::string s_in, const std::string delimiter = ",")
    {
      Point<dim>  p;
      int         d   = 0;
      size_t      pos = 0;
      std::string coord;
      std::string s = s_in;

      // split parts between delimiters
      while ((pos = s.find(delimiter)) != std::string::npos)
        {
          coord = s.substr(0, pos);

          if (d < dim)
            p[d] = std::stod(coord);
          else
            break;
          s.erase(0, pos + delimiter.length());
          d++;
        }
      // last part after delimiter
      if (d < dim)
        p[d] = std::stod(s);
      return p;
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

    /**
     * For a given CellAccessor @param cell, compute the interface thickness parameter
     * epsilon from the characteristic cell length and a given @param thickness_scale_factor.
     */
    template <int dim, typename cell_type>
    inline double
    compute_cell_size_dependent_interface_thickness(const cell_type cell,
                                                    const double    thickness_scale_factor)
    {
      return cell->diameter() / std::sqrt(dim) * thickness_scale_factor;
    }

    /**
     * For a given @param cell_diameter, compute the interface thickness parameter
     * epsilon using a given @param thickness_scale_factor.
     */
    template <int dim>
    inline double
    compute_cell_size_dependent_interface_thickness(const double cell_diameter,
                                                    const double thickness_scale_factor)
    {
      return cell_diameter / std::sqrt(dim) * thickness_scale_factor;
    }

    template <int dim>
    inline double
    compute_initial_epsilon(const Parameters<double> &param, const Triangulation<dim> &tria)
    {
      double eps = param.reinit.constant_epsilon > 0.0 ?
                     param.reinit.constant_epsilon :
                     compute_cell_size_dependent_interface_thickness<dim>(
                       GridTools::minimal_cell_diameter(tria), param.reinit.scale_factor_epsilon);

      // divide by the number of subdivisions if a mesh-dependent interface thickness is computed
      if (param.reinit.constant_epsilon <= 0.0)
        eps /= param.ls.n_subdivisions;
      return eps;
    }

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
        return floor(log10(x));
      else if (x <= 1e-16) // negative number
        return floor(log10(abs(x)));
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
                                 const Point<dim> &       starting_point,
                                 const vector &           unit_vec,
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
                                   const Mapping<dim> &      mapping)
    {
      return std::min(1e-2,
                      std::max(std::pow(10,
                                        UtilityFunctions::get_exponent_power_ten(
                                          std::pow(GridTools::volume<dim>(triangulation, mapping),
                                                   1. / dim))) *
                                 1e-3,
                               1e-12));
    }

    template <int dim, int spacedim, typename number>
    void
    check_constraints(const DoFHandler<dim, spacedim> &dof_handler,
                      const AffineConstraints<number> &constraints)
    {
#ifndef DEBUG
      return;
#endif

      IndexSet locally_active_dofs;
      DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

      AssertThrow(constraints.is_consistent_in_parallel(
                    Utilities::MPI::all_gather(dof_handler.get_communicator(),
                                               dof_handler.locally_owned_dofs()),
                    locally_active_dofs,
                    dof_handler.get_communicator()),
                  ExcInternalError());
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
                             VectorType &      predictor,
                             const double      current_time_increment,
                             const double      old_time_increment)
    {
      predictor.copy_locally_owned_data_from(old_vec);
      predictor.add(current_time_increment / old_time_increment,
                    old_vec,
                    -current_time_increment / old_time_increment,
                    old_old_vec);
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
     * @note: semantics slightly modified
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
                 dynamic_cast<const FE_Q_DG0<dim> *>(&dof_handler_1.get_fe()))
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
     * @note: The interpolation_matrix should be computed using
     *        UtilityFunctions::create_dof_interpolation_matrix().
     *
     */
    template <int dim>
    void
    compute_gradient_at_interpolated_dof_values(
      FECellIntegrator<dim, 1, double> &values,
      FECellIntegrator<dim, 1, double> &interpolated_values,
      const FullMatrix<double> &        interpolation_matrix)
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
  } // namespace UtilityFunctions
} // namespace MeltPoolDG
