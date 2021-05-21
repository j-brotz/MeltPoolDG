#pragma once
// dealii
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG
{
  using namespace dealii;
  using VectorType = LinearAlgebra::distributed::Vector<double>;

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
    template <typename MeshType>
    MPI_Comm
    get_mpi_comm(const MeshType &mesh)
    {
      const auto *tria_parallel =
        dynamic_cast<const parallel::TriangulationBase<MeshType::dimension> *>(
          &(mesh.get_triangulation()));

      return tria_parallel != nullptr ? tria_parallel->get_communicator() : MPI_COMM_SELF;
    }

    template <int dim, int n_components>
    void
    fill_dof_vector_from_cell_operation(
      VectorType &                                                                vec,
      const MatrixFree<dim, double, VectorizedArray<double>> &                    matrix_free,
      unsigned int                                                                dof_idx,
      unsigned int                                                                quad_idx,
      unsigned int                                                                fe_degree,
      unsigned int                                                                n_q_points_1D,
      const std::function<const VectorizedArray<double> &(const unsigned int cell,
                                                          const unsigned int q)> &cell_operation)
    {
      FE_DGQArbitraryNodes<1> fe_coarse(QGauss<1>(n_q_points_1D).get_points());
      FE_Q<1>                 fe_fine(fe_degree);

      /// create 1D projection matrix for sum factorization
      FullMatrix<double> matrix(fe_fine.dofs_per_cell, fe_coarse.dofs_per_cell);
      FETools::get_projection_matrix(fe_coarse, fe_fine, matrix);

      AlignedVector<VectorizedArray<double>> projection_matrix_1d(fe_fine.dofs_per_cell *
                                                                  fe_coarse.dofs_per_cell);

      for (unsigned int i = 0, k = 0; i < fe_coarse.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe_fine.dofs_per_cell; ++j, ++k)
          projection_matrix_1d[k] = matrix(j, i);

      FECellIntegrator<dim, n_components, double> fe_eval(matrix_free, dof_idx, quad_idx);

      // @todo: replace n_q_points_1D argument completely from fe_eval?
      AssertThrow(fe_eval.n_q_points == std::pow(n_q_points_1D, dim),
                  ExcMessage("The number of quadrature points in 1D must comply with the number of "
                             "quadrature points of the cell operation."));

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          fe_eval.reinit(cell);

          for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
            fe_eval.begin_values()[q] = cell_operation(cell, q);

          // perform basis change from quadrature points to support points
          internal::FEEvaluationImplBasisChange<
            internal::evaluate_general,
            internal::EvaluatorQuantity::value,
            dim,
            0,
            0,
            VectorizedArray<double>,
            VectorizedArray<double>>::do_forward(n_components, // n_components
                                                 projection_matrix_1d,
                                                 fe_eval.begin_values(),
                                                 fe_eval.begin_dof_values(),
                                                 n_q_points_1D, // number of
                                                                // quadrature points
                                                 fe_degree + 1  // number of support points
          );

          // write values back into global vector
          fe_eval.set_dof_values(vec);
        }
    }

    /*
     * @todo: replace Tensor<1, n_components, VectorizedArray> as template argument --> C++20
     */
    template <int dim, int n_components>
    void
    fill_dof_vector_from_cell_operation_vec(
      VectorType &                                            vec,
      const MatrixFree<dim, double, VectorizedArray<double>> &matrix_free,
      unsigned int                                            dof_idx,
      unsigned int                                            quad_idx,
      unsigned int                                            fe_degree,
      unsigned int                                            n_q_points_1D,
      const std::function<const Tensor<1, n_components, VectorizedArray<double>>(
        const unsigned int cell,
        const unsigned int q)> &                              cell_operation)
    {
      FE_DGQArbitraryNodes<1> fe_coarse(QGauss<1>(n_q_points_1D).get_points());
      FE_Q<1>                 fe_fine(fe_degree);

      /// create 1D projection matrix for sum factorization
      FullMatrix<double> matrix(fe_fine.dofs_per_cell, fe_coarse.dofs_per_cell);
      FETools::get_projection_matrix(fe_coarse, fe_fine, matrix);

      AlignedVector<VectorizedArray<double>> projection_matrix_1d(fe_fine.dofs_per_cell *
                                                                  fe_coarse.dofs_per_cell);

      for (unsigned int i = 0, k = 0; i < fe_coarse.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe_fine.dofs_per_cell; ++j, ++k)
          projection_matrix_1d[k] = matrix(j, i);

      FECellIntegrator<dim, n_components, double> fe_eval(matrix_free, dof_idx, quad_idx);

      // @todo: replace n_q_points_1D argument completely from fe_eval?
      AssertThrow(fe_eval.n_q_points == std::pow(n_q_points_1D, dim),
                  ExcMessage("The number of quadrature points in 1D must comply with the number of "
                             "quadrature points of the cell operation."));

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          fe_eval.reinit(cell);

          for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
            {
              const auto temp = cell_operation(cell, q);
              for (int c = 0; c < n_components; ++c)
                fe_eval.begin_values()[c * fe_eval.n_q_points + q] = temp[c];
            }

          // perform basis change from quadrature points to support points
          internal::FEEvaluationImplBasisChange<
            internal::evaluate_general,
            internal::EvaluatorQuantity::value,
            dim,
            0,
            0,
            VectorizedArray<double>,
            VectorizedArray<double>>::do_forward(n_components, // n_components
                                                 projection_matrix_1d,
                                                 fe_eval.begin_values(),
                                                 fe_eval.begin_dof_values(),
                                                 n_q_points_1D, // number of quadrature points
                                                 fe_degree + 1  // number of support points
          );

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
    VectorizedArray<number>
    heaviside(const VectorizedArray<number> &in, const number limit = 0.0)
    {
      return compare_and_apply_mask<SIMDComparison::greater_than>(in,
                                                                  VectorizedArray<double>(limit),
                                                                  1.0,
                                                                  0.0);
    }

    /**
     * For two indicator vectors, representing e.g. implicit geometries, this function computes a
     * boolean operation and returns the resulting vector. The user has to take care on distributing
     * relevant constraints afterwards.
     */
    template <typename VectorType>
    VectorType
    merge_two_indicator_fields(const VectorType &indicator_1,
                               const VectorType &indicator_2,
                               BooleanType       type                     = BooleanType::Union,
                               const double      indicator_value_interior = 1.0,
                               const double      indicator_value_exterior = -1.0)
    {
      AssertThrow(indicator_1.size() == indicator_2.size(),
                  ExcMessage("The two level set vectors to be merged must be of equal length."));

      AssertThrow(indicator_value_interior != indicator_value_exterior,
                  ExcMessage("The indicator value in the interior of the implicit geometry must be "
                             "different than the one in the exterior. Make sure that the function "
                             "arguments indicator_value_interior and indicator_value_exterior are "
                             "set correctly."));

      AssertThrow(indicator_1.linfty_norm() ==
                    std::max(indicator_value_exterior, indicator_value_interior),
                  ExcMessage(
                    "The maximum indicator value does not correspond to the given bounds "
                    "of indicator_value_interior and indicator_value_exterior. Make sure that "
                    "the indicator_value_interior and indicator_value_exterior comply with the "
                    "indicator field."));

      VectorType merge_indicator(indicator_1);

      bool is_interior_larger_than_exterior = indicator_value_interior > indicator_value_exterior;

      for (const auto &i : indicator_1.locally_owned_elements())
        switch (type)
          {
            case BooleanType::Union:
              merge_indicator[i] = (is_interior_larger_than_exterior) ?
                                     std::max(indicator_1[i], indicator_2[i]) :
                                     std::min(indicator_1[i], indicator_2[i]);
              break;
            case BooleanType::Intersection:
              merge_indicator[i] = (is_interior_larger_than_exterior) ?
                                     std::min(indicator_1[i], indicator_2[i]) :
                                     std::max(indicator_1[i], indicator_2[i]);
              break;
            case BooleanType::Subtraction:
              merge_indicator[i] =
                (indicator_1[i] == indicator_2[i] && indicator_1[i] == indicator_value_interior) ?
                  indicator_value_exterior :
                  indicator_1[i];
              break;
            default:
              AssertThrow(false, ExcNotImplemented());
          }

      merge_indicator.compress(VectorOperation::insert);

      return merge_indicator;
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

    template <typename value_type>
    inline value_type
    interpolate(const value_type &ls, const double val1, const double val2)
    {
      return (1. - ls) * val1 + ls * val2;
    }

    template <typename value_type>
    inline value_type
    interpolate_cubic(const value_type &ls, const double val1, const double val2)
    {
      return val1 + (val2 - val1) * (-2. * ls * ls * ls + 3. * ls * ls);
    }

    template <typename value_type>
    inline value_type
    interpolate_cubic_derivative(const value_type &ls, const double val1, const double val2)
    {
      return (val2 - val1) * (-6. * ls * ls + 6. * ls);
    }

    template <typename number>
    VectorizedArray<number>
    limit_to_bounds(const VectorizedArray<number> &in,
                    const number                   lower_limit,
                    const number                   upper_limit)
    {
      const auto ub =
        compare_and_apply_mask<SIMDComparison::greater_than>(in, upper_limit, upper_limit, in);
      return compare_and_apply_mask<SIMDComparison::less_than>(ub, lower_limit, lower_limit, ub);
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
  } // namespace UtilityFunctions
} // namespace MeltPoolDG
