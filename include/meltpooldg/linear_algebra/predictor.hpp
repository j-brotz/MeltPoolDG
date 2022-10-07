/* ---------------------------------------------------------------------
 *
 * Author: Martin Kronbichler, Peter Munch, Magdalena Schreter,  TUM, October 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/utilities/time_iterator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <typename VectorType>
  class LeastSquaresProjection
  {
  public:
    LeastSquaresProjection(const std::vector<VectorType> &old_solution)
      : old_solution(old_solution)
    {}

    template <typename MatrixType>
    void
    vmult(const MatrixType &matrix, VectorType &solution, const VectorType &rhs) const
    {
      // TODO
      (void)matrix;
      (void)solution;
      (void)rhs;
      AssertThrow(false, ExcNotImplemented());
    }

  private:
    const std::vector<VectorType> &old_solution;

    std::vector<VectorType> internal_vectors;
  };

  template <typename VectorType, typename number = double>
  class Predictor
  {
  private:
    const PredictorType                      type;
    const std::vector<VectorType> &          solution_history;
    const TimeIterator<number> *             time_iterator;
    const LeastSquaresProjection<VectorType> lsp;

  public:
    Predictor(const PredictorType            type,
              const std::vector<VectorType> &solution_history,
              const TimeIterator<number> *   time_iterator = nullptr)
      : type(type)
      , solution_history(solution_history)
      , time_iterator(time_iterator)
      , lsp(solution_history)
    {
      Assert(solution_history.size() >= 1, ExcInternalError());
      Assert(solution_history.size() >= 2 || !(type == PredictorType::linear_extrapolation),
             ExcInternalError());
    }

    template <typename MatrixType>
    void
    vmult(const MatrixType &matrix, VectorType &solution, const VectorType &rhs) const
    {
      if (type == PredictorType::none)
        {
          solution = solution_history[0];
        }
      else if (type == PredictorType::linear_extrapolation)
        {
          AssertThrow(solution_history.size() >= 2,
                      ExcMessage("You must at least provide two old solution vectors."));

          UtilityFunctions::compute_linear_predictor(
            solution_history[0],
            solution_history[1],
            solution,
            time_iterator ? time_iterator->get_current_time_increment() : 1.0,
            time_iterator ? time_iterator->get_old_time_increment() : 1.0);
        }
      else if (type == PredictorType::least_squares_projection)
        {
          lsp.vmult(matrix, solution, rhs);
        }
      else
        AssertThrow(false, ExcNotImplemented());

      return;
    }
  };
} // namespace MeltPoolDG
