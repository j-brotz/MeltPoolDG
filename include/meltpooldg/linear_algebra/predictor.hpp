/* ---------------------------------------------------------------------
 *
 * Author: Martin Kronbichler, Peter Munch, Magdalena Schreter, UA/TUM
 *         October 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <typename VectorType>
  class LeastSquaresProjection
  {
  public:
    LeastSquaresProjection(const std::vector<VectorType> &solution_history,
                           const unsigned int n_old_solutions = numbers::invalid_unsigned_int)
      : old_solution(n_old_solutions == numbers::invalid_unsigned_int ? solution_history.size() :
                                                                        n_old_solutions)
    {
      Assert(n_old_solutions <= solution_history.size(), ExcInternalError());

      for (unsigned int i = 0; i < old_solution.size(); ++i)
        old_solution[i] = &solution_history[i];
    }

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
    std::vector<const VectorType *> old_solution;

    std::vector<VectorType> internal_vectors;
  };

  template <typename VectorType, typename number = double>
  class Predictor
  {
  private:
    const PredictorData<number>                   data;
    TimeIntegration::SolutionHistory<VectorType> &solution_history;
    const TimeIterator<number> *                  time_iterator;
    const LeastSquaresProjection<VectorType>      lsp;

  public:
    Predictor(const PredictorData<number>                   data,
              TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
              const TimeIterator<number> *                  time_iterator = nullptr)
      : data(data)
      , solution_history(solution_history_in)
      , time_iterator(time_iterator)
      , lsp(solution_history_in.get_all_solutions(), data.n_old_solution_vectors)
    {
      Assert(solution_history.size() >= 1, ExcInternalError());
      Assert(solution_history.size() >= 2 || data.type != PredictorType::linear_extrapolation,
             ExcInternalError());
    }

    template <typename MatrixType>
    void
    vmult(const MatrixType &matrix, VectorType &solution, const VectorType &rhs) const
    {
      if (data.type == PredictorType::none)
        {
          solution = solution_history.get_current_solution();
        }
      else if (data.type == PredictorType::linear_extrapolation)
        {
          UtilityFunctions::compute_linear_predictor(
            solution_history.get_current_solution(),
            solution_history.get_recent_old_solution(),
            solution,
            time_iterator ? time_iterator->get_current_time_increment() : 1.0,
            time_iterator ? time_iterator->get_old_time_increment() : 1.0);
        }
      else if (data.type == PredictorType::least_squares_projection)
        {
          lsp.vmult(matrix, solution, rhs);
        }
      else
        AssertThrow(false, ExcNotImplemented());

      // TODO: use SolutionHistory functionality;
      solution_history.get_recent_old_solution().swap(solution_history.get_current_solution());
      solution_history.get_current_solution().swap(solution);

      return;
    }
  };
} // namespace MeltPoolDG
