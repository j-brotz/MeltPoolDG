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
      const unsigned int n_old_solutions = old_solution.size();
      internal_vectors.resize(n_old_solutions);
      FullMatrix<double> projection_matrix(n_old_solutions, n_old_solutions);
      unsigned int       step = 0;
      for (; step < n_old_solutions; ++step)
        {
          internal_vectors[step].reinit(rhs, true);
          matrix.vmult(internal_vectors[step], *old_solution[step]);
          // modified Gram-Schmidt
          projection_matrix(0, step) = internal_vectors[step] * internal_vectors[0];
          for (unsigned int j = 0; j < step; ++j)
            {
              projection_matrix(j + 1, step) = internal_vectors[step].add_and_dot(
                -projection_matrix(j, step) /
                  projection_matrix(j, j), // TODO: check for division by zero?
                internal_vectors[j],
                internal_vectors[j + 1]);
            }

          if (projection_matrix(step, step) < 1e-16 * projection_matrix(0, 0))
            break;
        }

      // Solve least-squares system
      std::vector<double> project_sol(step);
      for (unsigned int s = 0; s < step; ++s)
        project_sol[s] = internal_vectors[s] * rhs;

      for (int s = step - 1; s >= 0; --s)
        {
          double sum = project_sol[s];
          for (unsigned int j = s + 1; j < step; ++j)
            sum -= project_sol[j] * projection_matrix(s, j);
          project_sol[s] = sum / projection_matrix(s, s); // TODO check for division by zero?
        }

      // extrapolate solution from old values
      solution.reinit(rhs, true);
      solution.equ(project_sol[0], *old_solution[0]);
      for (unsigned int s = 1; s < step; ++s)
        solution.add(project_sol[s], *old_solution[s]);
    }

  private:
    std::vector<const VectorType *> old_solution;

    mutable std::vector<VectorType> internal_vectors;
  };

  template <typename VectorType, typename number = double>
  class Predictor
  {
  private:
    const PredictorData<number>                   data;
    TimeIntegration::SolutionHistory<VectorType> &solution_history;
    const TimeIterator<number>                   *time_iterator;
    const LeastSquaresProjection<VectorType>      lsp;

    mutable unsigned int n_calls = 0;

  public:
    Predictor(const PredictorData<number>                   data,
              TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
              const TimeIterator<number>                   *time_iterator = nullptr)
      : data(data)
      , solution_history(solution_history_in)
      , time_iterator(time_iterator)
      , lsp(solution_history_in.get_all_solutions(), data.n_old_solution_vectors)
    {
      Assert(solution_history.size() >= 1, ExcInternalError());
      Assert(solution_history.size() >= 2 || data.type != PredictorType::linear_extrapolation,
             ExcInternalError());

      // TODO: extend for BlockVectors
      AssertThrow(!internal::is_block_vector<VectorType> ||
                    data.type != PredictorType::least_squares_projection,
                  ExcNotImplemented());
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
          // use LSP only if n_old_solution_vectors are stored
          if (n_calls >= data.n_old_solution_vectors)
            lsp.vmult(matrix, solution, rhs);
          else
            // use const predictor
            solution.swap(solution_history.get_current_solution());
        }
      else
        AssertThrow(false, ExcNotImplemented());

      solution_history.commit_old_solutions();
      solution_history.get_current_solution().swap(solution);

      n_calls += 1;
      return;
    }
  };
} // namespace MeltPoolDG
