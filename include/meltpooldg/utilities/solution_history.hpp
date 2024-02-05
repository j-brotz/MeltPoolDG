#pragma once

/**
 * Adopted from
 *
 * https://github.com/peterrum/pf-applications
 *
 * Authors: Peter Munch, Vladimir Ivannikov
 */

#include <deal.II/lac/generic_linear_algebra.h>

#include <functional>

namespace TimeIntegration
{
  using namespace dealii;

  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  template <typename VectorType>
  class SolutionHistory
  {
  public:
    SolutionHistory(unsigned int size)
      : solutions(size)
    {}

    void
    apply(std::function<void(VectorType &)> f)
    {
      for (unsigned int i = 0; i < solutions.size(); ++i)
        f(solutions[i]);
    }

    void
    apply_old(std::function<void(VectorType &)> f)
    {
      Assert(solutions.size() > 1, ExcInternalError());

      for (unsigned int i = 1; i < solutions.size(); ++i)
        f(solutions[i]);
    }


    void
    update_ghost_values(const bool check = false) const
    {
      for (unsigned int i = 0; i < solutions.size(); ++i)
        if (!check || !solutions[i].has_ghost_elements())
          solutions[i].update_ghost_values();
    }

    void
    zero_out_ghost_values(const bool check = false) const
    {
      for (unsigned int i = 0; i < solutions.size(); ++i)
        if (!check || solutions[i].has_ghost_elements())
          solutions[i].zero_out_ghost_values();
    }

    void
    commit_old_solutions()
    {
      //@note we don't use swap for now since in the HeatOperation
      // solutions[0] is initialized with a different DoFHandler
      // than solutions[1]
      for (int i = solutions.size() - 1; i >= 1; --i)
        solutions[i] = solutions[i - 1];
    }

    void
    set_recent_old_solution(const VectorType &src)
    {
      Assert(solutions.size() > 1, ExcInternalError());

      solutions[1] = src;
    }

    const VectorType &
    get_recent_old_solution() const
    {
      Assert(solutions.size() >= 1, ExcInternalError());
      return solutions[1];
    }

    VectorType &
    get_recent_old_solution()
    {
      Assert(solutions.size() >= 1, ExcInternalError());
      return solutions[1];
    }

    const VectorType &
    get_current_solution() const
    {
      return solutions[0];
    }

    VectorType &
    get_current_solution()
    {
      return solutions[0];
    }

    const std::vector<std::shared_ptr<VectorType>> &
    get_old_solutions() const
    {
      return std::vector<std::shared_ptr<VectorType>>(solutions.begin() + 1, solutions.end());
    }

    const std::vector<VectorType> &
    get_all_solutions() const
    {
      return solutions;
    }

    unsigned int
    size() const
    {
      return solutions.size();
    }

  private:
    std::vector<VectorType> solutions;
  };
} // namespace TimeIntegration

///////////////////////////////////////////////7
// KEPT AS TEMPLATES
//
// TODO
// SolutionHistory(std::vector<std::shared_ptr<VectorType>> solutions)
//: solutions(solutions)
//{}
// SolutionHistory<VectorType>
// filter(const bool keep_current = true,
// const bool keep_recent  = true,
// const bool keep_old     = true) const
//{
// std::vector<std::shared_ptr<VectorType>> subset;

// for (unsigned int i = 0; i < solutions.size(); ++i)
// if (can_process(i, keep_current, keep_recent, keep_old))
// subset.push_back(solutions[i]);

// return SolutionHistory(subset);
//}

// TODO ?
// virtual std::size_t
// memory_consumption() const
//{
// return MyMemoryConsumption::memory_consumption(solutions);
//}

// void
// extrapolate(VectorType &dst, const double factor) const
//{
// Assert(solutions.size() >= 1, ExcInternalError());

// dst = solutions[0];
// dst.add(-1.0, solutions[1]);
// dst.sadd(factor, solutions[0]);
//}
// bool
// can_process(const unsigned int index,
// const bool         keep_current,
// const bool         keep_recent,
// const bool         keep_old) const
//{
// return (index == 0 && keep_current) || (index == 1 && keep_recent) || (index > 1 && keep_old);
//}


// void
// apply_blockwise(std::function<void(BlockVectorType &)> f) const
//{
// for (unsigned int i = 0; i < solutions.size(); ++i)
// for (unsigned int b = 0; b < solutions[i]->n_blocks(); ++b)
// f(solutions[i]->block(b));
//}

// unsigned int
// n_blocks_total() const
//{
// unsigned int n_blocks = 0;
// for (const auto &sol : solutions)
// n_blocks += sol->n_blocks();

// return n_blocks;
//}
// void
// flatten(VectorType &dst) const
//{
// unsigned int b_dst = 0;

// for (unsigned int i = 0; i < solutions.size(); ++i)
// for (unsigned int b = 0; b < solutions[i]->n_blocks(); ++b)
//{
// dst.block(b_dst).copy_locally_owned_data_from(solutions[i]->block(b));
//++b_dst;
//}
//}

// std::vector<BlockVectorType *>
// get_all_blocks() const
//{
// std::vector<BlockVectorType *> solution_ptr;

// for (unsigned int i = 0; i < solutions.size(); ++i)
// for (unsigned int b = 0; b < solutions[i]->n_blocks(); ++b)
// solution_ptr.push_back(&solutions[i]->block(b));

// return solution_ptr;
//}
