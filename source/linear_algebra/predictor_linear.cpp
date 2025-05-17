#include <meltpooldg/linear_algebra/predictor_linear.hpp>
//
#include <deal.II/base/config.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <type_traits>

namespace MeltPoolDG
{
  // note: the content of this namespace will be part of deal.II
  namespace internal
  {
    template <
      typename VectorType,
      std::enable_if_t<not dealii::IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    unsigned int
    n_blocks(const VectorType &)
    {
      return 1;
    }

    template <typename VectorType,
              std::enable_if_t<dealii::IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    unsigned int
    n_blocks(const VectorType &vector)
    {
      return vector.n_blocks();
    }

    template <
      typename VectorType,
      std::enable_if_t<not dealii::IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    VectorType &
    block(VectorType &vector, const unsigned int b)
    {
      AssertDimension(b, 0);
      (void)b;
      return vector;
    }

    template <typename VectorType,
              std::enable_if_t<dealii::IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    typename VectorType::BlockType &
    block(VectorType &vector, const unsigned int b)
    {
      return vector.block(b);
    }

    template <
      typename VectorType,
      std::enable_if_t<not dealii::IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    const VectorType &
    block(const VectorType &vector, const unsigned int b)
    {
      AssertDimension(b, 0);
      (void)b;
      return vector;
    }

    template <typename VectorType,
              std::enable_if_t<dealii::IsBlockVector<VectorType>::value, VectorType> * = nullptr>
    const typename VectorType::BlockType &
    block(const VectorType &vector, const unsigned int b)
    {
      return vector.block(b);
    }
  } // namespace internal

  template <typename VectorType, typename number>
  void
  compute_linear_predictor(const VectorType &old_vec,
                           const VectorType &old_old_vec,
                           VectorType       &predictor,
                           const number      current_time_increment,
                           const number      old_time_increment)
  {
    if (std::abs(old_time_increment) < 1e-12)
      {
        predictor = old_vec;
        return;
      }

    const number fraction = current_time_increment / old_time_increment;

    for (unsigned int c = 0; c < internal::n_blocks(predictor); ++c)
      DEAL_II_OPENMP_SIMD_PRAGMA
    for (unsigned int i = 0; i < internal::block(predictor, c).locally_owned_size(); ++i)
      internal::block(predictor, c).local_element(i) =
        (fraction + 1.0) * internal::block(old_vec, c).local_element(i) -
        fraction * internal::block(old_old_vec, c).local_element(i);
  }

  template void
  compute_linear_predictor(const dealii::LinearAlgebra::distributed::Vector<double> &,
                           const dealii::LinearAlgebra::distributed::Vector<double> &,
                           dealii::LinearAlgebra::distributed::Vector<double> &,
                           const double,
                           const double);
  template void
  compute_linear_predictor(const dealii::LinearAlgebra::distributed::BlockVector<double> &,
                           const dealii::LinearAlgebra::distributed::BlockVector<double> &,
                           dealii::LinearAlgebra::distributed::BlockVector<double> &,
                           const double,
                           const double);
} // namespace MeltPoolDG
