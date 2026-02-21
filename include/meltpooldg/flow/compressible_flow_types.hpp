#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <type_traits>

namespace MeltPoolDG::CompressibleFlow
{
  /// Number of independent conserved variables for the compressible Navier-Stokes equations in
  /// `dim` dimensions.
  template <int dim>
  constexpr unsigned int conserved_variables_components = dim + 2;

  /// Compile-time indices for accessing individual conserved variables within a `dealii::Tensor`
  /// storing the state of the compressible flow solver used throughout the codebase.
  template <int dim>
  struct ConservedVariableIndex
  {
    constexpr static unsigned int density  = 0;
    constexpr static unsigned int momentum = 1;
    constexpr static unsigned int energy   = dim + 1;
  };

  /// Type alias for the conserved variables in the compressible flow solver given at a vectorized
  /// set of coordinates.
  template <int dim, typename number>
  using ConservedVariablesType =
    dealii::Tensor<1, conserved_variables_components<dim>, dealii::VectorizedArray<number>>;

  /// Type alias for the gradient of the conserved variables in the compressible flow solver given
  /// at a vectorized set of coordinates.
  template <int dim, typename number>
  using ConservedVariablesGradientType =
    dealii::Tensor<1,
                   conserved_variables_components<dim>,
                   dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;

  /// Type alias for the fluxes in the compressible flow solver given at a vectorized set of
  /// coordinates. This includes both convective and diffusive fluxes.
  template <int dim, typename number>
  using FluxType = dealii::Tensor<1,
                                  conserved_variables_components<dim>,
                                  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;

  /// Type alias for the fluxes at faces in the compressible flow solver given at a vectorized set
  /// of coordinates (contracted with normal vector). This includes both convective and diffusive
  /// fluxes.
  template <int dim, typename number>
  using FaceFluxType =
    dealii::Tensor<1, conserved_variables_components<dim>, dealii::VectorizedArray<number>>;

  /// Type alias for source terms in the compressible flow solver given at a vectorized set of
  /// coordinates.
  template <int dim, typename number>
  using SourceType =
    dealii::Tensor<1, conserved_variables_components<dim>, dealii::VectorizedArray<number>>;

  /// Concept ensuring compatibility of a type with the conserved variables of the compressible
  /// flow solver in `dim` dimensions.
  ///
  /// @note If this concept is used in the code base it is also assumed that the data stored in
  /// the type is indexed according to `ConservedVariableIndex<dim>`. However, it is not possible to
  /// enforce this assumption in the concept definition itself.
  template <typename T, int dim>
  concept IsConservedStateCompatible = requires {
    is_dealii_tensor<std::remove_cvref_t<T>>::value;
    T::rank == 1;
    T::dimension >= conserved_variables_components<dim>;
  };

  /// Concept ensuring compatibility of a type with the gradient of the conserved variables of the
  /// compressible flow solver in `dim` dimensions.
  ///
  /// @note If this concept is used in the code base it is also assumed that the data stored in
  /// the type is indexed according to `ConservedVariableIndex<dim>`. However, it is not possible to
  /// enforce this assumption in the concept definition itself.
  template <typename T, int dim>
  concept IsConservedGradientCompatible = requires {
    is_dealii_tensor<std::remove_cvref_t<T>>::value;
    T::rank == 1;
    T::dimension >= conserved_variables_components<dim>;
    T::value_type::rank == 1;
    T::value_type::dimension == dim;
  };
} // namespace MeltPoolDG::CompressibleFlow