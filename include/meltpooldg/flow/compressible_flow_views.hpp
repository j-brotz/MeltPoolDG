#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/compressible_flow_eos_utils_base.hpp>
#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>
#include <meltpooldg/flow/compressible_flow_views_mixins.hpp>

/**
 * @file
 *
 * This file provides a collection of view types for the compressible  Navier–Stokes solver. The
 * views offer convenient and semantically meaningful access to:
 * - Conserved variables,
 * - Their gradients,
 * - Derived primitive and thermodynamic quantities, and
 * - Associated convective and diffusive fluxes.
 *
 * The views are implemented as CRTP mixins, allowing flexible composition and extension for
 * different solver components while avoiding runtime overhead.
 *
 * In addition to direct access to conserved quantities, the views integrate the evaluation of
 * primitive variables and thermodynamic properties via a provided equation of state. If desired,
 * they also expose material properties such as dynamic viscosity and thermal conductivity.
 *
 * The overall design goal is to provide a clear and expressive interface for flux computations and
 * related operations, while fully abstracting the underlying data layout and storage details.
 *
 * ## Constness semantics
 * The views provided here follow the philosophy that view constness does not imply data constness.
 * A `const`-qualified view prevents modification of the view object itself, but not necessarily of
 * the referenced data. Data mutability is determined solely by the template parameter:
 * - `View<T>` and `const View<T>` allow modification of the underlying data,
 * - `View<const T>` provides read-only access to the underlying data.
 */
namespace MeltPoolDG::CompressibleFlow
{
  /**
   * View providing access to the conserved variables stored in the underlying data structure.
   * Besides direct access to the conserved variables, it enables computation of directly derived
   * quantities (e.g., velocity or specific total energy) via the corresponding mixin.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, IsConservedStateCompatible<dim> StateType>
  struct DofValueView
    : public StateMixin<dim, typename StateType::value_type, DofValueView<dim, StateType>>
  {
    DofValueView(StateType &state)
      : flow_state(&state)
    {}

    operator DofValueView<dim, const StateType>()
    {
      return DofValueView<dim, const StateType>(*flow_state);
    }

    StateType &
    value() const
    {
      return *flow_state;
    }

  private:
    mutable StateType *flow_state;
  };

  /**
   * View providing access to the conserved variables stored in the underlying data structure.
   * Besides direct access to the conserved variables, it enables computation of primitive
   * variables, thermodynamic quantities, and material properties via the provided EOS and material
   * data.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, typename number, IsConservedStateCompatible<dim> StateType>
  struct DofValueEvaluatorView : public StateMixin<dim,
                                                   typename StateType::value_type,
                                                   DofValueEvaluatorView<dim, number, StateType>>,
                                 public EOSMixin<dim,
                                                 typename StateType::value_type,
                                                 DofValueEvaluatorView<dim, number, StateType>>,
                                 public MaterialMixin<DofValueEvaluatorView<dim, number, StateType>>
  {
    DofValueEvaluatorView(StateType                                              &state,
                          const Flow::EOS::EquationOfStateUtils<dim, number>     *eos,
                          const Flow::CompressibleFluidMaterialPhaseData<number> &material_data)
      : flow_state(&state)
      , eos_evaluator(eos)
      , material_data(material_data)
    {}

    operator DofValueEvaluatorView<dim, number, const StateType>()
    {
      return DofValueEvaluatorView<dim, number, const StateType>(*flow_state,
                                                                 eos_evaluator,
                                                                 material_data);
    }

    StateType &
    value() const
    {
      return *flow_state;
    }

    const Flow::EOS::EquationOfStateUtils<dim, number> *
    eos() const
    {
      return eos_evaluator;
    }

    const Flow::CompressibleFluidMaterialPhaseData<number> &
    material() const
    {
      return material_data;
    }

  private:
    mutable StateType                                      *flow_state;
    const Flow::EOS::EquationOfStateUtils<dim, number>     *eos_evaluator;
    const Flow::CompressibleFluidMaterialPhaseData<number> &material_data;
  };

  /**
   * View providing access to the gradients of the conserved variables stored in the underlying data
   * structure. Besides direct access to the conserved variables, it enables computation of directly
   * derived quantities (e.g., gradient of the velocity) via the corresponding mixin.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the gradients of the conserved variables.
   */
  template <int dim, IsConservedGradientCompatible<dim> StateType>
  struct DofGradientView
    : public StateGradientMixin<
        dim,
        dealii::Tensor<1, StateType::dimension, typename StateType::value_type::value_type>,
        DofGradientView<dim, StateType>>
  {
    DofGradientView(StateType &state)
      : flow_state(&state)
    {}

    operator DofGradientView<dim, const StateType>()
    {
      return DofGradientView<dim, const StateType>(*flow_state);
    }

    StateType &
    gradient_value() const
    {
      return *flow_state;
    }

  private:
    mutable StateType *flow_state;
  };

  /**
   * View providing access to the gradients of conserved variables stored in the underlying data
   * structure. Besides direct access to the conserved variables, it enables computation of
   * gradients of primitive variables (e.g. gradient of the temperature) assuming the provided
   * equation of state does support this. Further one gets access to material properties via the
   * provided material data.
   *
   * The underlying `StateType` must store the conserved variables in a tensor-like container
   * indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam StateType  Type of the data structure storing the conserved variables.
   */
  template <int dim, typename number, IsConservedGradientCompatible<dim> StateType>
  struct DofGradientEvaluator
    : public StateGradientMixin<dim,
                                typename StateType::value_type::value_type,
                                DofGradientEvaluator<dim, number, StateType>>,
      public EOSGradientMixin<
        dim,
        dealii::Tensor<1, StateType::dimension, typename StateType::value_type::value_type>,
        DofGradientEvaluator<dim, number, StateType>>,
      public MaterialMixin<DofGradientEvaluator<dim, number, StateType>>
  {
    using gradient_type = std::remove_cvref_t<StateType>;

    DofGradientEvaluator(StateType                                              &state,
                         const Flow::EOS::EquationOfStateUtils<dim, number>     *eos,
                         const Flow::CompressibleFluidMaterialPhaseData<number> &material_data)
      : flow_state(&state)
      , eos_evaluator(eos)
      , material_data(material_data)
    {}

    operator DofGradientEvaluator<dim, number, const StateType>()
    {
      return DofGradientEvaluator<dim, number, const StateType>(*flow_state,
                                                                eos_evaluator,
                                                                material_data);
    }

    StateType &
    gradient_value() const
    {
      return *flow_state;
    }

    const Flow::EOS::EquationOfStateUtils<dim, number> *
    eos() const
    {
      return eos_evaluator;
    }

    const Flow::CompressibleFluidMaterialPhaseData<number> &
    material() const
    {
      return material_data;
    }


  private:
    mutable StateType                                      *flow_state;
    const Flow::EOS::EquationOfStateUtils<dim, number>     *eos_evaluator;
    const Flow::CompressibleFluidMaterialPhaseData<number> &material_data;
  };

  /**
   * View providing access to the conserved variables and their gradients stored in the underlying
   * data structure.
   *
   * Besides direct access to the conserved quantities and their gradients, the view enables
   * computation of primitive variables, thermodynamic quantities, and material properties via the
   * provided equation of state and material data. If supported by the equation of state, gradients
   * of primitive variables (e.g., the temperature gradient) can also be derived.
   *
   * The underlying `ValueState` and `GradientState` must store the conserved variables in a
   * tensor-like container indexed according to `TensorStorageIndex<dim>`.
   *
   * @tparam ValueState  Type of the data structure storing the conserved variables.
   * @tparam GradientState  Type of the data structure storing the gradients of conserved variables.
   */
  template <int dim,
            typename number,
            IsConservedStateCompatible<dim>    Value,
            IsConservedGradientCompatible<dim> Gradient>
  struct DofEvaluator
    : public StateMixin<dim,
                        typename Value::value_type,
                        DofEvaluator<dim, number, Value, Gradient>>,
      public StateGradientMixin<dim,
                                typename Gradient::value_type::value_type,
                                DofEvaluator<dim, number, Value, Gradient>>,
      public EOSMixin<dim, typename Value::value_type, DofEvaluator<dim, number, Value, Gradient>>,
      public EOSGradientMixin<dim, Value, DofEvaluator<dim, number, Value, Gradient>>,
      public MaterialMixin<DofEvaluator<dim, number, Value, Gradient>>
  {
    using state_type    = std::remove_cvref_t<Value>;
    using gradient_type = std::remove_cvref_t<Gradient>;

    DofEvaluator(Value                                                  &value_state,
                 Gradient                                               &gradient_state,
                 const Flow::EOS::EquationOfStateUtils<dim, number>     *eos,
                 const Flow::CompressibleFluidMaterialPhaseData<number> &material_data)
      : flow_state(&value_state)
      , flow_gradient_state(&gradient_state)
      , eos_evaluator(eos)
      , material_data(material_data)
    {}

    operator DofEvaluator<dim, number, const Value, const Gradient>()
    {
      return DofEvaluator<dim, number, const Value, const Gradient>(*flow_state,
                                                                    *flow_gradient_state,
                                                                    eos_evaluator,
                                                                    material_data);
    }

    Value &
    value() const
    {
      return *flow_state;
    }

    Gradient &
    gradient_value() const
    {
      return *flow_gradient_state;
    }

    const Flow::EOS::EquationOfStateUtils<dim, number> *
    eos() const
    {
      return eos_evaluator;
    }

    const Flow::CompressibleFluidMaterialPhaseData<number> &
    material() const
    {
      return material_data;
    }

  private:
    mutable Value                                          *flow_state;
    mutable Gradient                                       *flow_gradient_state;
    const Flow::EOS::EquationOfStateUtils<dim, number>     *eos_evaluator;
    const Flow::CompressibleFluidMaterialPhaseData<number> &material_data;
  };

  /**
   * View providing access to the fluxes stored in the underlying data structure. Besides direct
   * access to the fluxes, no further functionality is provided.
   *
   * The underlying `FluxType` must store the fluxes in a tensor-like container indexed according to
   * `TensorStorageIndex<dim>`.
   *
   * @tparam FluxType  Type of the data structure storing the fluxes.
   */
  template <int dim, typename FluxType>
  struct FluxView : public FluxMixin<dim, FluxView<dim, FluxType>>
  {
    explicit FluxView(FluxType &flux_in)
      : flux(&flux_in)
    {}

    operator FluxView<dim, const FluxType>()
    {
      return FluxView<dim, const FluxType>(*flux);
    }

    FluxType &
    value() const
    {
      return *flux;
    }

  private:
    mutable FluxType *flux;
  };
} // namespace MeltPoolDG::CompressibleFlow
