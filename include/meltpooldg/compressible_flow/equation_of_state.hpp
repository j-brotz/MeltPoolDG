#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/utilities/concepts.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>

#include <cmath>
#include <utility>

namespace MeltPoolDG::Flow
{
  /**
   * Concept defining the specific requirements for a value view to be used with any equation of
   * state.
   */
  template <typename T>
  concept EOSIsValueView = requires(const T v)
  {
    {v.density()};
    {v.velocity()};
    {v.total_energy()};
  };

  /**
   * Concept defining the specific requirements for a gradient view to be used with any equation of
   * state.
   */
  template <typename T>
  concept EOSIsGradientView = requires(const T g)
  {
    {g.grad_density()};
    {g.grad_velocity()};
    {g.grad_total_energy()};
  };

  /**
   * Concept defining the specific requirements for a material view to be used with the ideal gas
   * equation of state.
   */
  template <typename T>
  concept IdealGasIsMaterialView = requires(const T m)
  {
    {m.heat_capacity_ratio()};
    {m.specific_gas_constant()};
  };

  struct IdealGasEOS
  {
    /**
     * Compute the thermodynamic pressure for an ideal gas from the given flow state.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Pressure resulting from the given flow state and material properties.
     */
    template <EOSIsValueView ValueView, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      thermodynamic_pressure(const ValueView &value_view, const MaterialView &material_view)
    {
      return (material_view.heat_capacity_ratio() - 1.) *
             (value_view.total_energy() -
              value_view.density() * 0.5 *
                scalar_product(value_view.velocity(), value_view.velocity()));
    }

    /**
     * Compute the gradient of the temperature for an ideal gas from the given flow state and
     * material properties.
     *
     * @param value_view View providing access to the flow state.
     * @param gradient_view View providing access to the gradients of the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Gradient of the temperature resulting from the given flow state and material properties.
     */
    template <EOSIsValueView         ValueView,
              EOSIsGradientView      GradientView,
              IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      grad_temperature(const ValueView    &value_view,
                       const GradientView &gradient_view,
                       const MaterialView &material_view)
    {
      const auto grad_E =
        1. / value_view.density() *
        (gradient_view.grad_total_energy() -
         1. / value_view.density() * value_view.total_energy() * gradient_view.grad_density());

      return (material_view.heat_capacity_ratio() - 1.0) / material_view.specific_gas_constant() *
             (grad_E - matrix_vector_product(gradient_view.grad_velocity(), value_view.velocity()));
    }

    /**
     * Compute the speed of sound for an ideal gas from the given flow state and material
     * properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Speed of sound resulting from the given flow state and material properties.
     */
    template <EOSIsValueView ValueView, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      speed_of_sound(const ValueView &value_view, const MaterialView &material_view)
    {
      return std::sqrt(material_view.heat_capacity_ratio() *
                       thermodynamic_pressure(value_view, material_view) / value_view.density());
    }

    /**
     * Compute the temperature for an ideal gas from the given flow state and material properties.
     *
     * @param value_view View providing access to the flow state.
     * @param material_view View providing access to the material properties.
     *
     * @return Temperature resulting from the given flow state and material properties.
     */
    template <EOSIsValueView ValueView, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      temperature(const ValueView &value_view, const MaterialView &material_view)
    {
      return thermodynamic_pressure(value_view, material_view) /
             (material_view.specific_gas_constant() * value_view.density());
    }

    /**
     * Compute the inner energy from a given pressure for an ideal gas with the given material
     * properties.
     *
     * @param pressure Pressure for which the inner energy should be computed.
     * @param material_view View providing access to the material properties.
     *
     * @return Inner energy resulting from the given pressure and material properties.
     */
    template <typename ValueType, IdealGasIsMaterialView MaterialView>
    static inline DEAL_II_ALWAYS_INLINE //
      auto
      inner_energy_from_pressure(const ValueType &pressure, const MaterialView &material_view)
    {
      return pressure / (material_view.heat_capacity_ratio() - 1.);
    }
  };

  /**
   * Dispatches a functional to a concrete Equation of State (EOS) implementation.
   *
   * This utility facilitates static polymorphism by mapping a runtime \p eos_type  to a
   * compile-time EOS class. The provided callable \p f is invoked with a stateless instance of the
   * corresponding EOS implementation.
   *
   * @param eos_type The runtime identifier for the desired equation of state.
   * @param f The functional to execute.
   *
   * @return The result of invoking \p f with the appropriate EOS instance.
   *
   * @note All potential execution paths of \p f must return the same type to
   * satisfy the `decltype(auto)` return deduction.
   *
   * @throws dealii::ExcMessage if \p eos_type is not implemented in the switch.
   */
  template <typename F>
  inline DEAL_II_ALWAYS_INLINE //
    decltype(auto)
    dispatch_eos(CompressibleFlow::EquationOfState eos_type, F &&f)
  {
    switch (eos_type)
      {
        case CompressibleFlow::EquationOfState::ideal_gas:
          return std::forward<F>(f)(IdealGasEOS{});
        default:
          AssertThrow(false,
                      dealii::ExcMessage("The provided equation of state type is not supported!"));
      }
  }


  /**
   * CRTP mixin providing thermodynamic state evaluation via the above defined equation of state.
   *
   * @tparam Derived The derived class is expected to provide an \p eos_type() method returning the
   * type of the equation of state to be used, as well as the necessary data accessors required by
   * the EOS implementations. For details on the required data accessors, please refer to the
   * individual EOS implementation.
   */
  template <int dim, ArithmeticType ValueType, typename Derived>
  struct EOSValueMixin
  {
    decltype(auto)
    pressure() const
    {
      return dispatch_eos(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.thermodynamic_pressure(derived(), derived());
      });
    }

    decltype(auto)
    temperature() const
    {
      return dispatch_eos(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.temperature(derived(), derived());
      });
    }

    decltype(auto)
    speed_of_sound() const
    {
      return dispatch_eos(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.speed_of_sound(derived(), derived());
      });
    }

    decltype(auto)
    inner_energy_from_pressure(const ValueType &pressure) const
    {
      return dispatch_eos(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.inner_energy_from_pressure(pressure, derived());
      });
    }

  private:
    decltype(auto)
    eos_type() const
    {
      return static_cast<const Derived &>(*this).eos_type();
    }

    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }
  };

  /**
   * CRTP mixin providing thermodynamic state gradient evaluation via an equation of state.
   *
   * @tparam Derived The derived class is expected to provide an \p eos_type() method returning the
   * type of the equation of state to be used, as well as the necessary data accessors required by
   * the EOS implementations. For details on the required data accessors, please refer to the
   * individual EOS implementation.
   */
  template <int dim, typename Derived>
  struct EOSGradientMixin
  {
    decltype(auto)
    grad_temperature() const
    {
      return dispatch_eos(eos_type(), [&](auto eos) -> decltype(auto) {
        return eos.grad_temperature(derived(), derived(), derived());
      });
    }

  private:
    const Derived &
    derived() const
    {
      return static_cast<const Derived &>(*this);
    }

    decltype(auto)
    eos_type() const
    {
      return static_cast<const Derived &>(*this).eos_type();
    }
  };
} // namespace MeltPoolDG::Flow
