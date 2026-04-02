#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <memory>


namespace MeltPoolDG::Functions
{
  /**
   * @brief Wrapper function that flips the sign of another function.
   *
   * This class takes a shared pointer to a dealii::Function and returns the negative of its value,
   * gradient, and hessian. Time-related operations are forwarded to the underlying function.
   */
  template <int dim, typename number>
  class ChangedSignFunction : public dealii::Function<dim, number>
  {
  public:
    ChangedSignFunction(const std::shared_ptr<dealii::Function<dim, number>> fu_)
      : dealii::Function<dim, number>(fu_->n_components, fu_->get_time())
      , fu(fu_)
    {
      AssertThrow(fu, dealii::ExcMessage("The input function does not exist. Abort ..."));
    }

    number
    value(const dealii::Point<dim> &point, const unsigned int component) const override
    {
      return -fu->value(point, component);
    }

    dealii::Tensor<1, dim, number>
    gradient(const dealii::Point<dim> &point, const unsigned int component) const override
    {
      return -fu->gradient(point, component);
    }

    dealii::SymmetricTensor<2, dim, number>
    hessian(const dealii::Point<dim> &point, const unsigned int component) const override
    {
      return -fu->hessian(point, component);
    }

    void
    set_time(const number new_time) override
    {
      fu->set_time(new_time);
    }

    void
    advance_time(const number delta_t) override
    {
      fu->advance_time(delta_t);
    }

  private:
    const std::shared_ptr<dealii::Function<dim, number>> fu;
  };

  /**
   * @brief Adapter to create a two-component function from a scalar function.
   *
   * This class wraps a single-component dealii::Function and exposes it as an n component function
   * by returning the same scalar value for all components.
   */
  template <int dim, typename number>
  class NComponentFunction : public dealii::Function<dim>
  {
  public:
    NComponentFunction(const dealii::Function<dim> &function_in,
                       const unsigned int           n_components = 2)
      : dealii::Function<dim>(n_components)
      , function(function_in)
    {}

    number
    value(const dealii::Point<dim> &p, const unsigned int /* component */) const override
    {
      return function.value(p);
    }

  private:
    const dealii::Function<dim> &function;
  };

  /**
   * @brief Embeds a function into a higher-dimensional component space.
   *
   * This class inserts a given base function into a larger function with more components, starting
   * at a specified component index. Components outside the embedded range return zero.
   */
  template <int dim, typename number>
  class EmbeddedComponentsFunction : public dealii::Function<dim, number>
  {
  public:
    EmbeddedComponentsFunction(const dealii::Function<dim, number> &base_function,
                               const unsigned int                   n_components,
                               const unsigned int                   start_component)
      : dealii::Function<dim, number>(n_components)
      , base_function(base_function)
      , start(start_component)
    {
      Assert(start + base_function.n_components <= n_components,
             dealii::ExcMessage(
               "The base function does not fit into the specified number of components."));
    }

    number
    value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
    {
      if (component >= start and component < start + base_function.n_components)
        {
          return base_function.value(p, component - start);
        }
      else
        return 0.0;
    }

  private:
    const dealii::Function<dim, number> &base_function;
    const unsigned int                   start;
  };

  /**
   * @brief Extracts a subset of components from a multi-component function.
   *
   * This class provides a view onto a subset of components of a given base function, starting at a
   * specified index and exposing a reduced number of components.
   */
  template <int dim, typename number>
  class ExtractedComponentsFunction : public dealii::Function<dim, number>
  {
  public:
    ExtractedComponentsFunction(const dealii::Function<dim, number> &base_function,
                                const unsigned int                   start_component,
                                const unsigned int                   n_components)
      : dealii::Function<dim, number>(n_components)
      , base_function(base_function)
      , start(start_component)
      , n_components(n_components)
    {
      Assert(start + n_components <= base_function.n_components,
             dealii::ExcMessage(
               "The extracted function does not fit into the specified number of components."));
    }

    number
    value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
    {
      AssertIndexRange(component, n_components);
      return base_function.value(p, start + component);
    }

  private:
    const dealii::Function<dim, number> &base_function;
    const unsigned int                   start;
    const unsigned int                   n_components;
  };
} // namespace MeltPoolDG::Functions
