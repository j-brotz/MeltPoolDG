#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace dealii::Functions
{
  template <int dim, typename number>
  class ChangedSignFunction : public Function<dim, number>
  {
  public:
    ChangedSignFunction(const std::shared_ptr<dealii::Function<dim, number>> fu_)
      : Function<dim, number>(fu_->n_components, fu_->get_time())
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



  // this is a workaround to make a 2 component function out of a single component function
  template <int dim, typename number>
  class TwoComponentFunction : public dealii::Function<dim>
  {
  public:
    TwoComponentFunction(const dealii::Function<dim> &function_in)
      : dealii::Function<dim>(2 /* n_components */)
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
} // namespace dealii::Functions
