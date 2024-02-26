#pragma once
#include <deal.II/base/function.h>

namespace dealii::Functions
{
  template <int dim, typename Number = double>
  class ChangedSignFunction : public Function<dim, Number>
  {
  public:
    ChangedSignFunction(const std::shared_ptr<Function<dim, Number>> fu_)
      : Function<dim, Number>(fu_->n_components, fu_->get_time())
      , fu(fu_)
    {
      AssertThrow(fu, ExcMessage("The input function does not exist. Abort ..."));
    }

    Number
    value(const Point<dim> &point, const unsigned int component) const override
    {
      return -fu->value(point, component);
    }

    Tensor<1, dim>
    gradient(const Point<dim> &point, const unsigned int component) const override
    {
      return -fu->gradient(point, component);
    }

    SymmetricTensor<2, dim>
    hessian(const Point<dim> &point, const unsigned int component) const override
    {
      return -fu->hessian(point, component);
    }

    void
    set_time(const Number new_time) override
    {
      fu->set_time(new_time);
    }

    void
    advance_time(const Number delta_t) override
    {
      fu->advance_time(delta_t);
    }

  private:
    const std::shared_ptr<Function<dim, Number>> fu;
  };
} // namespace dealii::Functions
