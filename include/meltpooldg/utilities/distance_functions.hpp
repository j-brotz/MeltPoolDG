#pragma once
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/utility_functions.hpp>

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

namespace MeltPoolDG::DistanceFunctions
{
  using namespace dealii;

  /**
   *
   *  The following function describes the geometry of a circular manifold implicitly.
   *
   *    sign=-
   *
   *        *  *
   *     *        *
   *    *    +     *
   *    *          *
   *     *        *
   *        *  *
   *
   */
  template <int dim>
  inline double
  spherical_manifold(const Point<dim> &p, const Point<dim> &center, const double radius)
  {
    return -p.distance(center) + radius;
  }

  /**
   * Compute the minimal distance between a point @p p and an infinite line described by two support
   * points @p fix_p1 and @p fix_p2.
   *
   *    f1  x
   *        |
   *        |    d
   *        |--------- x
   *        |            p
   *        |
   *    f2  x
   *
   *      || (f2 - f1) x (f1 - p) ||
   * d = ---------------------------
   *            || f2 - f1 ||
   */
  template <int dim>
  inline double
  infinite_line(const Point<dim> &p, const Point<dim> &fix_p1, const Point<dim> &fix_p2)
  {
    Assert((fix_p2 - fix_p1).norm() >= std::numeric_limits<double>::epsilon(),
           ExcMessage("The support points must not lie on top of each other! Abort.."));
    if (dim == 3)
      return cross_product_3d(fix_p2 - fix_p1, fix_p1 - p).norm() / (fix_p2 - fix_p1).norm();
    else if (dim == 2)
      return std::abs((fix_p2 - fix_p1) * cross_product_2d(fix_p1 - p)) / (fix_p2 - fix_p1).norm();
    else
      AssertThrow(false, ExcMessage("distance to infinite line: dim must be 2 or 3."));
  }
} // namespace MeltPoolDG::DistanceFunctions
