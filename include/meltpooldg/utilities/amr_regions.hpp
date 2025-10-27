/**
 * @file
 * @brief Utilities for region-based adaptive mesh refinement (AMR).
 *
 * This file provides functions and classes that support region-based adaptive
 * mesh refinement, where refinement is restricted to specific user-defined
 * regions of the computational domain.
 *
 * The refinement strategy within each region can be customized depending on
 * the application. For instance, users may perform:
 *  - uniform refinement up to a specified refinement level, or
 *  - indicator-based refinement guided by error estimators, restricted to the
 *    selected regions.
 */

#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/tria.h>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace MeltPoolDG::AMR
{
  /**
   * @brief Type-erased wrapper representing a geometric region for adaptive mesh refinement.
   *
   * This class provides a generic interface for region-based adaptive mesh refinement. It allows
   * users to define arbitrary region objects (e.g., boxes, spheres, or complex predicates) that
   * implement a method:
   * @code
   * bool point_inside(const dealii::Point<dim, number> &p) const;
   * @endcode
   * and then wrap them in an `AMRRegion` instance for use in the provided refinement algorithms.
   *
   * Internally, a type erasure pattern is employed to store heterogeneous region types behind a
   * uniform interface.
   */
  template <int dim, typename number>
  class AMRRegion
  {
  public:
    /**
     * @brief Constructs an AMRRegion from a user-defined region object.
     *
     * @param region The region object to wrap.
     */
    template <typename T>
    AMRRegion(T &&region)
      : region(std::make_unique<Model<T>>(std::forward<T>(region)))
    {}

    /**
     * @brief Checks whether a given point lies inside this region.
     *
     * @param p The point to test.
     * @return `true` if the point is inside the region, otherwise `false`.
     */
    bool
    point_inside(const dealii::Point<dim, number> &p) const
    {
      return region->point_inside(p);
    }

  private:
    /**
     * @brief Abstract base interface for all region types.
     *
     * Defines the required interface (the `point_inside()` function) for region models.
     */
    struct Concept
    {
      virtual ~Concept() = default;

      virtual bool
      point_inside(const dealii::Point<dim, number> &p) const;
    };

    /**
     * @brief Concrete wrapper around a user-defined region type.
     *
     * This class template models the @ref Concept interface by forwarding
     * calls to the underlying region object.
     *
     * @tparam T The user-defined region type, which must implement
     * `bool point_inside(const dealii::Point<dim, number>&) const`.
     */
    template <typename T>
    struct Model : public Concept
    {
      Model(const T &t)
        : model(t)
      {}

      bool
      point_inside(const dealii::Point<dim, number> &p) const override
      {
        return model.point_inside(p);
      }

    private:
      /// The actual object for describing the behavior of the region.
      T model;
    };

    /// Pointer to the type erasure model which holds the actual object describing the behavior of
    /// the region.
    std::unique_ptr<Concept> region;
  };

  /**
   * @brief Marks cells in a triangulation for refinement if their centers fall inside specified regions.
   *
   * This function iterates over all active cells in the given triangulation and sets the refine
   * flag for each cell whose center lies within any of the provided adaptive mesh refinement (AMR)
   * regions.
   *
   * @param tria    The triangulation whose cells will be marked for refinement.
   * @param regions A list of AMR regions. If a cell center lies inside any of those regions, the
   * corresponding refinement flag will be set.
   *
   * @note If no regions are provided, the function returns immediately.
   */
  template <int dim, typename number>
  void
  set_refinement_flags_in_regions(dealii::Triangulation<dim>                &tria,
                                  const std::vector<AMRRegion<dim, number>> &regions = {})
  {
    if (regions.empty())
      return;

    for (auto &cell : tria.active_cell_iterators())
      for (const auto &region : regions)
        if (region.point_inside(cell->center()))
          {
            cell->set_refine_flag();
            break;
          }
  }
} // namespace MeltPoolDG::AMR