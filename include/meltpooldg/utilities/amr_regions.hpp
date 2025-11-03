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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/grid/tria.h>

#include <boost/serialization/access.hpp>

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
      point_inside(const dealii::Point<dim, number> &p) const = 0;
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
   * @brief Represents a spherical region for adaptive mesh refinement.
   */
  template <int dim, typename number>
  class SphereAMRRegion
  {
  public:
    /**
     * @brief Construct a spherical AMR region.
     *
     * @param center The center of the sphere.
     * @param radius The radius of the sphere.
     *
     * @throws dealii::ExcMessage if radius <= 0.
     */
    SphereAMRRegion(const dealii::Point<dim, number> &center, const number radius)
      : center(center)
      , radius(radius)
    {
      AssertThrow(radius > 0, dealii::ExcMessage("The radius cannot be negative."));
    }

    /**
     * @brief Check whether a point lies inside the sphere.
     *
     * @param p The point to test.
     * @return `true` if the point is inside the sphere (including the surface), `false` otherwise.
     */
    bool
    point_inside(const dealii::Point<dim, number> &p) const
    {
      return p.distance(center) <= radius;
    }

  private:
    const dealii::Point<dim, number> center;
    const number                     radius;
  };

  /**
   * @brief Represents a spherical shell region for adaptive mesh refinement (AMR).
   */
  template <int dim, typename number>
  class SphericalShellAMRRegion
  {
  public:
    /**
     * @brief Default constructor (required for boost serialization). Sets all members to 0.
     *
     * @note This constructor is not meant to be used by the user as it leaves the object in a state
     * where the main function of the class point_inside() cannot be used in a meaningful way.
     */
    SphericalShellAMRRegion() = default;

    /**
     * @brief Construct a spherical shell AMR region.
     *
     * @param center The center of the shell.
     * @param inner_radius The inner radius of the shell.
     * @param outer_radius The outer radius of the shell.
     *
     * @throws dealii::ExcMessage if inner_radius < 0 or inner_radius > outer_radius.
     */
    SphericalShellAMRRegion(const dealii::Point<dim, number> &center,
                            const number                      inner_radius,
                            const number                      outer_radius)
      : center(center)
      , inner_radius(inner_radius)
      , outer_radius(outer_radius)
    {
      AssertThrow(inner_radius >= 0, dealii::ExcMessage("The radius cannot be negative."));
      AssertThrow(inner_radius < outer_radius,
                  dealii::ExcMessage("The inner radius must be smaller than the outer radius"));
    }

    /**
     * @brief Check whether a point lies inside the spherical shell.
     *
     * @param p The point to test.
     * @return `true` if the point is inside the shell (including the surface), `false` otherwise.
     */
    bool
    point_inside(const dealii::Point<dim, number> &p) const
    {
      const number distance = p.distance(center);
      return (distance <= outer_radius and distance >= inner_radius);
    }

    /**
     * @brief Serialize or deserialize the object's data members.
     *
     * @param ar Reference to the archive object used for serialization/deserialization.
     * @param version Version number of the class (unused).
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &center;
      ar &inner_radius;
      ar &outer_radius;
    }

  private:
    dealii::Point<dim, number> center       = dealii::Point<dim, number>();
    number                     inner_radius = 0.;
    number                     outer_radius = 0.;
  };


  /**
   * @brief Marks cells in a triangulation for refinement if their centers fall inside specified regions.
   *
   * This function iterates over all active cells in the given triangulation and sets the refine
   * flag for each cell whose center lies within any of the provided adaptive mesh refinement
   * regions. Optionally, if the cell center does not lie within any region and the refine flag has
   * not been already set for the cell, the coarsen flag is set.
   *
   * @param tria    The triangulation whose cells will be marked for refinement.
   * @param regions A list of AMR regions. If a cell center lies inside any of those regions, the
   * corresponding refinement flag will be set.
   * @param do_coarsening If set, all cells which are not inside any of the regions and not yet marked for * refinement are marked for coarsening.
   *
   * @return Returns `true` if any refine or coarsen flag has been set, otherwise returns `false`.
   * @note If no regions are provided, the function returns immediately.
   */
  template <int dim, typename number>
  bool
  set_refinement_flags_in_regions(dealii::Triangulation<dim>                &tria,
                                  const std::vector<AMRRegion<dim, number>> &regions       = {},
                                  bool                                       do_coarsening = false)
  {
    if (regions.empty())
      return false;

    bool any_flag_set = false;
    for (auto &cell : tria.active_cell_iterators())
      if (not cell->refine_flag_set())
        for (const auto &region : regions)
          {
            if (region.point_inside(cell->center()))
              {
                cell->clear_coarsen_flag();
                cell->set_refine_flag();
                any_flag_set = true;
                break;
              }
            else if (do_coarsening)
              {
                cell->set_coarsen_flag();
                any_flag_set = true;
              }
          }

    return any_flag_set;
  }
} // namespace MeltPoolDG::AMR