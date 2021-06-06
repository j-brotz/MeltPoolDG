#pragma once
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::DistanceFunctions
{
  using namespace dealii;

  template <int dim>
  inline double
  spherical_manifold(const Point<dim> &p, const Point<dim> &center, const double radius)
  {
    if (dim == 3)
      return -std::sqrt(std::pow(p[0] - center[0], 2) + std::pow(p[1] - center[1], 2) +
                        std::pow(p[2] - center[2], 2)) +
             radius;
    else if (dim == 2)
      return -std::sqrt(std::pow(p[0] - center[0], 2) + std::pow(p[1] - center[1], 2)) + radius;
    else if (dim == 1)
      return -std::sqrt(std::pow(p[0] - center[0], 2)) + radius;
    else
      AssertThrow(false, ExcMessage("Spherical manifold: dim must be 1, 2 or 3."));
  }

  /**
   *  The following function describes the geometry of an ellipsoidal manifold implicitly
   *  WARNING: This is not a real distance function.
   *
   *
   *     sign=+         ------------
   *              -------          -------
   *         ------                      ------
   *       ---                                ---
   *      --                                    --
   *      --            sign = -                --
   *      --                                    --
   *       ---                                ---
   *         ------                      ------
   *              -------          -------
   *                    ------------
   *
   *
   */
  template <int dim>
  inline double
  ellipsoidal_manifold(const Point<dim> &p,
                       const Point<dim> &center,
                       const double      radius_x,
                       const double      radius_y,
                       const double      radius_z = 0)
  {
    if (dim == 3)
      return -std::pow(p[0] - center[0], 2) / std::pow(radius_x, 2) -
             std::pow(p[1] - center[1], 2) / std::pow(radius_y, 2) -
             std::pow(p[2] - center[2], 2) / std::pow(radius_z, 2) + 1;
    else if (dim == 2)
      return -std::pow(p[0] - center[0], 2) / std::pow(radius_x, 2) -
             std::pow(p[1] - center[1], 2) / std::pow(radius_y, 2) + 1;
    else if (dim == 1)
      return -std::pow(p[0] - center[0], 2) / std::pow(radius_x, 2) + 1;
    else
      AssertThrow(false, ExcMessage("Ellipsoidal manifold: dim must be 1, 2 or 3."));
  }


  template <int dim>
  inline double
  infinite_line(const Point<dim> &p, const Point<dim> &fix_p1, const Point<dim> &fix_p2)
  {
    if (dim == 3)
      return std::sqrt(std::pow((fix_p2[1] - fix_p1[1]) * (fix_p1[2] - p[2]) -
                                  (fix_p2[2] - fix_p1[2]) * (fix_p1[1] - p[1]),
                                2) +
                       std::pow((fix_p2[2] - fix_p1[2]) * (fix_p1[0] - p[0]) -
                                  (fix_p2[0] - fix_p1[0]) * (fix_p1[2] - p[2]),
                                2) +
                       std::pow((fix_p2[0] - fix_p1[0]) * (fix_p1[1] - p[1]) -
                                  (fix_p2[1] - fix_p1[1]) * (fix_p1[0] - p[0]),
                                2)) /
             std::sqrt(std::pow(fix_p2[0] - fix_p1[0], 2) + std::pow(fix_p2[1] - fix_p1[1], 2) +
                       std::pow(fix_p2[2] - fix_p1[2], 2));
    else if (dim == 2)
      return std::abs((fix_p2[0] - fix_p1[0]) * (fix_p1[1] - p[1]) -
                      (fix_p2[1] - fix_p1[1]) * (fix_p1[0] - p[0])) /
             std::sqrt(std::pow(fix_p2[0] - fix_p1[0], 2) + std::pow(fix_p2[1] - fix_p1[1], 2));
    else if (dim == 1)
      return std::abs(fix_p1[0] - p[0]);
    else
      AssertThrow(false, ExcMessage("distance to infinite line: dim must be 1, 2 or 3."));
  }

  //@todo: this function should be added to compute distance to slotted disc, not finished
  template <int dim>
  inline double
  signed_distance_slotted_disc(const Point<dim> &p,
                               const Point<dim> &center,
                               const double      radius,
                               const double      slot_w,
                               const double      slot_l)
  {
    if (dim == 2)
      {
        // default distance
        double d_AB       = std::numeric_limits<double>::max();
        double d_BC       = std::numeric_limits<double>::max();
        double d_CD       = std::numeric_limits<double>::max();
        double d_manifold = std::numeric_limits<double>::max();
        double d_min;
        // set corner points
        const double delta_y = radius - std::sqrt(std::pow(radius, 2) - (std::pow(slot_w, 2)) / 4);
        Point<dim>   pA      = Point<dim>(center[0] - slot_w / 2, center[1] - radius + delta_y);
        Point<dim>   pB      = Point<dim>(center[0] - slot_w / 2, center[1] + (slot_l - radius));
        Point<dim>   pC      = Point<dim>(center[0] + slot_w / 2, center[1] + (slot_l - radius));
        Point<dim>   pD      = Point<dim>(center[0] + slot_w / 2, center[1] - radius + delta_y);

        if (p[1] <= pA[1])
          {
            if (p[0] >= pA[0] && p[0] <= pD[0])
              { // region 10 and 11
                d_AB  = spherical_manifold<dim>(p, pA, 0.0);
                d_CD  = spherical_manifold<dim>(p, pD, 0.0);
                d_min = std::max(d_AB, d_CD);
              }
            else
              { // boundary region of 10 and 11
                d_AB       = spherical_manifold<dim>(p, pA, 0.0);
                d_CD       = spherical_manifold<dim>(p, pD, 0.0);
                d_manifold = spherical_manifold<dim>(p, center, radius);
                d_min      = std::max(d_AB, d_CD);
                d_min      = std::max(d_manifold, d_min);
              }
          }
        else if (p[1] >= pB[1])
          {
            if (p[0] <= pB[0])
              { // region 3
                d_BC       = std::abs(spherical_manifold<dim>(p, pB, 0.0));
                d_manifold = spherical_manifold<dim>(p, center, radius);
                d_min      = std::min(d_BC, d_manifold);
              }
            else if (p[0] >= pC[0])
              { // region 4
                d_BC       = std::abs(spherical_manifold<dim>(p, pC, 0.0));
                d_manifold = spherical_manifold<dim>(p, center, radius);
                d_min      = std::min(d_BC, d_manifold);
              }
            else if (p[0] > pB[0] && p[0] < pC[0])
              { // region 2
                d_BC       = infinite_line<dim>(p, pB, pC);
                d_manifold = spherical_manifold<dim>(p, center, radius);
                d_min      = std::min(d_BC, d_manifold);
              }
          }
        else if (p[0] > center[0] - radius && p[0] < center[0] + radius) // region 1, 5-7, 8, 9
          {
            if (p[0] > pB[0] && p[0] < pC[0]) // region 5-7
              {
                d_AB  = -infinite_line<dim>(p, pA, pB);
                d_BC  = -infinite_line<dim>(p, pB, pC);
                d_CD  = -infinite_line<dim>(p, pC, pD);
                d_min = std::max(d_AB, d_BC);
                d_min = std::max(d_CD, d_min);
              }
            else
              {
                d_AB       = infinite_line<dim>(p, pA, pB);
                d_CD       = infinite_line<dim>(p, pC, pD);
                d_manifold = spherical_manifold<dim>(p, center, radius);
                d_min      = std::min(d_AB, d_CD);
                d_min      = std::min(d_min, d_manifold);
              }
          }
        else
          { // outer region
            d_min = spherical_manifold<dim>(p, center, radius);
          }

        // return the sign of the smallest distance
        return UtilityFunctions::CharacteristicFunctions::sgn(d_min);
      }
  }

  /**
   *  This function defines the signed distance function of a rectangular manifold. The lower
   * left corner of the rectangle (lowest values for the x,y,z coordinates among the corner
   * points) and the upper left corner (highest values of the x,y,z coordinates) must be
   * provided as input. Inside the rectangle a positive and outside the rectangle a negative
   * value for the distance function is considered.
   */
  template <int dim>
  inline double
  rectangular_manifold(const Point<dim> &p,
                       const Point<dim> &lower_left_corner,
                       const Point<dim> &upper_right_corner)
  {
    if constexpr (dim == 3)
      {
        //@todo: compute distance
        // atm only the sign (if the point is inside or outside the box) is returned
        /**
         *
         *           sign(d)=-
         *
         *          (4)               (5)
         *            +---------------+
         *           /|    z         /|
         *          / |    ^        / |
         *         /  |    |       /  |
         *        /   |           /   |
         *   (7) +---------------+ (6)|
         *       |    |          |    |
         *       | (0)+----------|----+ (1)
         *       |   /sign(d)=+  |   /
         *       |  /            |  /  --> y
         *       | /      /      | /
         *       |/     x        |/
         *       +---------------+
         *    (3)                (2)
         *
         *
         *  (0) ... lower left
         *  (6) ... upper right
         *
         *
         */
        /// define corner points depending on the given lower_left_corner and upper_right_corner
        std::vector<Point<dim>> corner(dim * dim);
        corner[0] = lower_left_corner;
        corner[1] = Point<dim>(lower_left_corner[0], upper_right_corner[1], lower_left_corner[2]);
        corner[2] = Point<dim>(upper_right_corner[0], upper_right_corner[1], lower_left_corner[2]);
        corner[3] = Point<dim>(upper_right_corner[0], lower_left_corner[1], lower_left_corner[2]);
        corner[4] = Point<dim>(lower_left_corner[0], lower_left_corner[1], upper_right_corner[2]);
        corner[5] = Point<dim>(lower_left_corner[0], upper_right_corner[1], upper_right_corner[2]);
        corner[6] = upper_right_corner;
        corner[7] = Point<dim>(upper_right_corner[0], lower_left_corner[1], upper_right_corner[2]);

        Point<dim> center;
        for (int d = 0; d < dim; ++d)
          center[d] = 0.5 * (upper_right_corner[d] + lower_left_corner[d]);

        auto project = [&](const Point<3> &p, const int plane) -> Point<2> {
          if (plane == 0)
            return Point<2>(p[1], p[2]);
          else if (plane == 1)
            return Point<2>(p[0], p[2]);
          else
            return Point<2>(p[0], p[1]);
        };

        // test if point is on one of the 6 faces
        // plane x = const
        if ((p[0] == corner[0][0]) &&
            (rectangular_manifold<2>(project(p, 0), project(corner[0], 0), project(corner[5], 0)) >
             0))
          return 0.0;
        // plane y = const
        else if ((p[1] == corner[0][1]) && (rectangular_manifold<2>(project(p, 1),
                                                                    project(corner[0], 1),
                                                                    project(corner[7], 1)) > 0))
          return 0.0;
        // plane z = const
        else if ((p[2] == corner[0][2]) && (rectangular_manifold<2>(project(p, 2),
                                                                    project(corner[0], 2),
                                                                    project(corner[2], 2)) > 0))
          return 0.0;
        if ((p[0] == upper_right_corner[0]) &&
            (rectangular_manifold<2>(project(p, 0), project(corner[3], 0), project(corner[6], 0)) >
             0))
          return 0.0;
        // plane y = const
        else if ((p[1] == upper_right_corner[1]) &&
                 (rectangular_manifold<2>(project(p, 1),
                                          project(corner[1], 1),
                                          project(corner[6], 1)) > 0))
          return 0.0;
        // plane z = const
        else if ((p[2] == lower_left_corner[2]) &&
                 (rectangular_manifold<2>(project(p, 2),
                                          project(corner[4], 2),
                                          project(corner[6], 2)) > 0))
          return 0.0;

        // test if point is inside the rectangle
        if ((p[0] > lower_left_corner[0]) && (p[0] < upper_right_corner[0]))
          if ((p[1] > lower_left_corner[1]) && (p[1] < upper_right_corner[1]))
            if ((p[2] > lower_left_corner[2]) && (p[2] < upper_right_corner[2]))
              return +1.0;

        return -1.0;
      }
    else if constexpr (dim == 2)
      {
        if (lower_left_corner[0] <= p[0] && p[0] <= upper_right_corner[0] &&
            lower_left_corner[1] <= p[1] && p[1] <= upper_right_corner[1])
          // inside
          return std::min({p[0] - lower_left_corner[0],
                           upper_right_corner[0] - p[0],
                           p[1] - lower_left_corner[1],
                           upper_right_corner[1] - p[1]});
        else if (lower_left_corner[0] <= p[0] && p[0] <= upper_right_corner[0])
          // top or bottom
          return -std::min(std::abs(lower_left_corner[1] - p[1]),
                           std::abs(p[1] - upper_right_corner[1]));
        else if (lower_left_corner[1] <= p[1] && p[1] <= upper_right_corner[1])
          // left or right
          return -std::min(std::abs(lower_left_corner[0] - p[0]),
                           std::abs(p[0] - upper_right_corner[0]));
        else
          // corner
          return -std::min({p.distance(lower_left_corner),
                            p.distance(upper_right_corner),
                            p.distance(Point<2>(lower_left_corner[0], upper_right_corner[1])),
                            p.distance(Point<2>(upper_right_corner[0], lower_left_corner[1]))});
      }
    else if constexpr (dim == 1)
      {
        /**
         *        lower left      upper right
         *    (-)     (0) +--(+)---+ (1)  (-)    --> x
         */
        double center;
        center = 0.5 * (upper_right_corner[0] + lower_left_corner[0]);
        if (p[0] <= lower_left_corner[0])
          return -p.distance(lower_left_corner); /* point is outside of the rectangle */
        else if (p[0] >= upper_right_corner[0])
          return -p.distance(upper_right_corner); /* point is outside of the rectangle */
        else if (p[0] <= center)
          return p.distance(lower_left_corner); /* point is inside the left half of the rectangle */
        else
          return p.distance(
            upper_right_corner); /* point is inside the right half of the rectangle */
      }
    else
      AssertThrow(false, ExcMessage("Rectangular manifold: dim must be 1,2 or 3."));
    return 0.0;
  }
} // namespace MeltPoolDG::DistanceFunctions
