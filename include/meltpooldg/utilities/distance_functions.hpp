#pragma once
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/utility_functions.hpp>

namespace dealii::Functions::SignedDistance
{
  /**
   * Signed-distance level set function of a rectangle
   *
   * This function is zero on the rectangle, negative "inside" and positive
   * in the rest of $\mathbb{R}^{dim}$.
   *
   * @ingroup functions
   */
  template <int dim>
  class Rectangle : public Function<dim>
  {
  public:
    /**
     * Constructor, takes the bottom left point @bottom_left and the top right
     * point @top_right of the rectangle.
     */
    Rectangle(const Point<dim> &bottom_left, const Point<dim> &top_right)
      : bounding_box({bottom_left, top_right})
    {
      if constexpr (dim == 3)
        {
          boundary_faces.emplace_back(
            Functions::SignedDistance::Plane(bottom_left, -Point<dim>::unit_vector(0))); // left
          boundary_faces.emplace_back(
            Functions::SignedDistance::Plane(top_right, Point<dim>::unit_vector(0))); // right
          boundary_faces.emplace_back(
            Functions::SignedDistance::Plane(bottom_left,
                                             -Point<dim>::unit_vector(1))); // front
          boundary_faces.emplace_back(
            Functions::SignedDistance::Plane(top_right, Point<dim>::unit_vector(1))); // back
          boundary_faces.emplace_back(
            Functions::SignedDistance::Plane(bottom_left,
                                             -Point<dim>::unit_vector(2))); // bottom
          boundary_faces.emplace_back(
            Functions::SignedDistance::Plane(top_right, Point<dim>::unit_vector(2))); // top
        }
    }

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      const Point<dim> &bottom_left = bounding_box.get_boundary_points().first;
      const Point<dim> &top_right   = bounding_box.get_boundary_points().second;

      if constexpr (dim == 3)
        {
          // inside (1 case)
          if (bounding_box.point_inside(p))
            {
              double signed_distance = std::numeric_limits<double>::lowest();

              for (const auto &plane : boundary_faces)
                signed_distance = std::max(plane.value(p), signed_distance);

              return signed_distance;
            }
          // boundary faces (6 cases)
          for (unsigned int i = 0; i < boundary_faces.size(); ++i)
            {
              if (boundary_faces[i].value(p) >= 1e-16)
                {
                  // check if all other faces have negative distance values
                  bool all_faces_are_negative = true;
                  for (unsigned int j = 0; j < boundary_faces.size(); ++j)
                    {
                      if ((i != j) && boundary_faces[j].value(p) > 0)
                        {
                          all_faces_are_negative = false;
                          break;
                        }
                    }

                  if (all_faces_are_negative)
                    return boundary_faces[i].value(p);
                }
            }
          // corners (8 cases)
          for (unsigned int i = 0; i < 8; i++)
            {
              const auto face_indices = GeometryInfo<dim>::vertex_to_face[i];

              bool found = true;

              std::vector<unsigned int> processed_faces(6);
              std::iota(processed_faces.begin(), processed_faces.end(), 0);

              for (const auto &f : face_indices)
                {
                  if (boundary_faces[f].value(p) < 0)
                    {
                      found = false;
                      break;
                    }

                  processed_faces.erase(
                    std::find(processed_faces.begin(), processed_faces.end(), f));
                }

              if (found)
                {
                  for (const auto &f : processed_faces)
                    {
                      if (boundary_faces[f].value(p) > 0)
                        {
                          found = false;
                          break;
                        }
                    }
                }

              if (found)
                return bounding_box.vertex(i).distance(p);
            }
          // boundary lines (12 cases)
          for (unsigned int i = 0; i < 12; i++)
            {
              bool found = true;

              std::vector<unsigned int> processed_faces(6);
              std::iota(processed_faces.begin(), processed_faces.end(), 0);

              for (const auto &f : line_to_face.at(i))
                {
                  if (boundary_faces[f].value(p) < 0)
                    {
                      found = false;
                      break;
                    }

                  processed_faces.erase(
                    std::find(processed_faces.begin(), processed_faces.end(), f));
                }

              if (found)
                {
                  for (const auto &f : processed_faces)
                    {
                      if (boundary_faces[f].value(p) > 0)
                        {
                          found = false;
                          break;
                        }
                    }
                }

              if (found)
                return distance_to_line(
                  p,
                  bounding_box.vertex(GeometryInfo<dim>::line_to_cell_vertices(i, 0)),
                  bounding_box.vertex(GeometryInfo<dim>::line_to_cell_vertices(i, 1)));
            }
          AssertThrow(false,
                      ExcMessage("The distance of your requested point could not be calculated."));
          return 0;
        }
      if constexpr (dim == 2)
        {
          // inside
          if (bounding_box.point_inside(p))
            return -std::min({p[0] - bottom_left[0],
                              top_right[0] - p[0],
                              p[1] - bottom_left[1],
                              top_right[1] - p[1]});
          // top or bottom
          else if (bottom_left[0] <= p[0] && p[0] <= top_right[0])
            return std::min(std::abs(bottom_left[1] - p[1]), std::abs(p[1] - top_right[1]));
          // left or right
          else if (bottom_left[1] <= p[1] && p[1] <= top_right[1])
            return std::min(std::abs(bottom_left[0] - p[0]), std::abs(p[0] - top_right[0]));
          else
            // corner
            return std::min({p.distance(bottom_left),
                             p.distance(top_right),
                             p.distance(Point<2>(bottom_left[0], top_right[1])),
                             p.distance(Point<2>(top_right[0], bottom_left[1]))});
        }
      else if constexpr (dim == 1)
        {
          /**
           *        lower left      upper right
           *    (-)     (0) +--(+)---+ (1)  (-)    --> x
           */
          if (p[0] <= bottom_left[0])
            return p.distance(bottom_left); /* point is outside of the rectangle */
          else if (p[0] >= top_right[0])
            return p.distance(top_right); /* point is outside of the rectangle */
          else if (p[0] <= bounding_box.center()[0])
            return -p.distance(bottom_left); /* point is inside the left half of the rectangle */
          else
            return -p.distance(top_right); /* point is inside the right half of the rectangle */
        }
      else if constexpr (dim == 3)
        {}
      else
        return 0;
    }

  private:
    const BoundingBox<dim>                             bounding_box;
    std::vector<Functions::SignedDistance::Plane<dim>> boundary_faces;

    // TODO: move to GeometryInfo<dim> (?)
    const std::map<unsigned int, std::vector<unsigned int>> line_to_face = {{0, {4, 0}},
                                                                            {1, {4, 1}},
                                                                            {2, {4, 2}},
                                                                            {3, {4, 3}},
                                                                            {4, {5, 0}},
                                                                            {5, {5, 1}},
                                                                            {6, {5, 2}},
                                                                            {7, {5, 3}},
                                                                            {8, {2, 0}},
                                                                            {9, {2, 1}},
                                                                            {10, {3, 0}},
                                                                            {11, {3, 1}}};



    // TODO: move to internal
    /**
     * Compute the minimal distance between a point @p p and an infinite line described by two support
     * points @p bottom_left and @p top_right.
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
    static double
    distance_to_line(const Point<dim> &p,
                     const Point<dim> &bottom_left,
                     const Point<dim> &top_right)
    {
      Assert((top_right - bottom_left).norm() >= std::numeric_limits<double>::epsilon(),
             ExcMessage("The support points must not lie on top of each other! Abort.."));
      if (dim == 3)
        return cross_product_3d(top_right - bottom_left, bottom_left - p).norm() /
               (top_right - bottom_left).norm();
      else if (dim == 2)
        return std::abs((top_right - bottom_left) * cross_product_2d(bottom_left - p)) /
               (top_right - bottom_left).norm();
      else
        AssertThrow(false, ExcMessage("distance to infinite line: dim must be 2 or 3."));
    }
  };
} // namespace dealii::Functions::SignedDistance

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
