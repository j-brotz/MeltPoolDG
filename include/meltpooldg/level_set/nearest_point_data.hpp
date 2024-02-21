#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG::LevelSet
{
  BETTER_ENUM(NearestPointType,
              char,
              // closest point in normal direction
              closest_point_normal,
              // closest point in normal direction and enforcing collinearity; here we
              // perform an iterative sequence of (1) correction in normal direction until the
              // tolerance is fulfilled and (2) subsequent correction in tangential direction
              closest_point_normal_collinear,
              // closest point in normal direction and enforcing collinearity considering the
              // algorithm by Coquerelle (2014); here we
              // perform an iterative sequence of (1) correction in tangential direction and
              // (2) subsequent iterative correction in normal direction
              closest_point_normal_collinear_coquerelle,
              // nearest point in the point cloud
              nearest_point)

  template <typename number = double>
  struct NearestPointData
  {
    number           rel_tol               = 1e-6;
    int              max_iter              = 20;
    NearestPointType type                  = NearestPointType::closest_point_normal;
    double           narrow_band_threshold = -1;
    double           isocontour            = 0.0;
    unsigned int     verbosity_level       = 0;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::LevelSet