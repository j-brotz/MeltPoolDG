#pragma once

#include <meltpooldg/utilities/enum.hpp>


namespace MeltPoolDG::LevelSet
{
  BETTER_ENUM(LevelSetType,
              char,
              // tanh-based level set function
              tanh,
              // sine-based smoothed Heaviside function
              smoothed_heaviside,
              // sharp Heaviside function
              heaviside,
              // signed distance
              signed_distance)
}