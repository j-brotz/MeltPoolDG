#pragma once

#include <meltpooldg/utilities/enum.hpp>


namespace MeltPoolDG::LevelSet
{
  BETTER_ENUM(LevelSetType,
              char,
              // tanh-based level set function
              level_set,
              // sine-based smoothed Heaviside function
              heaviside,
              // sharp Heaviside function
              sharp_heaviside,
              // signed distance
              signed_distance)
}