#pragma once

#include <utility>

namespace MeltPoolDG::Heat::CutUtil
{
  /**
   * definition of the cell category numbering (active FE index)
   */
  enum CellCategory
  {
    liquid      = 0,
    intersected = 1,
    gas         = 2
  };

  /**
   * enumeration for the face type
   */
  enum FaceType
  {
    inside_face_liquid,
    inside_face_gas,
    intersected_face,
    mixed_face_liquid,
    mixed_face_gas
  };

  /**
   * This function categorizes the FaceType of the current face range.
   *
   * @param adjacent_cell_categories Pair which contains the category of the cells
   * on the two sides of the current range of faces.
   *
   */
  FaceType
  get_face_type(const std::pair<unsigned int, unsigned int> &adjacent_cell_categories)
  {
    // inside_face_liquid: both adjacent cells to the face are completely inside the liquid phase
    if (adjacent_cell_categories.first == CellCategory::liquid &&
        adjacent_cell_categories.second == CellCategory::liquid)
      {
        return inside_face_liquid;
      }
    // inside_face_gas: both adjacent cells to the face are completely inside the gas phase
    else if (adjacent_cell_categories.first == CellCategory::gas &&
             adjacent_cell_categories.second == CellCategory::gas)
      {
        return inside_face_gas;
      }
    // intersected_face: both adjacent cells to the face are intersected
    else if (adjacent_cell_categories.first == CellCategory::intersected &&
             adjacent_cell_categories.second == CellCategory::intersected)
      {
        return intersected_face;
      }
    // mixed_face_liquid: one adjacent cell of the face is completely inside the liquid phase
    // and the other adjacent cell of the face is intersected
    else if ((adjacent_cell_categories.first == CellCategory::liquid &&
              adjacent_cell_categories.second == CellCategory::intersected) ||
             (adjacent_cell_categories.first == CellCategory::intersected &&
              adjacent_cell_categories.second == CellCategory::liquid))
      {
        return mixed_face_liquid;
      }
    // mixed_face_gas one adjacent cell of the face is completely inside th gas phase
    // and the other adjacent cell of the face is intersected by the immersed boundary
    else
      {
        return mixed_face_gas;
      }
  }
} // namespace MeltPoolDG::Heat::CutUtil