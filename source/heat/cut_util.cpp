#include <meltpooldg/heat/cut_util.hpp>
//

#include <deal.II/base/exceptions.h>

namespace MeltPoolDG::Heat::CutUtil
{
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



  template <int dim>
  void
  set_fe_index(const dealii::DoFHandler<dim>                  &dof_handler,
               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier)
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (not cell->is_locally_owned())
          continue;

        const auto cell_location = mesh_classifier.location_to_level_set(cell);
        if (cell_location == dealii::NonMatching::LocationToLevelSet::inside)
          cell->set_active_fe_index(CellCategory::liquid);
        else if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected)
          cell->set_active_fe_index(CellCategory::intersected);
        else if (cell_location == dealii::NonMatching::LocationToLevelSet::outside)
          cell->set_active_fe_index(CellCategory::gas);
        else
          AssertThrow(false, dealii::ExcMessage("Location not found."));
      }
  }

  template void
  set_fe_index<1>(const dealii::DoFHandler<1> &, const dealii::NonMatching::MeshClassifier<1> &);
  template void
  set_fe_index<2>(const dealii::DoFHandler<2> &, const dealii::NonMatching::MeshClassifier<2> &);
  template void
  set_fe_index<3>(const dealii::DoFHandler<3> &, const dealii::NonMatching::MeshClassifier<3> &);
} // namespace MeltPoolDG::Heat::CutUtil