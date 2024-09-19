#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/non_matching/mesh_classifier.h>

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
  get_face_type(const std::pair<unsigned int, unsigned int> &adjacent_cell_categories);


  /**
   * This function is setting the FE index for every cell.
   *
   * @param dof_handler DoFHandler object, provides the cell iterator.
   * @param mesh_classifier The dealii::NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function. Note that the reclassify() function has to be called
   * before, so that the cell and face locations are categorized as one of the values of
   * dealii::NonMatching::LocationToLevelSet: inside, outside or intersected.
   *
   */
  template <int dim>
  void
  set_fe_index(const dealii::DoFHandler<dim>                  &dof_handler,
               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier);
} // namespace MeltPoolDG::Heat::CutUtil