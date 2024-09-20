#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <memory>
#include <utility>
#include <vector>

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



  /**
   * This function generates the immersed quadrature rules in the case that the
   * domain is described by a discrete level-set function.
   * The classes NonMatching::DiscreteQuadratureGenerator<dim> and
   * NonMatching::DiscreteFaceQuadratureGenerator<dim> are used to generate the immersed
   * quadrature rules, for the immersed geometry described by discrete level-set function
   * via a (level_set_dof_handler, level_set) pair.
   *
   * @tparam dim Dimension in which this function is to be used.
   * @tparam number Floating-point value type used for this function.
   * @tparam VectorType Vector type used for this function.
   *
   * @param level_set_dof_handler DoFHandler object of the discrete level-set.
   * @param level_set Discrete level-set vector.
   * @param matrix_free Matrix free object, provides the iterators.
   * @param mapping_info_cells Vector of dealii::NonMatching::MappingInfo objects, provides
   * the mapping information computation and mapping data storage of the cells on the
   * inner subdomain and the outer subdomain, respectively.
   * @param mapping_info_surface dealii::NonMatching::MappingInfo object, provides the mapping
   * information computation and mapping data storage of the surface.
   * @param fe_degree
   * @param is_two_phase
   *
   * @note Currently, the @p fe_degree has to be the same for both subdomains in
   * the case of a two-domain problem.
   *
   */
  template <int dim, typename number, typename VectorType>
  void
  compute_intersected_quadrature(
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>>>
      mapping_info_cells,
    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
                                                                           &mapping_info_surface,
    const dealii::DoFHandler<dim>                                          &level_set_dof_handler,
    const VectorType                                                       &level_set,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const int                                                               fe_degree,
    const bool                                                              is_two_phase);
} // namespace MeltPoolDG::Heat::CutUtil