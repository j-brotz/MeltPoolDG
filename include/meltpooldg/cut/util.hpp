#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace MeltPoolDG::CutUtil
{
  /**
   * Enum that names a type of CutFEM in use.
   * - not_cut: no CutFEM in use, the field in continuous
   * - one_phase_cut: The domain of the field is restricted to the liquid domain, i.e., positive
   *                  level set values. CutFEM is used for cells that contain the interface.
   * - two_phase_cut: The domain is cut at the interface with the liquid domain
   */
  BETTER_ENUM(CutPhaseType, char, not_cut, one_phase_cut, two_phase_cut)

  /**
   * determine the CutType of the @param dof_handler
   */
  template <int dim>
  CutPhaseType
  get_cut_type(const dealii::DoFHandler<dim> &dof_handler);

  /**
   * definition of aliases for MappingInfo types
   */
  template <int dim, typename number>
  using MappingInfoType =
    dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>;

  template <int dim, typename number>
  using MappingInfoVectorType = std::vector<std::shared_ptr<MappingInfoType<dim, number>>>;

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
    mixed_face_liquid_intersected, // "inside" cell in liquid phase; "outside" cell intersected
    mixed_face_intersected_liquid, // "inside" cell intersected; "outside" cell in liquid phase
    mixed_face_gas_intersected,    // "inside" cell in gas phase; "outside" cell intersected
    mixed_face_intersected_gas     // "inside" cell intersected; "outside" cell in gas phase
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
   * @param set_future Criteria, whether the active FE index should be set (false) or
   * the future FE index should be set (true) in this function.
   */
  template <int dim>
  void
  set_fe_index(const dealii::DoFHandler<dim>                  &dof_handler,
               const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
               const bool                                      set_future);



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
   * @param is_dg Consider FE_DGQ elements?
   * @param mapping_info_faces Vector of dealii::NonMatching::MappingInfo objects, provides
   * the mapping information computation and mapping data storage of the faces on the
   * inner subdomain and the outer subdomain, respectively.
   *
   * @note Currently, the @p fe_degree has to be the same for both subdomains in
   * the case of a two-domain problem.
   * @note Keep attention of the definition of the level-set field orientation. Currently, the single-phase region is assigned to the positive level-set region.
   *
   */
  template <int dim, typename number, typename VectorType>
  void
  compute_intersected_quadrature(
    MappingInfoVectorType<dim, number>                                      mapping_info_cells,
    MappingInfoType<dim, number>                                           &mapping_info_surface,
    const dealii::DoFHandler<dim>                                          &level_set_dof_handler,
    const VectorType                                                       &level_set,
    const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &matrix_free,
    const int                                                               fe_degree,
    const bool                                                              is_two_phase,
    const bool                                                              is_dg = false,
    MappingInfoVectorType<dim, number> mapping_info_faces                         = {});



  template <int dim, typename number, int n_components = 1>
  inline void
  evaluate_intersected_domain(
    dealii::FEPointEvaluation<n_components, dim, dim, dealii::VectorizedArray<number>> &point_eval,
    const dealii::FECellIntegrator<dim, n_components, number>                          &cell_eval,
    const dealii::EvaluationFlags::EvaluationFlags evaluation_flags,
    const unsigned int                             cell_batch,
    const unsigned int                             cell_lane,
    const unsigned int                             n_dofs_per_cell)
  {
    static constexpr unsigned int n_lanes = dealii::VectorizedArray<number>::size();

    point_eval.reinit(cell_batch * n_lanes + cell_lane);
    point_eval.evaluate(dealii::StridedArrayView<const number, n_lanes>(
                          &cell_eval.begin_dof_values()[0][cell_lane], n_dofs_per_cell),
                        evaluation_flags);
  }



  /**
   * This function checks whether the currently considered face requires the application
   * of ghost-penalty stabilization or not. A face is a ghost-penalty face if one of the two
   * adjacent cells is an intersected cell and the other cell is either inside the active
   * domain or is also an intersected cell.
   *
   * @tparam dim Dimension in which this function is to be used.
   *
   * @param mesh_classifier The NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function. Note that the reclassify() function has to be called
   * before, so that the cell and face locations are categorized as one of the values of
   * LocationToLevelSet: inside, outside or intersected.
   * @param cell The considered cell for the function evaluation.
   * @param face_index The index of the considered face, for which the check is done,
   * whether this face is a ghost-penalty-face or not.
   * @param inactive_location Location of the domain, which is not active (location for
   * which a FE_Nothing element is set).
   *
   * @note For multiple component problems, the @p inactive_location has to be chosen
   * carefully at the relevant code section, at which this function is called, as the
   * @p inactive_location depends on the currently considered component. For single
   * component problems, the default value of the inactive location is outside.
   *
   */
  template <int dim>
  bool
  face_has_ghost_penalty(const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
                         const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
                         const unsigned int                             face_index,
                         const dealii::NonMatching::LocationToLevelSet &inactive_location =
                           dealii::NonMatching::LocationToLevelSet::inside)
  {
    if (cell->at_boundary(face_index))
      return false;

    const dealii::NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);
    const dealii::NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected and
        neighbor_location != inactive_location)
      return true;

    if (neighbor_location == dealii::NonMatching::LocationToLevelSet::intersected and
        cell_location != inactive_location)
      return true;

    return false;
  }



  /**
   * This function checks whether the considered face is a newly created intersected face.
   * This occurs when both adjacent cells of the face are currently intersected cells and
   * were in the inactive location in the old state. This function is created for handling
   * moving interface problems.
   *
   * @tparam dim Dimension in which this function is to be used.
   *
   * @param mesh_classifier_old The NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function at the old state. The cell and face locations are
   * categorized as one of the values of LocationToLevelSet: inside, outside or intersected.
   * @param mesh_classifier The NonMatching::MeshClassifier object which contains the
   * information, how the active cells and faces of the triangulation are related to the
   * sign of a level set function at the current state. The cell and face locations are
   * categorized as one of the values of LocationToLevelSet: inside, outside or intersected.
   * @param cell The considered cell for the function evaluation.
   * @param face_index The index of the considered face, for which the check is done,
   * whether this face is a newly created intersected face or not.
   * @param inactive_location Location of the domain, which is not active (location for
   * which a FE_Nothing element is set).
   *
   * @note For multiple component problems, the @p inactive_location has to be chosen
   * carefully at the relevant code section, at which this function is called, as the
   * @p inactive_location depends on the currently considered component. For single
   * component problems, the default value of the inactive location is outside.
   * It has to be ensured that the @p mesh_classifier_old and @p mesh_classifier are
   * reclassified according to the correct states.
   *
   */
  template <int dim>
  bool
  is_new_intersected_face(const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
                          const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old,
                          const typename dealii::Triangulation<dim>::active_cell_iterator &cell,
                          const unsigned int                             face_index,
                          const dealii::NonMatching::LocationToLevelSet &inactive_location =
                            dealii::NonMatching::LocationToLevelSet::inside)
  {
    if (cell->at_boundary(face_index))
      return false;

    const dealii::NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);
    const dealii::NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    const dealii::NonMatching::LocationToLevelSet cell_location_old =
      mesh_classifier_old.location_to_level_set(cell);
    const dealii::NonMatching::LocationToLevelSet neighbor_location_old =
      mesh_classifier_old.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected and
        neighbor_location == dealii::NonMatching::LocationToLevelSet::intersected and
        cell_location_old == inactive_location and neighbor_location_old == inactive_location)
      return true;

    return false;
  }
} // namespace MeltPoolDG::CutUtil