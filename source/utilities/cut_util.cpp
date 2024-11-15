#include <meltpooldg/utilities/cut_util.hpp>
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/immersed_surface_quadrature.h>
#include <deal.II/non_matching/quadrature_generator.h>

namespace MeltPoolDG::CutUtil
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

        // dealii::NonMatching::MeshClassifier classifies positive level set values as "outside".
        // With our definition of the level set, "outside" correlates with the liquid phase.
        const auto cell_location = mesh_classifier.location_to_level_set(cell);
        if (cell_location == dealii::NonMatching::LocationToLevelSet::outside)
          cell->set_active_fe_index(CellCategory::liquid);
        else if (cell_location == dealii::NonMatching::LocationToLevelSet::intersected)
          cell->set_active_fe_index(CellCategory::intersected);
        else if (cell_location == dealii::NonMatching::LocationToLevelSet::inside)
          cell->set_active_fe_index(CellCategory::gas);
        else
          AssertThrow(false, dealii::ExcMessage("Location not found."));
      }
  }


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
    const bool                                                              is_two_phase)
  {
    dealii::hp::QCollection<1> q_collection((dealii::QGauss<1>(fe_degree + 1)));

    const unsigned int n_lanes = dealii::VectorizedArray<number>::size();

    dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
      q_collection, level_set_dof_handler, level_set);

    std::vector<dealii::Quadrature<dim>> quad_vec_cells_inner_domain;
    quad_vec_cells_inner_domain.reserve(
      (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) * n_lanes);

    std::vector<dealii::Quadrature<dim>> quad_vec_cells_outer_domain;
    if (is_two_phase)
      quad_vec_cells_outer_domain.reserve(
        (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) * n_lanes);

    std::vector<dealii::NonMatching::ImmersedSurfaceQuadrature<dim>> quad_vec_surface;
    quad_vec_surface.reserve((matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) *
                             n_lanes);

    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> vector_cell_iterators;
    vector_cell_iterators.reserve(
      (matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches()) * n_lanes);

    const unsigned int n_cell_batches =
      matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();

    for (unsigned int cell_batch = 0; cell_batch < n_cell_batches; ++cell_batch)
      for (unsigned int lane = 0; lane < n_lanes; ++lane)
        {
          if (lane < matrix_free.n_active_entries_per_cell_batch(cell_batch))
            {
              vector_cell_iterators.push_back(matrix_free.get_cell_iterator(cell_batch, lane));
              quadrature_generator.generate(matrix_free.get_cell_iterator(cell_batch, lane));
            }
          else
            {
              // fill empty lanes with dummy data
              vector_cell_iterators.push_back(matrix_free.get_cell_iterator(cell_batch, 0));
              quadrature_generator.generate(matrix_free.get_cell_iterator(cell_batch, 0));
            }

          quad_vec_cells_inner_domain.push_back(quadrature_generator.get_inside_quadrature());
          if (is_two_phase)
            quad_vec_cells_outer_domain.push_back(quadrature_generator.get_outside_quadrature());

          quad_vec_surface.push_back(quadrature_generator.get_surface_quadrature());
        }

    mapping_info_cells[0]->reinit_cells(vector_cell_iterators, quad_vec_cells_inner_domain);
    if (is_two_phase)
      mapping_info_cells[1]->reinit_cells(vector_cell_iterators, quad_vec_cells_outer_domain);

    mapping_info_surface.reinit_surface(vector_cell_iterators, quad_vec_surface);
  }


  template void
  set_fe_index<1>(const dealii::DoFHandler<1> &, const dealii::NonMatching::MeshClassifier<1> &);
  template void
  set_fe_index<2>(const dealii::DoFHandler<2> &, const dealii::NonMatching::MeshClassifier<2> &);
  template void
  set_fe_index<3>(const dealii::DoFHandler<3> &, const dealii::NonMatching::MeshClassifier<3> &);

  template void
  compute_intersected_quadrature(
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>>>>,
    dealii::NonMatching::MappingInfo<1, 1, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<1> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<1, double, dealii::VectorizedArray<double>> &,
    const int,
    const bool);
  template void
  compute_intersected_quadrature(
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>>>>,
    dealii::NonMatching::MappingInfo<2, 2, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<2, double, dealii::VectorizedArray<double>> &,
    const int,
    const bool);
  template void
  compute_intersected_quadrature(
    std::vector<
      std::shared_ptr<dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>>>>,
    dealii::NonMatching::MappingInfo<3, 3, dealii::VectorizedArray<double>> &,
    const dealii::DoFHandler<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const dealii::MatrixFree<3, double, dealii::VectorizedArray<double>> &,
    const int,
    const bool);
} // namespace MeltPoolDG::CutUtil