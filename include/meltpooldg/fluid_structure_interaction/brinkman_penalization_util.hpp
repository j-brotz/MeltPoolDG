#pragma once

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>

#include <functional>

namespace MeltPoolDG
{
  /**
   * Implementation of a discontinuous mask function, which returns one if a given coordinate is
   * inside an obstacle and zero otherwise.
   *
   * @param location Coordinates at which the mask function shall be computed.
   * @param property_pool Property pool holding all required information about the obstacles at
   * the given locations.
   * @param handle Handle to identify relevant obstacles in the property pool.
   *
   * @return Value of mask function at given coordinates.
   */
  template <int dim, typename VectorizedArrayType, typename ObstacleType>
  VectorizedArrayType
  discontinuous_mask_function(const dealii::Point<dim, VectorizedArrayType> &location,
                              dealii::Particles::PropertyPool<dim>          &property_pool,
                              const typename dealii::Particles::PropertyPool<dim>::Handle handle)
  {
    dealii::Point<dim, VectorizedArrayType> vectorized_obstacle_location;
    for (auto i = 0; i < dim; ++i)
      vectorized_obstacle_location[i] = VectorizedArrayType(property_pool.get_location(handle)[i]);
    const VectorizedArrayType distance = location.distance(vectorized_obstacle_location);
    return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than_or_equal>(
      distance,
      VectorizedArrayType(
        ObstacleType::get_property(property_pool, handle, ObstacleType::Properties::radius)),
      VectorizedArrayType(1.),
      VectorizedArrayType(0.));
  }

  /**
   * @brief Scratch data structure used for caching cell relevant data when computing the
   * Brinkman penalization force for a specific cell or cell batch in the case of vectroized
   * computations.
   */
  template <int dim,
            typename number,
            typename ObstacleType,
            typename VectorizedArrayType = dealii::VectorizedArray<number>>
  struct BrinkmanPenalizationCellScratchData
  {
    using MaskFunctionType =
      std::function<VectorizedArrayType(const dealii::Point<dim, VectorizedArrayType> &,
                                        dealii::Particles::PropertyPool<dim> &,
                                        typename dealii::Particles::PropertyPool<dim>::Handle)>;

    BrinkmanPenalizationCellScratchData(
      const ObstacleField<dim, number, ObstacleType> &obstacle_handler,
      const BrinkmanPenalizationData<number>         &brinkman_penalization_data,
      MaskFunctionType                                mask_function =
        discontinuous_mask_function<dim, VectorizedArrayType, ObstacleType>)
      : relevant_obstacles(ObstacleType::n_obstacle_properties)
      , obstacle_handler(obstacle_handler)
      , data(brinkman_penalization_data)
      , mask_function(mask_function)
    {}

    ~BrinkmanPenalizationCellScratchData()
    {
      for (auto handle : relevant_obstacle_handles)
        relevant_obstacles.deregister_particle(handle);
    }

    /// Property pool used to cache obstacle properties relevant for the current cell batch.
    dealii::Particles::PropertyPool<dim> relevant_obstacles;

    /// Handles of the particles which properties are stored in the @p relevant_obstacles
    /// property pool.
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> relevant_obstacle_handles;

    /// Reference to the obstacle handler managing all obstacles in the domain.
    const ObstacleField<dim, number, ObstacleType> &obstacle_handler;

    /// Data structure containing parameters required for computing the Brinkman penalization term.
    const BrinkmanPenalizationData<number> &data;

    /// Mask function used in the computation of the Brinkman penalization term. The default
    /// implementation returns 1.0 if the point lies inside the obstacle's radius, and 0.0
    /// otherwise.
    MaskFunctionType mask_function;
  };

  /**
   * Helper function to find all relevant obstacles in the obstacle field stored in @param data for
   * the given cell batch @param cell_batch and matrix-free object @param mf. The handles of the
   * relevant obstacles are stored in the @param data struct.
   */
  template <int dim, typename number, typename ObstacleType>
  void
  find_relevant_obstacles_in_cell_batch(
    BrinkmanPenalizationCellScratchData<dim, number, ObstacleType> &data,
    const dealii::MatrixFree<dim, number>                          &mf,
    const unsigned                                                  cell_batch)
  {
    for (auto handle : data.relevant_obstacle_handles)
      data.relevant_obstacles.deregister_particle(handle);

    data.relevant_obstacle_handles.clear();
    data.relevant_obstacle_handles =
      data.obstacle_handler.get_obstacles_in_cell_batch(data.relevant_obstacles, mf, cell_batch);
  }

  /**
   * Helper function which does essentially the same as above but for a single cell.
   *
   * @param data Stores relebvant obstacles which to be checked if in the given cell.
   * @param cell Cell for which relevant obstacles shall be find.
   */
  template <int dim, typename number, typename ObstacleType>
  void
  find_relevant_obstacles_in_cell(
    BrinkmanPenalizationCellScratchData<dim, number, ObstacleType, number> &data,
    const dealii::CellAccessor<dim>                                        &cell)
  {
    for (auto handle : data.relevant_obstacle_handles)
      data.relevant_obstacles.deregister_particle(handle);

    data.relevant_obstacle_handles.clear();
    data.relevant_obstacle_handles =
      data.obstacle_handler.get_obstacles_in_cell(data.relevant_obstacles, cell);
  }
} // namespace MeltPoolDG
