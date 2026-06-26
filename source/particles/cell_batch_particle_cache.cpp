#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/particles/cell_batch_particle_cache.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <functional>
#include <utility>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  void
  MatrixFreeCellBatchParticleCache<dim, number, ObstacleType>::update(
    CellListParticleHandler<dim, number, ObstacleType>                            &particle_handler,
    const typename CellListParticleHandler<dim, number, ObstacleType>::NotifyEvent event)
  {
    if (event == CellListParticleHandler<dim, number, ObstacleType>::NotifyEvent::
                   SortParticlesIntoSubdomainsAndCells or
        event ==
          CellListParticleHandler<dim, number, ObstacleType>::NotifyEvent::ObserverInitialization)
      {
        cell_batch_to_particle_cache.clear();
        cell_batch_to_particle_cache.resize(mf_context.mf.n_cell_batches());
        // dummy vectos for the cell loop, not used in the loop body
        dealii::Vector<number> dummy;

        std::function<void(const dealii::MatrixFree<dim, number> &,
                           dealii::Vector<number> &,
                           const dealii::Vector<number> &,
                           const std::pair<unsigned int, unsigned int> &)>
          cell_op = [&](const dealii::MatrixFree<dim, number> &mf,
                        dealii::Vector<number> &,
                        const dealii::Vector<number> &,
                        const std::pair<unsigned int, unsigned int> &cell_range) {
            for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
                 ++cell_batch)
              {
                const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> cell_iterators =
                  cells_in_cell_batch(mf, cell_batch);

                cell_batch_to_particle_cache[cell_batch].clear();
                for (const auto &cell : cell_iterators)
                  {
                    particle_handler.get_obstacles_in_cell(
                      cell, cell_batch_to_particle_cache[cell_batch]);
                  }
              }
          };

        mf_context.mf.cell_loop(cell_op, dummy, dummy);
      }
  }
} // namespace MeltPoolDG

template class MeltPoolDG::
  MatrixFreeCellBatchParticleCache<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::
  MatrixFreeCellBatchParticleCache<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::
  MatrixFreeCellBatchParticleCache<3, double, MeltPoolDG::SphericalParticle<3, double>>;
