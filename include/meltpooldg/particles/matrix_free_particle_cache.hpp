#pragma once

#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <functional>
#include <utility>

namespace MeltPoolDG::Particles
{
  template <int dim, typename number>
  struct MatrixFreeCellBatchParticleCache
  {
  public:
    MatrixFreeCellBatchParticleCache(const MatrixFreeContext<dim, number> mf_context)
      : mf_context(mf_context)
    {}

    void
    update(const ObstacleField<dim, number> &obstacle_field)
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
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              const boost::container::small_vector<dealii::TriaIterator<dealii::CellAccessor<dim>>,
                                                   dealii::VectorizedArray<number>::size()>
                cell_iterators = cells_in_cell_batch(mf, cell);

              obstacle_field.get_obstacles_in_cell(cell, cell_batch_to_particle_cache[cell]);
            }
        };

      mf_context.mf.cell_loop(cell_op, dummy, dummy);
    }

    const std::vector<DEMParticleAccessor<dim, number>> &
    obstacles_in_cell_batch(const unsigned int cell_batch_id) const
    {
      return cell_batch_to_particle_cache[cell_batch_id];
    }

    std::vector<DEMParticleAccessor<dim, number>> &
    obstacles_in_cell_batch(const unsigned int cell_batch_id)
    {
      return cell_batch_to_particle_cache[cell_batch_id];
    }

  private:
    const MatrixFreeContext<dim, number> mf_context;

    /// Vector caching, for each cell batch, the particles relevant to the cells in that batch. The
    /// index of the outer vector corresponds to the cell batch id.
    std::vector<std::vector<DEMParticleAccessor<dim, number>>> cell_batch_to_particle_cache;
  };
} // namespace MeltPoolDG::Particles