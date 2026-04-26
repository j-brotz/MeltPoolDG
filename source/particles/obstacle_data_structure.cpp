#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>

#include <boost/container/small_vector.hpp>

#include <memory>
#include <vector>

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::ObstacleCompleteDomainSearch(
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping,
  dealii::TimerOutput              &timer)
  : obstacle_handler(std::make_unique<dealii::Particles::ParticleHandler<dim>>(
      triangulation,
      mapping,
      ObstacleType::n_obstacle_properties))
  , properties_global_obstacles(
      std::make_unique<dealii::Particles::PropertyPool<dim>>(ObstacleType::n_obstacle_properties))
  , timer(timer)
{}

template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::ObstacleCompleteDomainSearch::
  ~ObstacleCompleteDomainSearch()
{
  deregister_property_pool();
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::reinit()
{
  deregister_property_pool();
  properties_global_obstacles->clear();
  broadcast_global_particles();
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  dealii::TimerOutput::Scope t(timer, "particles in cell search");

  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> handles;
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles->n_registered_slots();
       ++src_handle)
    {
      if (ObstacleType::is_in_cell(*properties_global_obstacles, src_handle, cell))
        {
          auto dst_handle = dst.register_particle();
          handles.emplace_back(dst_handle);
          dst.set_location(dst_handle, properties_global_obstacles->get_location(src_handle));
          auto dst_properties = dst.get_properties(dst_handle);
          auto src_properties = properties_global_obstacles->get_properties(src_handle);

          for (unsigned int n_property = 0; n_property < ObstacleType::n_obstacle_properties;
               ++n_property)
            {
              dst_properties[n_property] = src_properties[n_property];
            }
        }
    }
  return handles;
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim>                               &dst,
  const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const
{
  dealii::TimerOutput::Scope t(timer, "particles in cell search");

  std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> handles;
  for (unsigned int src_handle = 0; src_handle < properties_global_obstacles->n_registered_slots();
       ++src_handle)
    {
      for (const dealii::TriaIterator<dealii::CellAccessor<dim>> &cell : cells)
        {
          if (ObstacleType::is_in_cell(*properties_global_obstacles, src_handle, *cell))
            {
              auto dst_handle = dst.register_particle();
              handles.emplace_back(dst_handle);
              dst.set_location(dst_handle, properties_global_obstacles->get_location(src_handle));
              auto dst_properties = dst.get_properties(dst_handle);
              auto src_properties = properties_global_obstacles->get_properties(src_handle);

              for (unsigned int n_property = 0; n_property < ObstacleType::n_obstacle_properties;
                   ++n_property)
                {
                  dst_properties[n_property] = src_properties[n_property];
                }
              break;
            }
        }
    }
  return handles;
}

template <int dim, typename number, typename ObstacleType>
boost::container::small_vector<MeltPoolDG::DEMParticleAccessor<dim, number>, 3 * dim>
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::contact_particles(
  const DEMParticleAccessor<dim, number> &particle,
  const number                            relative_tolerance) const
{
  dealii::TimerOutput::Scope t(timer, "contact particle search");

  // We assume the max number of contacts per particle to be 3*dim, which is a reasonable
  // assumption for spherical particles including some tolerance offset for the contact
  // distance. This allows us to use a small_vector for efficient storage without dynamic memory
  // allocation in most cases.
  boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim> contacts;

  for (const auto &other : std::ranges::subrange<ParticleIterator<dim, number>>(
         ParticleIterator<dim, number>(*properties_global_obstacles, 0),
         ParticleIterator<dim, number>(*properties_global_obstacles,
                                       properties_global_obstacles->n_registered_slots())))
    {
      if ((particle.get_location() - other.get_location()).norm_square() <
            dealii::Utilities::fixed_power<2>((other.radius() + particle.radius()) *
                                              (1. + relative_tolerance)) and
          particle.id() != other.id())
        contacts.push_back(other);
    }

  return contacts;
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::broadcast_global_particles()
  const
{
  dealii::TimerOutput::Scope t(timer, "global particle broadcast");

  deregister_property_pool();
  properties_global_obstacles->clear();
  using Handle = typename dealii::Particles::PropertyPool<dim>::Handle;
  for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
       ++rank)
    {
      unsigned int rank_local_obstacles = obstacle_handler->n_locally_owned_particles();
      dealii::Utilities::MPI::broadcast(&rank_local_obstacles, 1, rank, mpi_communicator);
      for (unsigned int i = 0; i < rank_local_obstacles; ++i)
        {
          Handle              obstacle_handle = properties_global_obstacles->register_particle();
          dealii::Point<dim>  obstacle_location;
          std::vector<number> obstacle_properties(ObstacleType::n_obstacle_properties);
          if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == rank)
            {
              dealii::Particles::ParticleAccessor<dim> local_obstacle =
                *(std::next(obstacle_handler->begin(), i));
              obstacle_location                          = local_obstacle.get_location();
              dealii::ArrayView<number> local_properties = local_obstacle.get_properties();
              local_properties[ObstacleType::Properties::particle_id] = local_obstacle.get_id();
              for (unsigned int j = 0; j < local_properties.size(); ++j)
                obstacle_properties[j] = local_properties[j];
            }
          // broadcast particle location
          for (int d = 0; d < dim; ++d)
            {
              number location = obstacle_location[d];
              dealii::Utilities::MPI::broadcast(&location, 1, rank, mpi_communicator);
              obstacle_location[d] = location;
            }
          properties_global_obstacles->set_location(obstacle_handle, obstacle_location);
          // broadcast particle properties
          dealii::Utilities::MPI::broadcast(obstacle_properties.data(),
                                            obstacle_properties.size(),
                                            rank,
                                            mpi_communicator);
          dealii::ArrayView<number> local_properties =
            properties_global_obstacles->get_properties(obstacle_handle);
          for (unsigned int j = 0; j < local_properties.size(); ++j)
            local_properties[j] = obstacle_properties[j];
        }
    }
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, ObstacleType>::deregister_property_pool()
  const
{
  // The property pool containing the properties of all global particles is initialized once during
  // the `reinit()` call and remains unchanged thereafter. As a result, we don't need to track
  // individual handles explicitly. We know that handles are assigned sequentially, starting from
  // zero up to the number of registered slots. Therefore, a simple loop is sufficient to deregister
  // all particles before releasing the associated resources.
  if (properties_global_obstacles == nullptr)
    return;

  for (unsigned int i = 0; i < properties_global_obstacles->n_registered_slots(); ++i)
    properties_global_obstacles->deregister_particle(i);
}

template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template struct MeltPoolDG::
  ObstacleCompleteDomainSearch<3, double, MeltPoolDG::SphericalParticle<3, double>>;
