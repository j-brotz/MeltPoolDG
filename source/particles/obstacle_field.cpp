
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <fstream>


template <int dim, typename number, typename ObstacleType>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::ObstacleField(
  const ObstacleData               &data,
  const dealii::Triangulation<dim> &triangulation,
  const dealii::Mapping<dim>       &mapping)
  : data(data)
  , obstacle_handler(triangulation, mapping, ObstacleType::n_obstacle_properties)
  , obstacle_data_structure(obstacle_handler)
  , mpi_communicator(triangulation.get_mpi_communicator())
{
  read_particle_state_input_file(triangulation);
  obstacle_data_structure.reinit();
  AssertThrow(data.stationary_obstacles == true,
              dealii::ExcMessage("Currently only stationary particles are supported!"));
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_obstacles_in_cell(
  dealii::Particles::PropertyPool<dim> &dst,
  const dealii::CellAccessor<dim>      &cell) const
{
  return obstacle_data_structure.get_obstacles_in_cell(dst, cell);
}

template <int dim, typename number, typename ObstacleType>
std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::get_obstacles_in_cell_batch(
  dealii::Particles::PropertyPool<dim>  &dst,
  const dealii::MatrixFree<dim, number> &matrix_free,
  const unsigned int                     cell_batch_id) const
{
  return obstacle_data_structure.get_obstacles_in_cell_batch(dst, matrix_free, cell_batch_id);
}


template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::compute_forces_on_obstacles()
{
  if (forces.empty())
    return;

  // Update global particle property pool in case any of the forces needs an actual version.
  obstacle_data_structure.broadcast_global_particles();

  // Reset current particle forces
  for (dealii::Particles::ParticleAccessor<dim> obstacle : obstacle_handler)
    ObstacleType::set_force(dealii::Tensor<1, dim, number>(), obstacle);

  // Accumulate all forces acting on the particles
  for (const auto &force_type : forces)
    force_type.add_force_to_obstacles(*this);
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::print_accumulated_obstacle_force_norm(
  const dealii::ConditionalOStream pout) const
{
  dealii::Tensor<1, dim, number> accumulated_force;
  for (dealii::Particles::ParticleAccessor<dim> obstacle : obstacle_handler)
    accumulated_force += ObstacleType::get_force(obstacle);

  accumulated_force = dealii::Utilities::MPI::sum(accumulated_force, mpi_communicator);

  std::ostringstream output;
  output << std::scientific << std::setprecision(4)
         << "accumulated norm: " << accumulated_force.norm();
  Journal::print_line(pout, output.str(), "obstacle_force");
}

template <int dim, typename number, typename ObstacleType>
void
MeltPoolDG::ObstacleField<dim, number, ObstacleType>::read_particle_state_input_file(
  const dealii::Triangulation<dim> &triangulation)
{
  std::vector<dealii::Point<dim>> particle_locations;
  // Make global particle properties vector
  std::vector<std::vector<number>> properties{};
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::fstream file;
      file.open(data.obstacle_state_input_file, std::ios::in);
      AssertThrow(!(file.fail()),
                  dealii::ExcMessage("Unable to open particle data file \"" +
                                     data.obstacle_state_input_file + "\". Aborting!"));
      std::string line;
      std::getline(file, line); // Ignore the first line
      while (std::getline(file, line))
        {
          auto obstacle_properties = ObstacleType::read_state_input(line);
          properties.push_back(obstacle_properties.first);
          particle_locations.push_back(obstacle_properties.second);
        }
    }

  // Make the particles from the data in @struct particle_data
  std::vector<dealii::BoundingBox<dim>> local_bounding_box =
    dealii::GridTools::compute_mesh_predicate_bounding_box(
      triangulation, dealii::IteratorFilters::LocallyOwnedCell());
  std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_box =
    dealii::Utilities::MPI::all_gather(mpi_communicator, local_bounding_box);

  obstacle_handler.insert_global_particles(dealii::Utilities::MPI::this_mpi_process(
                                             mpi_communicator) == 0 ?
                                             particle_locations :
                                             std::vector<dealii::Point<dim, number>>{},
                                           global_bounding_box,
                                           properties);
}

template class MeltPoolDG::ObstacleField<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::ObstacleField<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::ObstacleField<3, double, MeltPoolDG::SphericalParticle<3, double>>;
