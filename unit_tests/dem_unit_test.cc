#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <memory>
#include <vector>


using namespace MeltPoolDG;

/**
 * Artificial torque class used to apply a torque to the particles that is proportional to their
 * velocity. This is used to test the implementation of torques in the DEM module.
 */
template <int dim, typename number>
class ObstacleArtificialTorque
{
  using ObstacleType = SphericalParticle<dim, number>;

public:
  /**
   * Constructor.
   *
   * @param torque_acceleration_factor Scaling factor for the applied torque.
   */
  ObstacleArtificialTorque(const number torque_acceleration_factor)
    : torque_acceleration_factor(torque_acceleration_factor)
  {}

  /**
   * Compute and apply the torque to all obstacles in the obstacle field. The torque is computed by
   * taking the norm of the translational velocity of the corresponding particle and multiplying it
   * by the `torque_acceleration_factor`.
   *
   * @param obstacle_field The obstacle field to which the torque is applied.
   */
  void
  add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
  {
    for (dealii::Particles::ParticleAccessor<dim> obstacle : obstacle_field.get_particle_handler())
      {
        if constexpr (dim == 2)
          {
            ObstacleType::accumulate_torque(
              dealii::Tensor<1, ObstacleType::size_angular_velocity, number>(
                {torque_acceleration_factor *
                 ObstacleType::template get_velocity<number>(obstacle).norm()}),
              obstacle);
          }
        else if constexpr (dim == 3)
          {
            ObstacleType::accumulate_torque(
              dealii::Tensor<1, ObstacleType::size_angular_velocity, number>(
                torque_acceleration_factor * ObstacleType::template get_velocity<number>(obstacle)),
              obstacle);
          }
      }
  }

private:
  /// Scaling factor for the applied torque.
  const number torque_acceleration_factor;
};

template <int dim, typename number>
class ObstacleSchillerNaumannFluidForce
{
  using ObstacleType = SphericalParticle<dim, number>;

public:
  /**
   * Constructor. Stores the fluid density and dynamic viscosity internally.
   */
  ObstacleSchillerNaumannFluidForce(const number fluid_density, const number dynamic_viscosity)
    : fluid_density(fluid_density)
    , dynamic_viscosity(dynamic_viscosity)
  {}

  /**
   * Applies the Schiller-Naumann drag force to all obstacles in the obstacle field. The force is
   * computed based on the obstacle's velocity, radius, and the fluid properties (density and
   * dynamic viscosity).
   *
   * Based on the obstacles velocity the Reynolds number is computed
   * /[
   * Re = \frac{\rho_f ||u_{obstacle}||  d}{\mu_f}.
   * /]
   * This Reynolds number can be inserted into the Schiller-Naumann drag coefficient formula
   * /[
   * C_d = \frac{24}{Re} + \frac{3.6}{Re^{0.313}},
   * /]
   * and finally we obtain the drag force by computing
   * /[
   * F_d = -\frac{1}{2} C_d \rho_f A_{proj} ||u_{obstacle}|| u_{obstacle}.
   * /]
   *
   * @param obstacle_field The obstacle field to which the fluid force is applied.
   */
  void
  add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
  {
    for (dealii::Particles::ParticleAccessor<dim> obstacle : obstacle_field.get_particle_handler())
      {
        if (ObstacleType::template get_velocity<number>(obstacle).norm() == 0)
          return;

        const number re = fluid_density / dynamic_viscosity *
                          ObstacleType::template get_velocity<number>(obstacle).norm() * 2. *
                          ObstacleType::get_property(obstacle, ObstacleType::Properties::radius);
        const number drag_coefficient = 24. / re + 3.6 / std::pow(re, 0.313);
        const number a_proj =
          dim == 2 ?
            2 * ObstacleType::get_property(obstacle, ObstacleType::Properties::radius) :
            M_PI *
              std::pow(ObstacleType::get_property(obstacle, ObstacleType::Properties::radius), 2);

        const number abs_force =
          0.5 * drag_coefficient * fluid_density *
          ObstacleType::template get_velocity<number>(obstacle).norm_square() * a_proj;

        dealii::Tensor<1, dim, number> force =
          -abs_force / ObstacleType::template get_velocity<number>(obstacle).norm() *
          ObstacleType::template get_velocity<number>(obstacle);
        ObstacleType::accumulate_force(force, obstacle);
      }
  }

private:
  const number fluid_density;
  const number dynamic_viscosity;
};

/**
 * The test class. It performs all the computations used for testing the DEM module. The test case
 * describes two spheres being in a fluid at rest and under the influence of gravity. The spheres
 * are initialized with different vertical velocities and different radii. In addition to gravity,
 * the fluid force according to Schiller-Naumann and an artificial torque is applied to the
 * particles. The time evolution of the system is printed to the terminal and used for testing.
 */
template <int dim, typename number, typename ObstacleType>
class DEMTest
{
public:
  /**
   * Constructor. Initializes the parallel output stream.
   */
  DEMTest()
    : pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}

  /**
   * Print information about all obstacles in the obstacle field to the terminal. This includes
   * current vertical position, velocity and norm of the angular velocity.
   */
  void
  print_obstacle_field_information()
  {
    // Step 1: Collect local data on all processes
    constexpr unsigned  root = 0;
    std::vector<number> locations;
    std::vector<number> velocities;
    std::vector<number> angular_velocities;
    for (const auto &particle : obstacle_field->get_particle_handler())
      {
        locations.push_back(particle.get_location()[dim - 1]);
        velocities.push_back(ObstacleType::template get_velocity<number>(particle)[dim - 1]);
        angular_velocities.push_back(ObstacleType::get_angular_velocity(particle).norm());
      }

    // Step 2: Gather all data on root (printing) process
    auto global_locations  = dealii::Utilities::MPI::gather(MPI_COMM_WORLD, locations, root);
    auto global_velocities = dealii::Utilities::MPI::gather(MPI_COMM_WORLD, velocities, root);
    auto global_angular_velocities =
      dealii::Utilities::MPI::gather(MPI_COMM_WORLD, angular_velocities, root);

    // Step 3: Print all data on root process
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == root)
      {
        unsigned particle_count = 0;
        for (unsigned i = 0; i < global_locations.size(); ++i)
          for (unsigned j = 0; j < global_locations[i].size(); ++j)
            {
              std::ostringstream str;
              str << "Particle #" << particle_count << ": " << std::setw(18) << std::right
                  << "Vertical position: " << global_locations[i][j] << ", "
                  << "Vertical velocity: " << global_velocities[i][j] << ", "
                  << "Norm omega: " << global_angular_velocities[i][j];
              Journal::print_line(pcout, str.str(), "");
              ++particle_count;
            }
      }
  }

  /**
   * Sets up the triangulation and mapping for the simulation domain. The triangulation is a
   * subdivided hyper-rectangle, and the mapping is a linear mapping.
   */
  void
  setup_triangulation_and_mapping()
  {
    this->triangulation =
      std::make_unique<dealii::parallel::distributed::Triangulation<dim>>(MPI_COMM_WORLD);

    std::vector<unsigned int>  repetitions;
    dealii::Point<dim, number> p1;
    dealii::Point<dim, number> p2;
    if constexpr (dim == 2)
      {
        repetitions = {6, 16};
        p1          = dealii::Point<2, number>(0, 0);
        p2          = dealii::Point<2, number>(3, 8);
      }
    else if constexpr (dim == 3)
      {
        repetitions = {6, 6, 16};
        p1          = dealii::Point<3, number>(0, 0, 0);
        p2          = dealii::Point<3, number>(3, 3, 8);
      }

    dealii::GridGenerator::subdivided_hyper_rectangle(
      *this->triangulation, repetitions, p1, p2, true);

    constexpr unsigned mapping_fe_degree = 1;
    mapping = std::make_unique<dealii::MappingQGeneric<dim>>(mapping_fe_degree);
  }

  /**
   * Sets up the obstacle field with two spherical particles. The particles are initialized with
   * different initial vertical velocities, radii, and densities. The obstacle field is also
   * configured to include gravitational forces, Schiller-Naumann fluid forces, and an artificial
   * torque.
   */
  void
  setup_obstacle_field()
  {
    auto make_particle_property = [](const number initial_vertical_velocity,
                                     const number radius,
                                     const number density) -> std::vector<number> {
      std::vector<number> properties(ObstacleType::n_obstacle_properties, number(0.0));
      properties[ObstacleType::Properties::velocity + dim - 1] = initial_vertical_velocity;
      properties[ObstacleType::Properties::radius]             = radius;
      properties[ObstacleType::Properties::density]            = density;
      if constexpr (dim == 3)
        {
          properties[ObstacleType::Properties::volume] =
            4.0 / 3.0 * M_PI * std::pow(properties[ObstacleType::Properties::radius], 3);
          properties[ObstacleType::Properties::mass] =
            properties[ObstacleType::Properties::volume] *
            properties[ObstacleType::Properties::density];
          properties[ObstacleType::Properties::moment_of_inertia] =
            0.4 * properties[ObstacleType::Properties::mass] *
            std::pow(properties[ObstacleType::Properties::radius], 2);
        }
      else if constexpr (dim == 2)
        {
          properties[ObstacleType::Properties::volume] =
            M_PI * std::pow(properties[ObstacleType::Properties::radius], 2);
          properties[ObstacleType::Properties::mass] =
            properties[ObstacleType::Properties::volume] *
            properties[ObstacleType::Properties::density];
          properties[ObstacleType::Properties::moment_of_inertia] =
            0.5 * properties[ObstacleType::Properties::mass] *
            std::pow(properties[ObstacleType::Properties::radius], 2);
        }
      return properties;
    };


    std::vector<dealii::Point<dim>> particle_locations;
    particle_locations.reserve(2);
    if constexpr (dim == 2)
      {
        particle_locations.emplace_back(0.7, 7.);
        particle_locations.emplace_back(1.53, 7.);
      }
    else if constexpr (dim == 3)
      {
        particle_locations.emplace_back(0.7, 0.8, 7.);
        particle_locations.emplace_back(1.43, 1.3, 6.);
      }
    else
      {
        AssertThrow(false, dealii::ExcInternalError());
      }

    std::vector<std::vector<number>> particle_properties;
    particle_properties.reserve(2);
    particle_properties.emplace_back(make_particle_property(-0.2, 0.5, 1));
    particle_properties.emplace_back(make_particle_property(0.0, 0.3, 1.72));

    obstacle_field = std::make_unique<ObstacleField<dim, number, ObstacleType>>(
      obstacle_data, *triangulation, *mapping, particle_locations, particle_properties);

    constexpr number gravitational_acceleration = 10.0;
    obstacle_field->add_load_type(
      ObstacleGravitationalForce<dim, number, ObstacleType>(gravitational_acceleration));

    constexpr number fluid_density     = 870.;
    constexpr number dynamic_viscosity = 0.4;
    obstacle_field->add_load_type(
      ObstacleSchillerNaumannFluidForce<dim, number>(fluid_density, dynamic_viscosity));

    constexpr number artificial_torque_scaling_factor = 1.;
    obstacle_field->add_load_type(
      ObstacleArtificialTorque<dim, number>(artificial_torque_scaling_factor));
  }

  /**
   * Sets up the time iterator for the simulation with specified start time, end time, and time step
   * size.
   */
  void
  setup_time_iterator()
  {
    TimeIntegration::TimeSteppingData<number> data;
    data.start_time     = 0.0;
    data.end_time       = 0.05;
    data.time_step_size = 0.006;
    time_iterator       = std::make_unique<MeltPoolDG::TimeIntegration::TimeIterator<number>>(data);
  }

  /**
   * The main function that runs the DEM test. It sets up the time iterator, triangulation, mapping,
   * and obstacle field. It then enters a time-stepping loop where it advances the simulation
   * through time, updating the obstacle field and printing information about the obstacles at each
   * time step.
   */
  void
  run_test()
  {
    setup_time_iterator();

    setup_triangulation_and_mapping();

    setup_obstacle_field();

    print_obstacle_field_information();

    while (not time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(pcout);

        obstacle_field->advance_time(time_iterator->get_current_time(),
                                     time_iterator->get_current_time_increment());

        print_obstacle_field_information();
      }

    Journal::print_end(pcout);
  }

private:
  std::unique_ptr<dealii::Triangulation<dim>>                           triangulation;
  std::unique_ptr<dealii::Mapping<dim>>                                 mapping;
  MeltPoolDG::ObstacleData<number>                                      obstacle_data;
  std::unique_ptr<MeltPoolDG::ObstacleField<dim, number, ObstacleType>> obstacle_field;
  std::unique_ptr<MeltPoolDG::TimeIntegration::TimeIterator<number>>    time_iterator;
  dealii::ConditionalOStream                                            pcout;
};

int
main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  dealii::ConditionalOStream               pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  Journal::print_header(pcout, "DEM Unit Test 2D");
  DEMTest<2, double, MeltPoolDG::SphericalParticle<2, double>> dem_test_2d;
  dem_test_2d.run_test();

  pcout << std::endl;
  Journal::print_header(pcout, "DEM Unit Test 3D");
  DEMTest<3, double, MeltPoolDG::SphericalParticle<3, double>> dem_test_3d;
  dem_test_3d.run_test();

  return 0;
}
