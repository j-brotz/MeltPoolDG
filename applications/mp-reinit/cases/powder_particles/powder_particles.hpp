#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include "meltpooldg/level_set/level_set_data.hpp"
#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/powder_bed.hpp>

#include <string>
#include <vector>

#include "../../reinitialization_case.hpp"


namespace MeltPoolDG::Simulation::PowderParticles
{
  BETTER_ENUM(ParticleCase, char, two_particles, powder_bed)

  template <int dim, typename number>
  class SimulationPowderParticles : public LevelSet::ReinitializationCase<dim, number>
  {
  private:
    /// domain specific
    number domain_x_min = 0;
    number domain_x_max = 0;
    number domain_y_min = 0;
    number domain_y_max = 0;
    number domain_z_min = 0;
    number domain_z_max = 0;

    std::vector<unsigned int> cell_repetitions;

    /// case of particle distribution
    ParticleCase particle_case = ParticleCase::two_particles;

    /// two_particle specific parameters
    number two_partile_radius   = 15e-6;
    number two_partile_distance = 40e-6;

    /// powder_bed specific parameters
    MeltPool::PowderBedData<number> powder_bed_data;

    LevelSet::LevelSetType level_set_type = LevelSet::LevelSetType::tanh;

  public:
    SimulationPowderParticles(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() override;

    void
    set_field_conditions() override;
  };
} // namespace MeltPoolDG::Simulation::PowderParticles
