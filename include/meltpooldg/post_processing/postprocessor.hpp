#pragma once
#include <deal.II/base/mpi.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <string>
#include <utility>
#include <vector>


// @ todo: !!! clean-up and refactoring !!!

namespace MeltPoolDG
{
  template <int dim, typename number>
  class Postprocessor
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    const MPI_Comm                    mpi_communicator;
    const OutputData<number>         &output_data;
    const dealii::Mapping<dim>       &mapping;
    const dealii::Triangulation<dim> &triangulation;
    const ConditionalOStream          pcout;
    const bool                        do_simplex;
    const number                      end_time;
    number                            time_at_last_output = 0.0;

    // list of indices for the requested variables
    std::vector<unsigned int> idx_req_vars;


    std::vector<std::pair<number, std::string>> times_and_names;

  public:
    Postprocessor(const MPI_Comm                                   mpi_communicator_in,
                  const OutputData<number>                        &output_data_in,
                  const TimeIntegration::TimeSteppingData<number> &time_data,
                  const dealii::Mapping<dim>                      &mapping_in,
                  const dealii::Triangulation<dim>                &triangulation_in,
                  const ConditionalOStream                        &pcout_in);

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                          n_time_step,
            const GenericDataOut<dim, number> &generic_data_out,
            const number                       time                                    = -1.0,
            const bool                         force_output                            = false,
            const bool                         force_update_requested_output_variables = false);

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                                                 n_time_step,
            const std::function<void(GenericDataOut<dim, number> &)> &attach_output_vectors,
            const number                                              time = -1.0);

    /**
     * Determines whether postprocessing should be performed now.
     */
    inline bool
    is_output_timestep(const int n_time_step, const number time) const
    {
      if (n_time_step == 0)
        return true;
      if (std::abs(time - end_time) <= 1e-10)
        return true;
      if (time - time_at_last_output >= output_data.write_time_step_size)
        return true;
      return !(n_time_step % output_data.write_frequency);
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &times_and_names;
      ar &time_at_last_output;
    }

  private:
    void
    clean_pvd();

    void
    write_paraview_files(const unsigned int                 n_time_step,
                         const number                       time,
                         const GenericDataOut<dim, number> &generic_data_out,
                         const bool force_update_requested_output_variables = false);

    void
    print_boundary_ids();
  };
} // namespace MeltPoolDG
