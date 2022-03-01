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

#include <functional>
#include <string>
#include <utility>
#include <vector>


using namespace dealii;

// @ todo: !!! clean-up and refactoring !!!

namespace MeltPoolDG
{
  template <int dim>
  class Postprocessor
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const MPI_Comm            mpi_communicator;
    const OutputData<double> &output_data;
    const Mapping<dim>       &mapping;
    const Triangulation<dim> &triangulation;
    const ConditionalOStream  pcout;
    const bool                do_simplex;
    const double              end_time;
    double                    time_at_last_output = 0.0;

    // list of indices for the requested variables
    std::vector<unsigned int> idx_req_vars;


    std::vector<std::pair<double, std::string>> times_and_names;

  public:
    Postprocessor(const MPI_Comm                  mpi_communicator_in,
                  const OutputData<double>       &output_data_in,
                  const TimeSteppingData<double> &time_data,
                  const Mapping<dim>             &mapping_in,
                  const Triangulation<dim>       &triangulation_in,
                  const ConditionalOStream       &pcout_in);

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                  n_time_step,
            const GenericDataOut<dim> &generic_data_out,
            const double               time                                    = -1.0,
            const bool                 force_output                            = false,
            const bool                 force_update_requested_output_variables = false);

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                                         n_time_step,
            const std::function<void(GenericDataOut<dim> &)> &attach_output_vectors,
            const double                                      time = -1.0);

    /**
     * Determines whether postprocessing should be performed now.
     */
    inline bool
    is_output_timestep(const int n_time_step, const double time) const
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
    write_paraview_files(const unsigned int         n_time_step,
                         const double               time,
                         const GenericDataOut<dim> &generic_data_out,
                         const bool force_update_requested_output_variables = false);

    void
    print_boundary_ids();
  };
} // namespace MeltPoolDG
