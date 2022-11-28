#pragma once
#include <deal.II/base/table_handler.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>


using namespace dealii;

// @ todo: !!! clean-up and refactoring !!!

namespace MeltPoolDG
{
  template <int dim>
  class Postprocessor
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const MPI_Comm              mpi_communicator;
    const ParaviewData<double> &pv_data;
    const Mapping<dim> &        mapping;
    const Triangulation<dim> &  triangulation;
    const ConditionalOStream    pcout;
    const bool                  do_simplex;
    const double                end_time;
    double                      time_at_last_output = 0.0;

    // list of indices for the requested variables
    std::vector<unsigned int> idx_req_vars;


    std::vector<std::pair<double, std::string>> times_and_names;

  public:
    Postprocessor(const MPI_Comm                  mpi_communicator_in,
                  const ParaviewData<double> &    pv_data_in,
                  const TimeSteppingData<double> &time_data,
                  const Mapping<dim> &            mapping_in,
                  const Triangulation<dim> &      triangulation_in,
                  const ConditionalOStream &      pcout_in);

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                    n_time_step,
            const GenericDataOut<dim> &  generic_data_out,
            const double                 time           = -1.0,
            const bool                   force_output   = false,
            const std::function<void()> &post_operation = {});

    /**
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                                         n_time_step,
            const std::function<void(GenericDataOut<dim> &)> &attach_output_vectors,
            const double                                      time           = -1.0,
            const std::function<void()> &                     post_operation = {});

    /**
     * Determines whether postprocessing should be performed now.
     */
    inline bool
    now(const int n_time_step, const double time)
    {
      return (!pv_data.do_output) ? false :
             (n_time_step == 0) || (std::abs(time - end_time) <= 1e-10) ?
                                    true :
             (time - time_at_last_output >= pv_data.write_time_step_size) ?
                                    true :
                                    !(n_time_step % pv_data.write_frequency);
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
                         const GenericDataOut<dim> &generic_data_out);

    void
    print_boundary_ids();

    /*
     * @todo
     */
    void
    compute_error(const int              n_q_points,
                  const VectorType &     approximate_solution,
                  const Function<dim> &  ExactSolution,
                  const DoFHandler<dim> &dof_handler,
                  const Mapping<dim> &   mapping);

    std::vector<double>
    compute_volume_of_phases(const int              degree,
                             const int              n_q_points,
                             const DoFHandler<dim> &dof_handler,
                             const VectorType &     solution_levelset,
                             const double           time,
                             const MPI_Comm &       mpi_communicator,
                             TableHandler &         volume_table,
                             const double           max_value = 1,
                             const double           min_value = -1.0);
  };
} // namespace MeltPoolDG
