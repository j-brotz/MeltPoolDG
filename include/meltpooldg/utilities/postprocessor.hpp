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
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
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

    //@todo: delete
    std::vector<std::vector<double>> volumes;
    TableHandler                     volume_table;

    const MPI_Comm              mpi_communicator;
    const ParaviewData<double> &pv_data;
    const Mapping<dim> &        mapping;
    const Triangulation<dim> &  triangulation;
    const ConditionalOStream    pcout;
    bool                        do_simplex;
    std::vector<unsigned int>   idx_req_vars;

    std::vector<std::pair<double, std::string>> times_and_names;

  public:
    Postprocessor(const MPI_Comm              mpi_communicator_in,
                  const ParaviewData<double> &pv_data_in,
                  const Mapping<dim> &        mapping_in,
                  const Triangulation<dim> &  triangulation_in,
                  const ConditionalOStream &  pcout_in);

    /*
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                                         n_time_step,
            const std::function<void(GenericDataOut<dim> &)> &attach_output_vectors,
            const double                                      time           = -1.0,
            const std::function<void()> &                     post_operation = {});

  private:
    void
    write_paraview_files(const unsigned int   n_time_step,
                         const double         time,
                         GenericDataOut<dim> &generic_data_out);

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
                             const double           max_value = 1,
                             const double           min_value = -1.0);

    void
    collect_volume_fraction(const std::vector<double> &volume_fraction);

    void
    print_volume_fraction_table(const MPI_Comm &mpi_communicator, const std::string filename);
  };
} // namespace MeltPoolDG
