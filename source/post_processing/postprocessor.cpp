#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

namespace MeltPoolDG
{
  template <int dim>
  Postprocessor<dim>::Postprocessor(const MPI_Comm                  mpi_communicator_in,
                                    const OutputData<double>       &output_data_in,
                                    const TimeSteppingData<double> &time_data,
                                    const Mapping<dim>             &mapping_in,
                                    const Triangulation<dim>       &triangulation_in,
                                    const ConditionalOStream       &pcout_in)
    : mpi_communicator(mpi_communicator_in)
    , output_data(output_data_in)
    , mapping(mapping_in)
    , triangulation(triangulation_in)
    , pcout(pcout_in)
    , do_simplex(!triangulation.all_reference_cells_are_hyper_cube())
    , end_time(time_data.end_time)
    , time_at_last_output(time_data.start_time)
  {
    if (output_data.write_time_step_size > 0.0)
      {
        AssertThrow(output_data.write_time_step_size >= time_data.time_step_size,
                    dealii::ExcMessage(
                      "The time step size for writing output files must be equal or larger "
                      "than the simulation time step size."));
      }
  }

  /*
   *  This function collects and performs all relevant postprocessing steps.
   */
  template <int dim>
  void
  Postprocessor<dim>::process(const int                  n_time_step,
                              const GenericDataOut<dim> &data_out,
                              const double               time,
                              const bool                 force_output,
                              const bool                 force_update_requested_output_variables)
  {
    if (not(is_output_timestep(n_time_step, time) || force_output))
      return;

    if (output_data.paraview.enable)
      {
        write_paraview_files(n_time_step, time, data_out, force_update_requested_output_variables);

        if (output_data.paraview.print_boundary_id)
          print_boundary_ids();

        // record written output time
        time_at_last_output = time;
      }
  }

  /*
   *  This function collects and performs all relevant postprocessing steps.
   */
  template <int dim>
  void
  Postprocessor<dim>::process(
    const int                                         n_time_step,
    const std::function<void(GenericDataOut<dim> &)> &attach_output_vectors,
    const double                                      time)
  {
    if (not is_output_timestep(n_time_step, time))
      return;

    if (output_data.paraview.enable)
      {
        GenericDataOut<dim> data_out(mapping, time, output_data.output_variables);

        attach_output_vectors(data_out);

        write_paraview_files(n_time_step, time, data_out);

        if (output_data.paraview.print_boundary_id)
          print_boundary_ids();

        // record written output time
        time_at_last_output = time;
      }
  }

  template <int dim>
  void
  Postprocessor<dim>::write_paraview_files(const unsigned int         n_time_step,
                                           const double               time,
                                           const GenericDataOut<dim> &generic_data_out,
                                           const bool force_update_requested_output_variables)
  {
    Journal::print_line(pcout, "write paraview files", "postprocessor");

    namespace fs = std::filesystem;
    dealii::DataOut<dim> data_out;

    // do search algorithm only once
    if (idx_req_vars.size() == 0 || force_update_requested_output_variables)
      idx_req_vars = generic_data_out.get_indices_data_request(output_data.output_variables);

    for (const auto &i : idx_req_vars)
      {
        const auto &data = generic_data_out.entries[i];

        data_out.add_data_vector(*std::get<0>(data),
                                 *std::get<1>(data),
                                 std::get<2>(data),
                                 std::get<3>(data));
      }

    if (output_data.paraview.output_subdomains)
      {
        const auto &tria = std::get<0>(generic_data_out.entries.front())->get_triangulation();
        dealii::Vector<double> subdomains(tria.n_active_cells());
        subdomains = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

        data_out.add_data_vector(subdomains, "subdomains");
      }
    if (output_data.paraview.output_material_id)
      {
        const auto &tria = std::get<0>(generic_data_out.entries.front())->get_triangulation();
        dealii::Vector<double> material_id(tria.n_active_cells());

        for (const auto &cell : tria.active_cell_iterators())
          material_id[cell->active_cell_index()] = cell->material_id();

        data_out.add_data_vector(material_id, "material_id", DataOut<dim>::type_cell_data);
      }

    dealii::DataOutBase::VtkFlags flags;
    if ((do_simplex == false) && (dim > 1))
      flags.write_higher_order_cells = output_data.paraview.write_higher_order_cells;
    data_out.set_flags(flags);

    unsigned int n_patches = output_data.paraview.n_patches;

    if (n_patches == 0)
      for (const auto &data : generic_data_out.entries)
        n_patches = std::max(n_patches, std::get<0>(data)->get_fe().degree);

    data_out.build_patches(mapping, n_patches);

    std::string pvtu_filename;
    if (output_data.paraview.n_groups == 1)
      {
        pvtu_filename = fs::path(
          output_data.paraview.filename + "_" +
          Utilities::int_to_string(n_time_step, output_data.paraview.n_digits_timestep) + ".vtu");

        data_out.write_vtu_in_parallel(fs::path(output_data.directory) / pvtu_filename,
                                       mpi_communicator);
      }
    else
      {
        pvtu_filename = data_out.write_vtu_with_pvtu_record(output_data.directory + "/",
                                                            output_data.paraview.filename,
                                                            n_time_step,
                                                            mpi_communicator,
                                                            output_data.paraview.n_digits_timestep,
                                                            output_data.paraview.n_groups);
      }

    // write a pvd file relating the pvtu-file to a simulation time
    // if times_and_names is not empty
    const unsigned int len = times_and_names.size();

    // Only append the *.pvtu-file to the *.pvd file, if the to be appended *.pvtu-file has not
    // been written before. This might happen in case of a restart, when the time of the last
    // written simulation output corresponds to the one in the initial state of the restart.
    if (times_and_names.empty() || (times_and_names[len - 1].first != time &&
                                    times_and_names[len - 1].second != pvtu_filename))
      times_and_names.emplace_back(time, pvtu_filename);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && time >= 0.0)
      {
        clean_pvd();

        std::ofstream pvd_output(fs::path(output_data.directory) /
                                 fs::path(output_data.paraview.filename + ".pvd"));
        dealii::DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  }

  template <int dim>
  void
  Postprocessor<dim>::clean_pvd()
  {
    namespace fs = std::filesystem;

    // Clean non-existing *.pvtu-files from *.pvd.
    //
    // This is used if we perform a restart and write the output to a different directory
    // compared to the original simulation.
    //
    std::erase_if(times_and_names, [this](const std::pair<double, std::string> &x) {
      return !fs::exists(fs::path(output_data.directory) / fs::path(x.second));
    });
  }

  template <int dim>
  void
  Postprocessor<dim>::print_boundary_ids()
  {
    const unsigned int rank    = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

    namespace fs = std::filesystem;

    std::ofstream output(fs::path(output_data.directory) /
                         fs::path(output_data.paraview.filename + "_boundary_ids" +
                                  dealii::Utilities::int_to_string(rank, n_digits) + ".vtk"));

    dealii::GridOut           grid_out;
    dealii::GridOutFlags::Vtk flags;
    flags.output_cells         = false;
    flags.output_faces         = true;
    flags.output_edges         = false;
    flags.output_only_relevant = false;
    grid_out.set_flags(flags);
    grid_out.write_vtk(triangulation, output);
  }

  template class Postprocessor<1>;
  template class Postprocessor<2>;
  template class Postprocessor<3>;
} // namespace MeltPoolDG
