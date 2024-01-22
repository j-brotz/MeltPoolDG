#include <deal.II/numerics/data_out_dof_data.h>

#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <filesystem>

namespace MeltPoolDG
{
  template <int dim>
  Postprocessor<dim>::Postprocessor(const MPI_Comm                  mpi_communicator_in,
                                    const ParaviewData<double>     &pv_data_in,
                                    const TimeSteppingData<double> &time_data,
                                    const Mapping<dim>             &mapping_in,
                                    const Triangulation<dim>       &triangulation_in,
                                    const ConditionalOStream       &pcout_in)
    : mpi_communicator(mpi_communicator_in)
    , pv_data(pv_data_in)
    , mapping(mapping_in)
    , triangulation(triangulation_in)
    , pcout(pcout_in)
    , do_simplex(!triangulation.all_reference_cells_are_hyper_cube())
    , end_time(time_data.end_time)
    , time_at_last_output(time_data.start_time)
  {
    if (pv_data.write_time_step_size > 0.0)
      {
        AssertThrow(pv_data.write_time_step_size >= time_data.time_step_size,
                    ExcMessage(
                      "The time step size for writing paraview files must be equal or larger "
                      "than the simulation time step size."));
      }
  }

  /*
   *  This function collects and performs all relevant postprocessing steps.
   */
  template <int dim>
  void
  Postprocessor<dim>::process(const int                    n_time_step,
                              const GenericDataOut<dim>   &data_out,
                              const double                 time,
                              const bool                   force_output,
                              const std::function<void()> &post_operation)
  {
    if (now(n_time_step, time) || force_output)
      {
        Journal::print_line(pcout, "write paraview files", "postprocessor");

        write_paraview_files(n_time_step, time, data_out);

        if (pv_data.print_boundary_id)
          print_boundary_ids();

        // record written output time
        time_at_last_output = time;
      }

    if (post_operation)
      post_operation();
  }

  /*
   *  This function collects and performs all relevant postprocessing steps.
   */
  template <int dim>
  void
  Postprocessor<dim>::process(
    const int                                         n_time_step,
    const std::function<void(GenericDataOut<dim> &)> &attach_output_vectors,
    const double                                      time,
    const std::function<void()>                      &post_operation)
  {
    if (now(n_time_step, time))
      {
        GenericDataOut<dim> data_out(mapping, time, pv_data.output_variables);

        attach_output_vectors(data_out);

        Journal::print_line(pcout, "write paraview files", "postprocessor");

        write_paraview_files(n_time_step, time, data_out);

        if (pv_data.print_boundary_id)
          print_boundary_ids();

        // record written output time
        time_at_last_output = time;
      }

    if (post_operation)
      post_operation();
  }

  template <int dim>
  void
  Postprocessor<dim>::write_paraview_files(const unsigned int         n_time_step,
                                           const double               time,
                                           const GenericDataOut<dim> &generic_data_out)
  {
    namespace fs = std::filesystem;
    DataOut<dim> data_out;

    // do search algorithm only once
    if (idx_req_vars.size() == 0)
      idx_req_vars = generic_data_out.get_indices_data_request(pv_data.output_variables);

    for (const auto &i : idx_req_vars)
      {
        const auto &data = generic_data_out.entries[i];

        data_out.add_data_vector(*std::get<0>(data),
                                 *std::get<1>(data),
                                 std::get<2>(data),
                                 std::get<3>(data));
      }

    if (pv_data.output_subdomains)
      {
        const auto    &tria = std::get<0>(generic_data_out.entries.front())->get_triangulation();
        Vector<double> subdomains(tria.n_active_cells());
        subdomains = Utilities::MPI::this_mpi_process(mpi_communicator);

        data_out.add_data_vector(subdomains, "subdomains");
      }
    if (pv_data.output_material_id)
      {
        const auto    &tria = std::get<0>(generic_data_out.entries.front())->get_triangulation();
        Vector<double> material_id(tria.n_active_cells());

        for (const auto &cell : tria.active_cell_iterators())
          material_id[cell->active_cell_index()] = cell->material_id();

        data_out.add_data_vector(material_id, "material_id", DataOut<dim>::type_cell_data);
      }

    DataOutBase::VtkFlags flags;
    if ((do_simplex == false) && (dim > 1))
      flags.write_higher_order_cells = pv_data.write_higher_order_cells;
    data_out.set_flags(flags);

    unsigned int n_patches = pv_data.n_patches;

    if (n_patches == 0)
      for (const auto &data : generic_data_out.entries)
        n_patches = std::max(n_patches, std::get<0>(data)->get_fe().degree);

    data_out.build_patches(mapping, n_patches);
    const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(pv_data.directory + "/",
                                                                          pv_data.filename,
                                                                          n_time_step,
                                                                          mpi_communicator,
                                                                          pv_data.n_digits_timestep,
                                                                          pv_data.n_groups);

    // write a pvd file relating the pvtu-file to a simulation time
    //
    // if times_and_names is not empty
    const unsigned int len = times_and_names.size();

    // Only append the *.pvtu-file to the *.pvd file, if the to be appended *.pvtu-file has not
    // been written before. This might happen in case of a restart, when the time of the last
    // written simulation output corresponds to the one in the initial state of the restart.
    if (times_and_names.empty() || (times_and_names[len - 1].first != time &&
                                    times_and_names[len - 1].second != pvtu_filename))
      times_and_names.emplace_back(time, pvtu_filename);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && time >= 0.0)
      {
        clean_pvd();

        std::ofstream pvd_output(fs::path(pv_data.directory) / fs::path(pv_data.filename + ".pvd"));
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
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
      return !fs::exists(fs::path(pv_data.directory) / fs::path(x.second));
    });
  }

  template <int dim>
  void
  Postprocessor<dim>::print_boundary_ids()
  {
    const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

    const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

    namespace fs = std::filesystem;

    std::ofstream output(fs::path(pv_data.directory) /
                         fs::path(pv_data.filename + "_boundary_ids" +
                                  Utilities::int_to_string(rank, n_digits) + ".vtk"));

    GridOut           grid_out;
    GridOutFlags::Vtk flags;
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
