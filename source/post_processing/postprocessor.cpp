#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG
{
  template <int dim>
  Postprocessor<dim>::Postprocessor(const MPI_Comm                  mpi_communicator_in,
                                    const ParaviewData<double> &    pv_data_in,
                                    const TimeSteppingData<double> &time_data,
                                    const Mapping<dim> &            mapping_in,
                                    const Triangulation<dim> &      triangulation_in,
                                    const ConditionalOStream &      pcout_in)
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
                              const GenericDataOut<dim> &  data_out,
                              const double                 time,
                              const std::function<void()> &post_operation)
  {
    if (now(n_time_step, time))
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
    const std::function<void()> &                     post_operation)
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
        const auto &   tria = std::get<0>(generic_data_out.entries.front())->get_triangulation();
        Vector<double> subdomains(tria.n_active_cells());
        subdomains = Utilities::MPI::this_mpi_process(mpi_communicator);

        data_out.add_data_vector(subdomains, "subdomains");
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

    // write a pvd file relating the pvtu-file with a simulation time
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 && time >= 0.0)
      {
        times_and_names.emplace_back(time, pvtu_filename);
        std::ofstream pvd_output(pv_data.directory + "/" + pv_data.filename + ".pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  }

  template <int dim>
  void
  Postprocessor<dim>::print_boundary_ids()
  {
    const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

    const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

    std::string filename =
      pv_data.filename + "_boundary_ids" + Utilities::int_to_string(rank, n_digits) + ".vtk";
    std::ofstream output(filename.c_str());

    GridOut           grid_out;
    GridOutFlags::Vtk flags;
    flags.output_cells         = false;
    flags.output_faces         = true;
    flags.output_edges         = false;
    flags.output_only_relevant = false;
    grid_out.set_flags(flags);
    grid_out.write_vtk(triangulation, output);
  }

  /*
   * @todo
   */

  template <int dim>
  void
  Postprocessor<dim>::compute_error(const int              n_q_points,
                                    const VectorType &     approximate_solution,
                                    const Function<dim> &  ExactSolution,
                                    const DoFHandler<dim> &dof_handler,
                                    const Mapping<dim> &   mapping)
  {
    const auto &triangulation = dof_handler.get_triangulation();

    const QGauss<dim> quadrature(n_q_points);
    Vector<double>    norm_per_cell(triangulation.n_active_cells());

    dealii::VectorTools::integrate_difference(mapping,
                                              dof_handler,
                                              approximate_solution,
                                              ExactSolution,
                                              norm_per_cell,
                                              quadrature,
                                              dealii::VectorTools::L2_norm);

    pcout << "L2 error =    " << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
          << compute_global_error(triangulation, norm_per_cell, dealii::VectorTools::L2_norm)
          << std::endl;

    Vector<double> difference_per_cell(triangulation.n_active_cells());

    dealii::VectorTools::integrate_difference(mapping,
                                              dof_handler,
                                              approximate_solution,
                                              ExactSolution,
                                              difference_per_cell,
                                              quadrature,
                                              dealii::VectorTools::L1_norm);

    double h1_error = dealii::VectorTools::compute_global_error(triangulation,
                                                                difference_per_cell,
                                                                dealii::VectorTools::L1_norm);
    pcout << "L1 error = " << h1_error << std::endl;
  }

  template <int dim>
  std::vector<double>
  Postprocessor<dim>::compute_volume_of_phases(const int              degree,
                                               const int              n_q_points,
                                               const DoFHandler<dim> &dof_handler,
                                               const VectorType &     solution_levelset,
                                               const double           time,
                                               const MPI_Comm &       mpi_communicator,
                                               TableHandler &         volume_table,
                                               const double           max_value,
                                               const double           min_value)
  {
    FE_Q<dim>     fe(degree);
    FEValues<dim> fe_values(fe,
                            QGauss<dim>(n_q_points),
                            update_values | update_JxW_values | update_quadrature_points);

    std::vector<double> phi_at_q(QGauss<dim>(n_q_points).size());

    std::vector<double> volume_fraction;
    double              vol_phase_1 = 0;
    double              vol_phase_2 = 0;
    const double        threshhold  = 0.5;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values.get_function_values(solution_levelset,
                                        phi_at_q); // compute values of old solution

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              const double phi_normalized =
                UtilityFunctions::CharacteristicFunctions::normalize(phi_at_q[q_index],
                                                                     min_value,
                                                                     max_value);
              if (phi_normalized >= threshhold)
                vol_phase_1 += fe_values.JxW(q_index);
              else
                vol_phase_2 += fe_values.JxW(q_index);
            }
        }
    volume_fraction.emplace_back(Utilities::MPI::sum(vol_phase_1, mpi_communicator));
    volume_fraction.emplace_back(Utilities::MPI::sum(vol_phase_2, mpi_communicator));
    //@ todo: write template class for formatted output table
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      std::cout << "vol phase 1: " << volume_fraction[0] << " vol phase 2: " << volume_fraction[1]
                << std::endl;

    volume_table.add_value("time", time);
    volume_table.add_value("vol phase 1", volume_fraction[0]);
    volume_table.add_value("vol phase 2", volume_fraction[1]);

    return volume_fraction;
  }

  template class Postprocessor<1>;
  template class Postprocessor<2>;
  template class Postprocessor<3>;
} // namespace MeltPoolDG
