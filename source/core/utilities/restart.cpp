#include <meltpooldg/utilities/restart.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/solution_transfer.h>

#include <memory>


namespace MeltPoolDG::Restart
{
  namespace fs = std::filesystem;

  template <typename number>
  RestartMonitor<number>::RestartMonitor(const RestartData<number>                   &data,
                                         const TimeIntegration::TimeIterator<number> &time)
    : data(data)
    , dir(fs::path(data.prefix).parent_path())
    , prefix(fs::path(data.prefix).filename())
    , time(time)
    , real_time_start(std::chrono::system_clock::now())
  {
    last_written_time = compute_current_time();
    if (not fs::exists(dir) and data.save)
      fs::create_directory(dir);
  }

  template <typename number>
  bool
  RestartMonitor<number>::do_load() const
  {
    return data.load >= 0;
  }

  template <typename number>
  bool
  RestartMonitor<number>::do_save() const
  {
    if (data.save < 0)
      return false;

    const number current_time = compute_current_time();

    const bool do_output = current_time - last_written_time >= data.write_time_step_size;

    if (do_output)
      last_written_time = current_time;

    return do_output;
  }

  template <typename number>
  void
  RestartMonitor<number>::prepare_save()
  {
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return;

    for (int i = data.save - 2; i >= 0; --i)
      {
        try
          {
            for (const auto &s : suffices)
              {
                if (fs::exists(data.prefix + "_" + std::to_string(i) + s))
                  fs::copy(data.prefix + "_" + std::to_string(i) + s,
                           data.prefix + "_" + std::to_string(i + 1) + s,
                           fs::copy_options::overwrite_existing);
              }
          }
        catch (...)
          {
            AssertThrow(false, dealii::ExcMessage("You ran into an assert with std::filesystem."));

            // TODO
            //// copy parameter file (workaround since overwrite_existing complains with
            /// certain / compilers)
            // const auto path_orig = fs::path(parameter_filename);
            // const auto path_dest =
            // fs::path(output.directory) / fs::path(parameter_filename).filename();

            // if (!fs::equivalent(path_orig, path_dest))
            //{
            // if (fs::exists(path_dest))
            // fs::remove(path_dest);

            // fs::copy(path_orig, path_dest, fs::copy_options::overwrite_existing);
            //}
          }
      }
  }

  template <typename number>
  number
  RestartMonitor<number>::compute_current_time() const
  {
    // note: we use nanoseconds to increase the precision of the real time in seconds
    return data.time_type == TimeType::real ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::system_clock::now() - real_time_start)
                                                  .count() /
                                                1e9 :
                                              time.get_current_time();
  }


  template <int dim, typename VectorType>
  void
  serialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                     const std::string                                     &prefix)
  {
    DoFHandlerAndVectorDataType<dim, VectorType> data;
    attach_vectors(data);
    data.shrink_to_fit();

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    auto tria = dynamic_cast<dealii::parallel::distributed::Triangulation<dim> *>(
      const_cast<dealii::Triangulation<dim> *>(&data[0].first->get_triangulation()));
    AssertThrow(tria, dealii::ExcNotImplemented());

    // the SolutionTransfer instances may not be destructed until Triangulation::save() is called
    std::vector<std::shared_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers(
      n_dof_handlers);
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        std::vector<VectorType *>       new_grid_solutions;
        std::vector<const VectorType *> old_grid_solutions;
        data[j].second(new_grid_solutions);

        for (const auto &i : new_grid_solutions)
          {
            i->update_ghost_values();
            old_grid_solutions.push_back(i);
          }

        solution_transfers[j] =
          std::make_unique<dealii::SolutionTransfer<dim, VectorType>>(*data[j].first);
        solution_transfers[j]->prepare_for_serialization(old_grid_solutions);
      }

    tria->save(prefix + "_tria");
  }

  template <int dim, typename VectorType>
  void
  deserialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                       const std::function<void()>                           &post,
                       const std::function<void()>                           &setup_dof_system,
                       const std::string                                     &prefix)
  {
    DoFHandlerAndVectorDataType<dim, VectorType> data;
    attach_vectors(data);
    data.shrink_to_fit();

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    auto tria = dynamic_cast<dealii::parallel::distributed::Triangulation<dim> *>(
      const_cast<dealii::Triangulation<dim> *>(&data[0].first->get_triangulation()));
    AssertThrow(tria, dealii::ExcNotImplemented());

    tria->load(prefix + "_tria");

    setup_dof_system();

    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        std::vector<VectorType *> new_grid_solutions;
        data[j].second(new_grid_solutions);

        dealii::SolutionTransfer<dim, VectorType> solution_transfer(*data[j].first);
        solution_transfer.deserialize(new_grid_solutions);
      }

    post();
  }

  template class RestartMonitor<double>;

  template void
  serialize_internal(
    const AttachDoFHandlerAndVectorsType<1, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::string &);
  template void
  serialize_internal(
    const AttachDoFHandlerAndVectorsType<2, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::string &);
  template void
  serialize_internal(
    const AttachDoFHandlerAndVectorsType<3, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::string &);

  template void
  deserialize_internal(
    const AttachDoFHandlerAndVectorsType<1, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::function<void()> &,
    const std::function<void()> &,
    const std::string &);
  template void
  deserialize_internal(
    const AttachDoFHandlerAndVectorsType<2, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::function<void()> &,
    const std::function<void()> &,
    const std::string &);
  template void
  deserialize_internal(
    const AttachDoFHandlerAndVectorsType<3, dealii::LinearAlgebra::distributed::Vector<double>> &,
    const std::function<void()> &,
    const std::function<void()> &,
    const std::string &);
} // namespace MeltPoolDG::Restart