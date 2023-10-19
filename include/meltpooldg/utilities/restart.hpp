/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, November 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/numerics/solution_transfer.h>

#include <meltpooldg/interface/parameters.hpp>

#include <filesystem>

namespace MeltPoolDG::Restart
{
  namespace fs = std::filesystem;

  template <typename number = double>
  class RestartMonitor
  {
    // TODO: take TimeIterator
  public:
    RestartMonitor(const RestartData<number> &data, const TimeIterator<number> &time)
      : data(data)
      , dir(fs::path(data.prefix).parent_path())
      , prefix(fs::path(data.prefix).filename())
      , time(time)
      , real_time_start(std::chrono::system_clock::now())
    {
      last_written_time = compute_current_time();
      if (!fs::exists(dir) && data.save)
        fs::create_directory(dir);
    }

    bool
    do_load() const
    {
      return data.load >= 0;
    }

    bool
    do_save() const
    {
      if (data.save < 0)
        return false;

      const number current_time = compute_current_time();

      const bool do_output = (current_time - last_written_time) >= data.write_time_step_size;

      if (do_output)
        last_written_time = current_time;

      return do_output;
    }

    /**
     * copy existing restart files
     */
    void
    prepare_save()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
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
                  AssertThrow(false, ExcMessage("You ran into an assert with std::filesystem."));

                  // TODO
                  //// copy parameter file (workaround since overwrite_existing complains with
                  /// certain / compilers)
                  // const auto path_orig = fs::path(parameter_filename);
                  // const auto path_dest =
                  // fs::path(paraview.directory) / fs::path(parameter_filename).filename();

                  // if (!fs::equivalent(path_orig, path_dest))
                  //{
                  // if (fs::exists(path_dest))
                  // fs::remove(path_dest);

                  // fs::copy(path_orig, path_dest, fs::copy_options::overwrite_existing);
                  //}
                }
            }
        }
    }

  private:
    const RestartData<number>                         &data;
    fs::path                                           dir;
    fs::path                                           prefix;
    const TimeIterator<number>                        &time;
    mutable number                                     last_written_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> real_time_start;
    std::vector<std::string>                           suffices = {"_tria",
                                                                   "_tria.info",
                                                                   "_tria_fixed.data",
                                                                   "_problem.restart"};

    number
    compute_current_time() const
    {
      // note: we use nanoseconds to increase the precision of the real time in seconds
      return (data.time_type == RestartTimeType::real ?
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now() - real_time_start)
                    .count() /
                  1e9 :
                time.get_current_time());
    }
  };

  /***************************************************************************************
   * internal functions
   ***************************************************************************************/

  template <int dim, typename VectorType>
  void
  serialize_internal(
    const std::function<
      void(std::vector<std::pair<const DoFHandler<dim> *,
                                 std::function<void(std::vector<VectorType *> &)>>> &data)>
                      &attach_vectors,
    const std::string &prefix)
  {
    std::vector<
      std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
      data;
    attach_vectors(data);

    const unsigned int n = data.size();

    Assert(n > 0, ExcNotImplemented());

    auto triangulation = const_cast<Triangulation<dim> *>(&data[0].first->get_triangulation());

    Assert(triangulation, ExcNotImplemented());

    if (dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation))
      {
        auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation);

        std::vector<std::shared_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>>
          solution_transfer(n);

        std::vector<std::vector<VectorType *>>       new_grid_solutions(n);
        std::vector<std::vector<const VectorType *>> old_grid_solutions(n);

        for (unsigned int j = 0; j < n; ++j)
          {
            data[j].second(new_grid_solutions[j]);

            for (const auto &i : new_grid_solutions[j])
              {
                i->update_ghost_values();
                old_grid_solutions[j].push_back(i);
              }
            solution_transfer[j] =
              std::make_shared<parallel::distributed::SolutionTransfer<dim, VectorType>>(
                *data[j].first);
            solution_transfer[j]->prepare_for_serialization(old_grid_solutions[j]);
          }

        tria->save(prefix + "_tria");
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

  template <int dim, typename VectorType>
  void
  deserialize_internal(
    const std::function<
      void(std::vector<std::pair<const DoFHandler<dim> *,
                                 std::function<void(std::vector<VectorType *> &)>>> &data)>
                                &attach_vectors,
    const std::function<void()> &post,
    const std::function<void()> &setup_dof_system,
    const std::string           &prefix)
  {
    std::vector<
      std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
      data;
    attach_vectors(data);

    const unsigned int n = data.size();

    Assert(n > 0, ExcNotImplemented());

    auto triangulation = const_cast<Triangulation<dim> *>(&data[0].first->get_triangulation());

    Assert(triangulation, ExcNotImplemented());

    if (dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation))
      {
        auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation);

        tria->load(prefix + "_tria");

        setup_dof_system();

        std::vector<std::shared_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>>
          solution_transfer(n);

        std::vector<std::vector<VectorType *>> new_grid_solutions(n);

        for (unsigned int j = 0; j < n; ++j)
          {
            data[j].second(new_grid_solutions[j]);

            solution_transfer[j] =
              std::make_shared<parallel::distributed::SolutionTransfer<dim, VectorType>>(
                *data[j].first);
            solution_transfer[j]->deserialize(new_grid_solutions[j]);
          }

        post();
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

} // namespace MeltPoolDG::Restart
