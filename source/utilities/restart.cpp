#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <meltpooldg/utilities/restart.hpp>

namespace MeltPoolDG::Restart
{
  namespace fs = std::filesystem;

  template <typename number>
  RestartMonitor<number>::RestartMonitor(const RestartData<number>  &data,
                                         const TimeIterator<number> &time)
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

    const bool do_output = (current_time - last_written_time) >= data.write_time_step_size;

    if (do_output)
      last_written_time = current_time;

    return do_output;
  }

  template <typename number>
  void
  RestartMonitor<number>::prepare_save()
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
  }

  template <typename number>
  number
  RestartMonitor<number>::compute_current_time() const
  {
    // note: we use nanoseconds to increase the precision of the real time in seconds
    return (data.time_type == TimeType::real ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::system_clock::now() - real_time_start)
                                                   .count() /
                                                 1e9 :
                                               time.get_current_time());
  }

  template class RestartMonitor<double>;
} // namespace MeltPoolDG::Restart