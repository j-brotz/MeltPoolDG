#pragma once

#include <meltpooldg/interface/parameters.hpp>

#include <filesystem>

namespace MeltPoolDG
{
  namespace fs = std::filesystem;

  template <typename number = double>
  class RestartMonitor
  {
    // TODO: take TimeIterator
  public:
    RestartMonitor(const RestartData<number> &data)
      : data(data)
      , dir(fs::path(data.prefix).parent_path())
      , prefix(fs::path(data.prefix).filename())
    {
      if (!fs::exists(dir) && data.save)
        fs::create_directory(dir);
    }

    bool
    now(const unsigned int n_time_step) const
    {
      return (n_time_step % data.write_frequency == 0 && data.save);
    }

    /**
     * copy existing restart files
     */
    void
    prepare_save()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          for (int i = data.keep - 2; i >= 0; --i)
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
    const RestartData<double> &data;
    fs::path                   dir;
    fs::path                   prefix;
    std::vector<std::string>   suffices = {"_tria",
                                         "_tria.info",
                                         "_tria_fixed.data",
                                         "_problem.restart"};
  };
} // namespace MeltPoolDG
