/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, November 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/utilities/restart_data.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG::Restart
{
  namespace fs = std::filesystem;

  template <typename number = double>
  class RestartMonitor
  {
  public:
    RestartMonitor(const RestartData<number> &data, const TimeIterator<number> &time);

    bool
    do_load() const;

    bool
    do_save() const;

    /**
     * copy existing restart files
     */
    void
    prepare_save();

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
    compute_current_time() const;
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
    const std::string &prefix);

  template <int dim, typename VectorType>
  void
  deserialize_internal(
    const std::function<
      void(std::vector<std::pair<const DoFHandler<dim> *,
                                 std::function<void(std::vector<VectorType *> &)>>> &data)>
                                &attach_vectors,
    const std::function<void()> &post,
    const std::function<void()> &setup_dof_system,
    const std::string           &prefix);

} // namespace MeltPoolDG::Restart
