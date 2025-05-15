#pragma once

#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>
#include <meltpooldg/utilities/restart_data.hpp>

#include <chrono>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>


namespace MeltPoolDG::Restart
{
  template <typename number>
  class RestartMonitor
  {
  public:
    RestartMonitor(const RestartData<number>                   &data,
                   const TimeIntegration::TimeIterator<number> &time);

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
    std::filesystem::path                              dir;
    std::filesystem::path                              prefix;
    const TimeIntegration::TimeIterator<number>       &time;
    mutable number                                     last_written_time = 0.0;
    std::chrono::time_point<std::chrono::system_clock> real_time_start;
    std::vector<std::string>                           suffices = {"_tria",
                                                                   "_tria.info",
                                                                   "_tria_fixed.data",
                                                                   "_tria_variable.data",
                                                                   "_problem.restart"};

    number
    compute_current_time() const;
  };

  /***************************************************************************************
   * internal functions
   ***************************************************************************************/

  template <int dim, typename VectorType>
  void
  serialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                     const std::string                                     &prefix);

  /**
   * Reconstruct all solution vectors that are saved in the restart data, given by the @param prefix
   *
   * @param attach_vectors  lambda function of type AttachDoFHandlerAndVectorsType, that attaches
   *                        all DoFHandlers and their respective DoFVectors that ought to be
   *                        reconstructed
   * @param post this lambda function is run after AMR was executed
   * @param setup_dof_system setup the dof system, this includes:
   *                         - distribute DoFs on the new mesh
   *                         - create partitioning for the new mesh
   *                         - setup constraints on the new mesh
   *                         - reinit the MatrixFree object for the new DoFs (ScratchData::build())
   *                         - initialize all DoF vectors for the new DoF
   */
  template <int dim, typename VectorType>
  void
  deserialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                       const std::function<void()>                           &post,
                       const std::function<void()>                           &setup_dof_system,
                       const std::string                                     &prefix);

} // namespace MeltPoolDG::Restart
