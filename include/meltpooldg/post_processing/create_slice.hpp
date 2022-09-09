/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, September 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/post_processing/post_processor_base.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG::PostProcessing
{
  template <int dim>
  class SliceCreator : public PostProcessorBase
  {
  private:
    const GenericDataOut<dim> &        generic_data_out;
    const Triangulation<dim - 1, dim> &tria_slice;
    const std::vector<std::string>     request_variables;
    const ParaviewData<double> &       pv_data;
    // internal

    std::vector<std::pair<double, std::string>> times_and_names;

  public:
    /**
     * Constructor.
     */
    SliceCreator(const GenericDataOut<dim> &        generic_data_out,
                 const Triangulation<dim - 1, dim> &tria_slice,
                 const std::vector<std::string>     request_variables,
                 const ParaviewData<double> &       pv_data)
      : generic_data_out(generic_data_out)
      , tria_slice(tria_slice)
      , request_variables(request_variables)
      , pv_data(pv_data)
    {}

    void
    process(const unsigned int n_time_step) override
    {
      if constexpr (dim > 1)
        {
          MappingQ1<dim - 1, dim>            mapping_slice;
          DataOutResample<dim, dim - 1, dim> data_out(tria_slice, mapping_slice);

          // add vectors
          for (const auto &r : request_variables)
            {
              data_out.add_data_vector(generic_data_out.get_dof_handler(r),
                                       generic_data_out.get_vector(r),
                                       r);
            }
          data_out.update_mapping(generic_data_out.get_mapping());
          data_out.build_patches();

          const std::string pvtu_filename =
            data_out.write_vtu_with_pvtu_record(pv_data.directory + "/",
                                                pv_data.filename + "_slice",
                                                n_time_step,
                                                tria_slice.get_communicator(),
                                                pv_data.n_digits_timestep,
                                                pv_data.n_groups);

          // write a pvd file relating the pvtu-file with a simulation time
          if (Utilities::MPI::this_mpi_process(tria_slice.get_communicator()) == 0 &&
              generic_data_out.get_time() >= 0.0)
            {
              times_and_names.emplace_back(generic_data_out.get_time(), pvtu_filename);
              std::ofstream pvd_output(pv_data.directory + "/" + pv_data.filename + "_slice.pvd");
              DataOutBase::write_pvd_record(pvd_output, times_and_names);
            }
        }
    }
  };
} // namespace MeltPoolDG::PostProcessing
