/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, September 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/post_processing/post_processor_base.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG::PostProcessingTools
{
  /**
   * Create a (dim-1,dim) slice through a (dim,dim) triangulation.
   *
   * @note: The post processor only supports dim > 1.
   */
  template <int dim>
  class SliceCreator : public PostProcessorBase<dim>
  {
  private:
    const GenericDataOut<dim> *        generic_data_out = nullptr;
    const Triangulation<dim - 1, dim> &tria_slice;
    const std::vector<std::string>     request_variables;
    const ParaviewData<double> &       pv_data;

    // Collect file name and corresponding time step for the pvd-file
    std::vector<std::pair<double, std::string>> times_and_names;

  public:
    /**
     * Constructor.
     */
    SliceCreator(const GenericDataOut<dim> &        generic_data_out,
                 const Triangulation<dim - 1, dim> &tria_slice,
                 const std::vector<std::string>     request_variables,
                 const ParaviewData<double> &       pv_data)
      : generic_data_out(&generic_data_out)
      , tria_slice(tria_slice)
      , request_variables(request_variables)
      , pv_data(pv_data)
    {}

    /**
     * If the address of generic_data_out has changed, use the
     * reinit function passing the new @p generic_data_out_in.
     */
    void
    reinit(const GenericDataOut<dim> &generic_data_out_in)
    {
      generic_data_out = &generic_data_out_in;
    }

    /**
     * Create *.vtu files of the requested variables for the given
     * discretized slice at the current @p n_time_step and include it
     * into the *.pvd file.
     */
    void
    process(const unsigned int n_time_step) override
    {
      AssertThrow(generic_data_out != nullptr, ExcMessage("GenericDataOut is null."));
      AssertThrow(dim > 1, ExcMessage("SliceCreator supports only dim>1."));

      if (dim > 1)
        {
          // create DataOut of the slice
          MappingQ1<dim - 1, dim>            mapping_slice;
          DataOutResample<dim, dim - 1, dim> data_out(tria_slice, mapping_slice);

          if (request_variables[0] == "all" && request_variables.size() == 1)
            {
              for (const auto &data : generic_data_out->entries)
                {
                  // strip components of vector
                  // @todo: problem with dim components for dim-1 slice (vectors get messed up)
                  data_out.add_data_vector(*std::get<0>(data),
                                           *std::get<1>(data),
                                           std::get<2>(data),
                                           std::get<3>(data));
                }
            }
          else if (request_variables.size() > 1)
            {
              // add vectors of requested variables from GenericDataOut
              for (const auto &r : request_variables)
                {
                  //@todo: vectorial data is currently written componentwise
                  //       problem with dim components for dim-1 slice (vectors get messed up)
                  data_out.add_data_vector(generic_data_out->get_dof_handler(r),
                                           generic_data_out->get_vector(r),
                                           r);
                }
            }
          else
            AssertThrow(false, ExcMessage("SliceCreator: error in request variables."));

          data_out.update_mapping(generic_data_out->get_mapping());
          data_out.build_patches();

          // write slice data to vtu files
          const std::string pvtu_filename =
            data_out.write_vtu_with_pvtu_record(pv_data.directory + "/",
                                                pv_data.filename + "_slice",
                                                n_time_step,
                                                tria_slice.get_communicator(),
                                                pv_data.n_digits_timestep,
                                                pv_data.n_groups);

          // write a pvd file relating the pvtu-file with a simulation time
          if (Utilities::MPI::this_mpi_process(tria_slice.get_communicator()) == 0)
            {
              times_and_names.emplace_back(generic_data_out->get_time(), pvtu_filename);
              std::ofstream pvd_output(pv_data.directory + "/" + pv_data.filename + "_slice.pvd");
              DataOutBase::write_pvd_record(pvd_output, times_and_names);
            }
        }
    }
  };
} // namespace MeltPoolDG::PostProcessingTools
