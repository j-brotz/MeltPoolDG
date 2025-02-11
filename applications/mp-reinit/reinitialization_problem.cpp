#include "reinitialization_problem.hpp"

#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/level_set/reinitialization_DG_operation.hpp>
#include <meltpooldg/level_set/reinitialization_operation.hpp>
#include <meltpooldg/level_set/reinitialization_operation_adaflo_wrapper.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  void
  ReinitializationProblem<dim>::run()
  {
    initialize();
    bool first_time_step = true;

    while (!time_iterator->is_finished())
      {
        // Set the first time step size to zero in order to get the initial mass in postprocessing
        // of the DG case
        if (first_time_step && param.reinit.fe.type == FiniteElementType::FE_DGQ)
          {
            time_iterator->set_current_time_increment(0.0, std::numeric_limits<double>::max());
            first_time_step = false;
          }
        else
          {
            if (param.reinit.reinitilization_DG_specific_data.do_CFL_based_time_stepping)
              {
                double const time_step = reinit_operation->compute_CFL_based_timestep();

                time_iterator->set_current_time_increment(time_step,
                                                          std::numeric_limits<double>::max());
              }
          }


        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout());

        reinit_operation->solve();

        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time());

        if (param.amr.do_amr)
          refine_mesh();
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::initialize()
  {
    // setup scratch data
    {
      scratch_data = std::make_shared<ScratchData<dim>>(simulation_case->mpi_communicator,
                                                        param.base.verbosity_level,
                                                        param.reinit.linear_solver.do_matrix_free);

      // setup mapping
      scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(param.reinit.fe));

      // create quadrature rule
      reinit_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(param.reinit.fe));

      scratch_data->attach_dof_handler(dof_handler);
      reinit_dof_idx = scratch_data->attach_constraint_matrix(constraints);
      normal_dof_idx = reinit_dof_idx;

      // setup DoFHandler
      dof_handler.reinit(*simulation_case->triangulation);
    }

    setup_dof_system();

    // initialize the time iterator
    time_iterator = std::make_unique<TimeIterator<double>>(param.time_stepping);

    // initialize the reinitialization operation class
    if (param.reinit.implementation == "meltpooldg")
      {
        if (param.reinit.fe.type != FiniteElementType::FE_DGQ)
          {
            reinit_operation =
              std::make_unique<ReinitializationOperation<dim>>(*scratch_data,
                                                               param.reinit,
                                                               param.normal_vec,
                                                               param.reinit.fe.get_n_subdivisions(),
                                                               *time_iterator,
                                                               reinit_dof_idx,
                                                               reinit_quad_idx,
                                                               reinit_dof_idx,
                                                               normal_dof_idx);
          }
        else
          {
            reinit_operation = std::make_unique<ReinitializationDGOperation<dim>>(*scratch_data,
                                                                                  param.reinit,
                                                                                  *time_iterator,
                                                                                  reinit_dof_idx,
                                                                                  reinit_quad_idx,
                                                                                  reinit_dof_idx,
                                                                                  param.normal_vec,
                                                                                  param.curv);
          }
        reinit_operation->reinit();
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (param.reinit.implementation == "adaflo")
      {
        reinit_operation = std::make_unique<ReinitializationOperationAdaflo<dim>>(
          *scratch_data,
          *time_iterator,
          reinit_dof_idx,
          reinit_quad_idx,
          normal_dof_idx, // normal vec @todo
          param.time_stepping,
          param.normal_vec,
          param.reinit.interface_thickness_parameter.value,
          param.reinit.fe.get_n_subdivisions());
        reinit_operation->reinit();
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());

    reinit_operation->set_initial_condition(*simulation_case->get_initial_condition("level_set"));

    if (param.reinit.fe.type == FiniteElementType::FE_DGQ)
      {
        // For a pure reinit problem this could be done inside reinit_operation, but we want to be
        // able to set it from an external field in a coupled advection/reinit problem
        reinit_operation->get_sign_indicator_function()->copy_locally_owned_data_from(
          reinit_operation->get_level_set());
      }
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::setup_dof_system()
  {
    // setup DoFHandler
    FiniteElementUtils::distribute_dofs<dim, 1>(param.reinit.fe, dof_handler);

    // re-create partitioning
    scratch_data->create_partitioning();

    // Strong enforcement of hanging node constraints and periodic boundary conditions for
    // continuous Galerkin finite elements
    if (param.reinit.fe.type != FiniteElementType::FE_DGQ)
      {
        MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                        simulation_case->get_periodic_bc(),
                                                        reinit_dof_idx);
      }

    // create the matrix-free object
    if (param.reinit.fe.type != FiniteElementType::FE_DGQ)
      scratch_data->build(false, false);
    else
      scratch_data->build(true, true);

    if (reinit_operation)
      reinit_operation->reinit();

    // initialize postprocessor
    post_processor =
      std::make_unique<Postprocessor<dim>>(scratch_data->get_mpi_comm(reinit_dof_idx),
                                           param.output,
                                           param.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(reinit_dof_idx),
                                           scratch_data->get_pcout(1));
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::refine_mesh()
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(simulation_case->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(reinit_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(reinit_operation->get_level_set());
      constraints.close();
      constraints.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(reinit_dof_idx),
                                         scratch_data->get_face_quadrature(reinit_dof_idx),
                                         {},
                                         locally_relevant_solution,
                                         estimated_error_per_cell);

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        param.amr.upper_perc_to_refine,
        param.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      reinit_operation->attach_vectors(vectors);
    };

    const auto post = [&]() {
      constraints.distribute(reinit_operation->get_level_set());

      VectorType temp(reinit_operation->get_level_set());
      temp.copy_locally_owned_data_from(reinit_operation->get_level_set());
      reinit_operation->set_initial_condition(temp);
    };

    const auto setup_dof_system = [&]() { this->setup_dof_system(); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 param.amr,
                                 dof_handler,
                                 time_iterator->get_current_time_step_number());

    // In the DG case the artificial diffusitivity needs to be recalculated if mesh is refined
    reinit_operation->set_artificial_diffusitivity();
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::output_results(const unsigned int time_step, const double time)
  {
    if (!post_processor->is_output_timestep(time_step, time) &&
        !param.output.do_user_defined_postprocessing)
      return;
    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      reinit_operation->attach_output_vectors(data_out);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         time,
                                         param.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (param.output.do_user_defined_postprocessing)
      simulation_case->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, time);
  }


  template class ReinitializationProblem<1>;
  template class ReinitializationProblem<2>;
  template class ReinitializationProblem<3>;
} // namespace MeltPoolDG::LevelSet

int
main(int argc, char *argv[])
{
  using namespace MeltPoolDG;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;
  // check command line arguments
  if (argc == 1)
    {
      if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        std::cout << "ERROR: No .json parameter files has been provided!" << std::endl;
      return 1;
    }
  else if (argc == 2)
    {
      input_file = std::string(argv[argc - 1]);
      run_simulation<LevelSet::ReinitializationCaseParameters<double>,
                     LevelSet::ReinitializationCase,
                     LevelSet::ReinitializationProblem>(input_file, mpi_comm);
    }
  else if (argc == 3 and
           (std::string(argv[1]) == "--help" or std::string(argv[1]) == "--help-detail"))
    {
      input_file = std::string(argv[argc - 1]);

      dealii::ParameterHandler                         prm;
      LevelSet::ReinitializationCaseParameters<double> parameters;
      parameters.process_parameters_file(prm, input_file);

      if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        parameters.print_parameters(prm,
                                    std::cout,
                                    std::string(argv[1]) == "--help-detail" /*print_details*/);

      return 0;
    }
  else
    AssertThrow(false, dealii::ExcMessage("no input file specified"));

  return 0;
}
