#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_adaflo_wrapper.hpp>
#include <meltpooldg/reinitialization/reinitialization_problem.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  void
  ReinitializationProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator->is_finished())
      {
        time_iterator->compute_next_time_increment();
        time_iterator->print_me(scratch_data->get_pcout());

        reinit_operation->solve();

        output_results(time_iterator->get_current_time_step_number(),
                       time_iterator->get_current_time(),
                       base_in);

        if (base_in->parameters.amr.do_amr)
          refine_mesh(base_in);
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  std::string
  ReinitializationProblem<dim>::get_name()
  {
    return "reinitialization";
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup scratch data
     */
    {
      scratch_data = std::make_shared<ScratchData<dim>>(
        base_in->mpi_communicator,
        base_in->parameters.base.verbosity_level,
        base_in->parameters.ls.reinit.linear_solver.do_matrix_free);
      /*
       *  setup mapping
       */
      scratch_data->set_mapping(
        FiniteElementUtils::create_mapping<dim>(base_in->parameters.base.fe));
      /*
       *  create quadrature rule
       */
      scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(base_in->parameters.base.fe));

      scratch_data->attach_dof_handler(dof_handler);
      reinit_dof_idx = scratch_data->attach_constraint_matrix(constraints);
      normal_dof_idx = reinit_dof_idx;
      /*
       *  setup DoFHandler
       */
      dof_handler.reinit(*base_in->triangulation);
    }

    setup_dof_system(base_in);

    /*
     *  initialize the time iterator
     */
    time_iterator = std::make_shared<TimeIterator<double>>(base_in->parameters.time_stepping);
    /*
     *  set initial conditions of the levelset function
     */
    VectorType solution_level_set;
    scratch_data->initialize_dof_vector(solution_level_set, reinit_dof_idx);

    auto ic = base_in->get_initial_condition("level_set");

    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     dof_handler,
                                     *ic,
                                     solution_level_set);
    constraints.distribute(solution_level_set);
    /*
     *    initialize the reinitialization operation class
     */

    if (base_in->parameters.ls.reinit.implementation == "meltpooldg")
      {
        reinit_operation = std::make_shared<ReinitializationOperation<dim>>(
          *scratch_data,
          base_in->parameters.ls.reinit,
          base_in->parameters.ls.normal_vec,
          base_in->parameters.ls.get_n_subdivisions(),
          *time_iterator,
          reinit_dof_idx,
          reinit_quad_idx,
          reinit_dof_idx,
          normal_dof_idx);
        reinit_operation->reinit();
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if (base_in->parameters.ls.reinit.implementation == "adaflo")
      {
        reinit_operation =
          std::make_shared<ReinitializationOperationAdaflo<dim>>(*scratch_data,
                                                                 *time_iterator,
                                                                 reinit_dof_idx,
                                                                 reinit_quad_idx,
                                                                 normal_dof_idx, // normal vec @todo
                                                                 base_in->parameters);
        reinit_operation->reinit();
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());

    /*
     * set initial conditions
     */
    reinit_operation->set_initial_condition(solution_level_set);
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup DoFHandler
     */
    FiniteElementUtils::distribute_dofs<dim, 1>(base_in->parameters.base.fe, dof_handler);

    /*
     *  re-create partitioning
     */
    scratch_data->create_partitioning();
    /*
     *  make hanging nodes constraints
     */
    MeltPoolDG::Constraints::make_HNC_plus_PBC<dim>(*scratch_data,
                                                    base_in->get_periodic_bc(),
                                                    reinit_dof_idx);
    /*
     *  create the matrix-free object
     */
    scratch_data->build(false, false);

    if (reinit_operation)
      reinit_operation->reinit();

    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(reinit_dof_idx),
                                           base_in->parameters.output,
                                           base_in->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(reinit_dof_idx),
                                           scratch_data->get_pcout(1));
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(reinit_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(reinit_operation->get_level_set());
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
        base_in->parameters.amr.upper_perc_to_refine,
        base_in->parameters.amr.lower_perc_to_coarsen);

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

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 dof_handler,
                                 time_iterator->get_current_time_step_number());
  }

  template <int dim>
  void
  ReinitializationProblem<dim>::output_results(const unsigned int                   time_step,
                                               const double                         time,
                                               std::shared_ptr<SimulationBase<dim>> base_in)
  {
    if (!post_processor->is_output_timestep(time_step, time) &&
        !base_in->parameters.output.do_user_defined_postprocessing)
      return;
    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      reinit_operation->attach_output_vectors(data_out);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         time,
                                         base_in->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    if (base_in->parameters.output.do_user_defined_postprocessing)
      base_in->do_postprocessing(generic_data_out);

    // postprocessing
    post_processor->process(time_step, generic_data_out, time);
  }


  template class ReinitializationProblem<1>;
  template class ReinitializationProblem<2>;
  template class ReinitializationProblem<3>;
} // namespace MeltPoolDG::LevelSet
