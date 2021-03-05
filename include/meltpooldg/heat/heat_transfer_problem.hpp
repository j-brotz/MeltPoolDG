/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class HeatTransferProblem : public ProblemBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    TimeIterator<double> time_iterator;
    DoFHandler<dim>      dof_handler;

    AffineConstraints<double> temp_constraints;
    AffineConstraints<double> temp_hanging_nodes_constraints;

    unsigned int temp_dof_idx;
    unsigned int temp_hanging_nodes_dof_idx;
    unsigned int temp_quad_idx;

    std::shared_ptr<ScratchData<dim>>           scratch_data;
    std::shared_ptr<HeatTransferOperation<dim>> heat_operation;
    std::shared_ptr<Postprocessor<dim>>         post_processor;

  public:
    HeatTransferProblem() = default;

    void
    run(std::shared_ptr<SimulationBase<dim>> base_in) final
    {
      initialize(base_in);

      while (!time_iterator.is_finished())
        {
          const auto dt = time_iterator.get_next_time_increment();
          const auto n  = time_iterator.get_current_time_step_number();

          scratch_data->get_pcout()
            << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();

          heat_operation->solve(dt);

          // ... and output the results to vtk files.
          output_results(n);

          if (base_in->parameters.amr.do_amr)
            refine_mesh(base_in);
        }
    }

    std::string
    get_name() final
    {
      return "heat_transfer";
    };

  private:
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize(std::shared_ptr<SimulationBase<dim>> base_in)
    {
      /*
       *  setup DoFHandler
       */
      dof_handler.reinit(*base_in->triangulation);
      /*
       *  setup scratch data
       */
      scratch_data = std::make_shared<ScratchData<dim>>(/*do_matrix_free*/ true);

      /*
       *  setup mapping
       */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
      if (base_in->parameters.base.do_simplex)
        scratch_data->set_mapping(
          MappingFE<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree)));
      else
#endif
        scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));

      scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_dof_handler(dof_handler);

      temp_dof_idx = scratch_data->attach_constraint_matrix(temp_constraints);
      temp_hanging_nodes_dof_idx =
        scratch_data->attach_constraint_matrix(temp_hanging_nodes_constraints);

      /*
       *  create quadrature rule
       */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
      if (base_in->parameters.base.do_simplex)
        {
          temp_quad_idx = scratch_data->attach_quadrature(
            QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
        }
      else
#endif
        {
          temp_quad_idx =
            scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
        }

      setup_dof_system(base_in, false);
      /*
       *  initialize the time stepping scheme
       */
      time_iterator.initialize(
        TimeIteratorData<double>{base_in->parameters.heat.time_stepping.start_time,
                                 base_in->parameters.heat.time_stepping.end_time,
                                 base_in->parameters.heat.time_stepping.time_step_size,
                                 base_in->parameters.heat.time_stepping.max_n_steps,
                                 false /*cfl_condition-->not supported yet*/});
      /*
       *  set initial conditions of the levelset function
       */
      AssertThrow(
        base_in->get_initial_condition("heat_transfer"),
        ExcMessage(
          "It seems that your SimulationBase object does not contain "
          "a valid initial field function for the temperature field. A shared_ptr to your initial field "
          "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
          "  this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), 'temperature') "));

      /*
       *    compute intial conditions of the level set
       */
      VectorType initial_solution;
      scratch_data->initialize_dof_vector(initial_solution, temp_dof_idx);

      dealii::VectorTools::project(scratch_data->get_mapping(),
                                   dof_handler,
                                   temp_constraints,
                                   scratch_data->get_quadrature(temp_quad_idx),
                                   *base_in->get_initial_condition("heat_transfer"),
                                   initial_solution);
      initial_solution.update_ghost_values();
      /*
       *    initialize the heat operation class
       */
      heat_operation =
        std::make_shared<HeatTransferOperation<dim>>(base_in->get_bc("heat_transfer"),
                                                     *scratch_data,
                                                     base_in->parameters.heat,
                                                     temp_dof_idx,
                                                     temp_hanging_nodes_dof_idx,
                                                     temp_quad_idx);

      heat_operation->set_initial_condition(initial_solution);
      /*
       *  initialize postprocessor
       */
      post_processor =
        std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(temp_dof_idx),
                                             base_in->parameters.paraview,
                                             scratch_data->get_mapping(),
                                             scratch_data->get_triangulation(temp_dof_idx));
      /*
       *  output results of initialization --> initial refinement is done afterwards (!)
       *  @todo: find a way to plot vectors on the refined mesh, which are only relevant for output
       *  and which must not be transferred to the new mesh everytime refine_mesh() is called.
       */
      output_results(0);
      /*
       *    Do initial refinement steps if requested
       */
      if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
        for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
          {
            scratch_data->get_pcout()
              << " T.size " << heat_operation->get_temperature().size() << std::endl;
            refine_mesh(base_in);
          }
    }

    void
    setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_reinit = true)
    {
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
      if (base_in->parameters.base.do_simplex)
        {
          dof_handler.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
        }
      else
#endif
        {
          dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
        }
      /*
       *  create partitioning
       */
      scratch_data->create_partitioning();
      /*
       *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
       *  dirichlet constraints are supported)
       */
      temp_hanging_nodes_constraints.clear();
      temp_hanging_nodes_constraints.reinit(
        scratch_data->get_locally_relevant_dofs(temp_hanging_nodes_dof_idx));
      DoFTools::make_hanging_node_constraints(dof_handler, temp_hanging_nodes_constraints);
      temp_hanging_nodes_constraints.close();

      temp_constraints.clear();
      temp_constraints.reinit(scratch_data->get_locally_relevant_dofs(temp_dof_idx));
      if (base_in->get_bc("heat_transfer") && !base_in->get_dirichlet_bc("heat_transfer").empty())
        {
          for (const auto &bc : base_in->get_dirichlet_bc(
                 "heat_transfer")) // @todo: add name of bc at a more central place
            {
              dealii::VectorTools::interpolate_boundary_values(
                scratch_data->get_mapping(), dof_handler, bc.first, *bc.second, temp_constraints);
            }
        }
      temp_constraints.close();
      temp_constraints.merge(temp_constraints,
                             AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

      scratch_data->build();

      if (do_reinit)
        heat_operation->reinit();
    }

    /*
     *  This function is to create paraview output
     */
    void
    output_results(const unsigned int n_time_step)
    {
      /**
       * collect all relevant output data
       */
      const auto attach_output_vectors = [&](DataOut<dim> &data_out) {
        heat_operation->attach_output_vectors(data_out);
      };
      /**
       * do the output operation
       */
      post_processor->process(n_time_step, attach_output_vectors);
    }
    /*
     *  perform mesh refinement
     */
    void
    refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
    {
      (void)base_in;
      AssertThrow(false, ExcNotImplemented());

      // std::vector<
      // std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
      // data;

      // data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
      // heat_operation->attach_vectors(vectors); // temperature
      //});

      // const auto post = [&]() {
      /**
       * temperature
       */
      // temp_constraints.distribute(heat_operation->get_temperature());
      //};

      // const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

      // refine_grid<dim, VectorType>(mark_cells_for_refinement,
      // data,
      // post,
      // setup_dof_system,
      // base_in->parameters.amr,
      // time_iterator.get_current_time_step_number());
    }
  };
} // namespace MeltPoolDG::Heat
