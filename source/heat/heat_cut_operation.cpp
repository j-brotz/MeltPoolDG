#include <meltpooldg/heat/heat_cut_operation.hpp>
//

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/cut_util.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/functions.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

#include <algorithm>
#include <string>

namespace MeltPoolDG::Heat
{
  template <int dim>
  HeatCutOperation<dim>::HeatCutOperation(
    const ScratchData<dim>                     &scratch_data_in,
    const HeatData<double>                     &heat_data_in,
    const MaterialData<double>                 &material_data_in,
    const Evaporation::EvaporationData<double> &evapor_data_in,
    const TimeIterator<double>                 &time_iterator_in,
    const unsigned int                          temp_dof_idx_in,
    const unsigned int                          temp_hanging_nodes_dof_idx_in,
    const unsigned int                          temp_quad_idx_in,
    const bool                                  do_solidification_in,
    const unsigned int                          ls_dof_idx_in,
    const VectorType                           &level_set_in,
    const unsigned int                          vel_dof_idx_in,
    const VectorType                           *velocity_in)
    : scratch_data(scratch_data_in)
    , heat_data(heat_data_in)
    , time_iterator(time_iterator_in)
    , temp_dof_idx(temp_dof_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , temp_quad_idx(temp_quad_idx_in)
    , solution_history(std::max(2U, heat_data.predictor.n_old_solution_vectors))
    , ls_dof_idx(ls_dof_idx_in)
    , level_set(level_set_in)
    , cut_solution_transfer(0.0 /* gamma_degree_0 only for DG */,
                            heat_data.cut.ghost_penalty.gamma_A,
                            0.0 /* gamma_degree_2 not implemented*/,
                            heat_data.cut.two_phase,
                            heat_data.verbosity_level)
    , mapping_info_surface(scratch_data.get_mapping(),
                           dealii::update_values | dealii::update_gradients |
                             dealii::update_JxW_values | dealii::update_normal_vectors)
    , newton(heat_data.nlsolve)
  {
    AssertThrow(heat_data.linear_solver.do_matrix_free, dealii::ExcNotImplemented());

    // liquid domain
    mapping_info_cells.push_back(
      std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<double>>>(
        scratch_data.get_mapping(),
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
          dealii::update_normal_vectors));
    // gas domain
    if (heat_data.cut.two_phase)
      mapping_info_cells.push_back(
        std::make_shared<
          dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<double>>>(
          scratch_data.get_mapping(),
          dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_normal_vectors));

    heat_operator =
      std::make_unique<HeatCutOperator<dim, double>>(scratch_data,
                                                     heat_data,
                                                     material_data_in,
                                                     evapor_data_in,
                                                     temp_dof_idx,
                                                     temp_hanging_nodes_dof_idx,
                                                     temp_quad_idx,
                                                     solution_history.get_current_solution(),
                                                     mapping_info_surface,
                                                     mapping_info_cells,
                                                     do_solidification_in,
                                                     vel_dof_idx_in,
                                                     velocity_in);

    preconditioner = make_preconditioner<dim, HeatCutOperator<dim, double>, VectorType>(
      heat_data.linear_solver.preconditioner_type, heat_operator.get());

    mesh_classifier = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      scratch_data.get_dof_handler(ls_dof_idx), mc_level_set);
    mesh_classifier_old = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      scratch_data.get_dof_handler(ls_dof_idx), mc_level_set);

    setup_newton();

    reinit_vector = [this](VectorType &vec, const DoFHandler<dim> &dh) {
      Assert(&dh == &scratch_data.get_dof_handler(temp_dof_idx), dealii::ExcInternalError());
      vec.reinit(dh.locally_owned_dofs(),
                 scratch_data.get_locally_relevant_dofs(temp_dof_idx),
                 dh.get_communicator());
    };
  }

  template <int dim>
  void
  HeatCutOperation<dim>::register_laser_intensity_function_and_direction(
    std::shared_ptr<const dealii::Function<dim, double>> laser_intensity_profile_in,
    const dealii::Tensor<1, dim, double>                &laser_direction_in)
  {
    heat_operator->register_laser_intensity_function_and_direction(laser_intensity_profile_in,
                                                                   laser_direction_in);
  }

  template <int dim>
  void
  HeatCutOperation<dim>::register_reinit_matrix_free(
    const std::function<void(const dealii::DoFHandler<dim> &)> reinit_matrix_free_in)
  {
    reinit_matrix_free = reinit_matrix_free_in;
  }

  template <int dim>
  void
  HeatCutOperation<dim>::classify_cells() const
  {
    const auto &ls_dof_handler = scratch_data.get_dof_handler(ls_dof_idx);
    mc_level_set.reinit(ls_dof_handler.locally_owned_dofs(),
                        dealii::DoFTools::extract_locally_relevant_dofs(ls_dof_handler),
                        scratch_data.get_mpi_comm(ls_dof_idx));
    mc_level_set.copy_locally_owned_data_from(level_set);
    mc_level_set.update_ghost_values();
    mesh_classifier->reclassify();
  }

  template <int dim>
  void
  HeatCutOperation<dim>::adapt_to_new_interface_position()
  {
    std::swap(mesh_classifier_old, mesh_classifier);
    classify_cells();

    {
      const ScopedName   sc("heat::cut_solution_transfer");
      TimerOutput::Scope scope(scratch_data.get_timer(), sc);

      Assert(reinit_matrix_free != nullptr,
             dealii::ExcMessage("You must register the reinit_matrix_free lambda function first!"));

      // transfer old solution according to the new interface position,
      // the matrix-free object is reinitialized within the reinit function
      cut_solution_transfer.reinit(
        const_cast<dealii::DoFHandler<dim> &>(scratch_data.get_dof_handler(temp_dof_idx)),
        const_cast<dealii::Triangulation<dim> &>(scratch_data.get_triangulation()),
        solution_history.get_current_solution(),
        *mesh_classifier_old,
        *mesh_classifier,
        reinit_vector,
        reinit_matrix_free);
    }
    scratch_data.initialize_dof_vector(solution_history.get_current_solution(), temp_dof_idx);
    solution_history.get_current_solution().copy_locally_owned_data_from(
      cut_solution_transfer.get_updated_solution());
    solution_history.get_current_solution().update_ghost_values();

    compute_intersected_quadrature();
  }

  template <int dim>
  void
  HeatCutOperation<dim>::compute_intersected_quadrature()
  {
    level_set.update_ghost_values();

    CutUtil::compute_intersected_quadrature(mapping_info_cells,
                                            mapping_info_surface,
                                            scratch_data.get_dof_handler(ls_dof_idx),
                                            level_set,
                                            scratch_data.get_matrix_free(),
                                            heat_data.fe.degree,
                                            heat_data.cut.two_phase);
  }

  template <int dim>
  void
  HeatCutOperation<dim>::distribute_dofs(dealii::DoFHandler<dim> &dof_handler) const
  {
    AssertThrow(heat_data.fe.type == FiniteElementType::FE_Q,
                dealii::ExcMessage("For now, only standard FE_Q elements are supported."));

    classify_cells();

    dealii::FE_Q<dim>             fe_q(heat_data.fe.degree);
    dealii::FE_Nothing<dim>       fe_n;
    dealii::hp::FECollection<dim> fe_collection;

    if (heat_data.cut.two_phase)
      {
        fe_collection.push_back(dealii::FESystem<dim, dim>(fe_q, 1, fe_n, 1)); // liquid
        fe_collection.push_back(
          dealii::FESystem<dim, dim>(fe_q, 1, fe_q, 1)); // intersected (liquid and gas)
        fe_collection.push_back(dealii::FESystem<dim, dim>(fe_n, 1, fe_q, 1)); // gas
      }
    else // single phase
      {
        fe_collection.push_back(fe_q); // liquid
        fe_collection.push_back(fe_q); // intersected
        fe_collection.push_back(fe_n); // gas
      }

    CutUtil::set_fe_index<dim>(dof_handler, *mesh_classifier, false /* set_future */);

    dof_handler.distribute_dofs(fe_collection);
  }

  template <int dim>
  void
  HeatCutOperation<dim>::reinit()
  {
    {
      ScopedName sc("heat::n_dofs");
      DoFMonitor::add_n_dofs(sc, scratch_data.get_dof_handler(temp_dof_idx).n_dofs());
    }
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, temp_hanging_nodes_dof_idx); });

    scratch_data.initialize_dof_vector(volumetric_heat_source, temp_hanging_nodes_dof_idx);

    heat_operator->reinit();

    preconditioner.reinit(scratch_data, temp_dof_idx);

    compute_intersected_quadrature();
  }

  template <int dim>
  void
  HeatCutOperation<dim>::set_initial_condition(const dealii::Function<dim> &initial_temperature)
  {
    if (heat_data.cut.two_phase)
      // For the two-phase case, the initial temperature function must be set up with 2 components
      // because the FESystem for cut is set up with 2 to handle both phases.
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       scratch_data.get_dof_handler(temp_dof_idx),
                                       dealii::Functions::TwoComponentFunction<dim>(
                                         initial_temperature),
                                       solution_history.get_current_solution());
    else
      dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                       scratch_data.get_dof_handler(temp_dof_idx),
                                       initial_temperature,
                                       solution_history.get_current_solution());

    scratch_data.get_constraint(temp_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim>
  void
  HeatCutOperation<dim>::distribute_constraints()
  {}

  template <int dim>
  void
  HeatCutOperation<dim>::setup_newton()
  {
    newton.residual = [&](const VectorType & /*solution_update*/, VectorType &rhs) {
      heat_operator->create_rhs(rhs, solution_history.get_recent_old_solution());
      scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(rhs);
    };

    newton.solve_with_jacobian = [&](const VectorType &rhs, VectorType &solution_update) -> int {
      const int iter = LinearSolver::solve<VectorType>(*heat_operator,
                                                       solution_update,
                                                       rhs,
                                                       heat_data.linear_solver,
                                                       preconditioner,
                                                       "heat_operation");

      scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(solution_update);
      return iter;
    };

    newton.reinit_vector = [&](VectorType &v) {
      scratch_data.initialize_dof_vector(v, temp_dof_idx);
    };

    newton.distribute_constraints = [&](VectorType &v) {
      scratch_data.get_constraint(temp_dof_idx).distribute(v);
    };

    newton.norm_of_solution_vector = [this]() -> double {
      return heat_operator->compute_cut_L2_norm(solution_history.get_current_solution());
    };
  }

  template <int dim>
  void
  HeatCutOperation<dim>::init_time_advance()
  {
    // Using solution_history.commit_old_solutions();  messes up the partitioner.
    // This is a workaround.
    auto &solutions = const_cast<std::vector<VectorType> &>(solution_history.get_all_solutions());
    for (int i = solutions.size() - 1; i >= 1; --i)
      solutions[i].swap(solutions[i - 1]);

    // predictor: copy old solution
    // Copying the vector messes up the partitioner. As a workaround we use the copy constructor to
    // construct a temporary vector that is then swap in place.
    VectorType temp = solution_history.get_recent_old_solution();
    solution_history.get_current_solution().swap(temp);

    heat_operator->init_time_advance(time_iterator.get_current_time_increment());
  }

  template <int dim>
  void
  HeatCutOperation<dim>::solve()
  {
    // TODO detect whether the level set has changed
    adapt_to_new_interface_position();

    init_time_advance();

    if (not solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().update_ghost_values();
    if (not solution_history.get_recent_old_solution().has_ghost_elements())
      solution_history.get_recent_old_solution().update_ghost_values();
    heat_operator->update_ghost_values();

    preconditioner.update();

    try
      {
        newton.solve(solution_history.get_current_solution());
      }
    catch (const ExcNewtonDidNotConverge &)
      {
        AssertThrow(false, ExcHeatTransferNoConvergence());
      }
  }

  /**
   * register vectors for adaptive mesh refinement solution transfer
   */
  template <int dim>
  void
  HeatCutOperation<dim>::attach_vectors(std::vector<VectorType *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
  }

  template <int dim>
  void
  HeatCutOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    if (heat_data.cut.two_phase)
      {
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 solution_history.get_current_solution(),
                                 std::vector<std::string>{"temperature", "temperature_gas"});
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 solution_history.get_recent_old_solution(),
                                 std::vector<std::string>{"temperature_old",
                                                          "temperature_old_gas"});
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                                 volumetric_heat_source,
                                 std::vector<std::string>{"heat_source", "heat_source_gas"});
      }
    else // one-phase
      {
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 solution_history.get_current_solution(),
                                 "temperature");
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 solution_history.get_recent_old_solution(),
                                 "temperature_old");
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                                 volumetric_heat_source,
                                 "heat_source");
      }
  }

  template <int dim>
  void
  HeatCutOperation<dim>::attach_output_vectors_failed_step(GenericDataOut<dim> &data_out) const
  {
    (void)data_out;
  }

  template <int dim>
  const HeatCutOperation<dim>::VectorType &
  HeatCutOperation<dim>::get_temperature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  HeatCutOperation<dim>::VectorType &
  HeatCutOperation<dim>::get_temperature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  const HeatCutOperation<dim>::VectorType &
  HeatCutOperation<dim>::get_heat_source() const
  {
    return volumetric_heat_source;
  }

  template <int dim>
  HeatCutOperation<dim>::VectorType &
  HeatCutOperation<dim>::get_heat_source()
  {
    return volumetric_heat_source;
  }



  template class HeatCutOperation<1>;
  template class HeatCutOperation<2>;
  template class HeatCutOperation<3>;
} // namespace MeltPoolDG::Heat
