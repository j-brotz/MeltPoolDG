// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/curvature/curvature_operation.hpp>
#include <meltpooldg/curvature/curvature_operation_adaflo_wrapper.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_adaflo_wrapper.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;
  using namespace Reinitialization;
  using namespace AdvectionDiffusion;

  template <int dim>
  LevelSetOperation<dim>::LevelSetOperation(const ScratchData<dim> &             scratch_data_in,
                                            const TimeIterator<double> &         time_stepping,
                                            std::shared_ptr<SimulationBase<dim>> base_in,
                                            const unsigned int                   ls_dof_idx_in,
                                            const unsigned int ls_hanging_nodes_dof_idx_in,
                                            const unsigned int ls_quad_idx_in,
                                            const unsigned int reinit_dof_idx_in,
                                            const unsigned int curv_dof_idx_in,
                                            const unsigned int normal_dof_idx_in,
                                            const unsigned int vel_dof_idx,
                                            const unsigned int ls_zero_bc_idx)
    : scratch_data(scratch_data_in)
    , time_stepping(time_stepping)
    , level_set_data(base_in->parameters.ls)
    , reinit_data(base_in->parameters.reinit)
    , ls_dof_idx(ls_dof_idx_in)
    , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx_in)
    , ls_quad_idx(ls_quad_idx_in)
    , curv_dof_idx(curv_dof_idx_in)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_time_iterator(
        TimeSteppingData<double>{0.0,
                                 std::numeric_limits<double>::max(),
                                 level_set_data.reinit_time_step_size > 0.0 ?
                                   level_set_data.reinit_time_step_size :
                                   scratch_data.get_min_cell_size() *
                                     reinit_data.scale_factor_epsilon /
                                     scratch_data.get_degree(ls_dof_idx),
                                 (unsigned int)level_set_data.n_initial_reinit_steps})
  {
    /*
     *    initialize the advection diffusion operation
     */
    if ((base_in->parameters.advec_diff.implementation ==
         "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
      {
        (void)ls_zero_bc_idx;
        advec_diff_operation =
          std::make_shared<AdvectionDiffusion::AdvectionDiffusionOperation<dim>>(
            scratch_data,
            base_in->parameters.advec_diff,
            time_stepping,
            ls_dof_idx,
            ls_hanging_nodes_dof_idx_in,
            ls_quad_idx_in,
            vel_dof_idx);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if ((base_in->parameters.advec_diff.implementation == "adaflo") ||
             (base_in->parameters.ls.implementation == "adaflo"))
      {
        advec_diff_operation =
          std::make_shared<AdvectionDiffusion::AdvectionDiffusionOperationAdaflo<dim>>(
            scratch_data,
            time_stepping,
            ls_zero_bc_idx,
            ls_dof_idx,
            ls_quad_idx_in,
            vel_dof_idx,
            base_in,
            "level_set");

        advec_diff_operation->reinit();
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *  Set the parameters for the levelset problem.
     *
     *  Already determined parameters from the initialize call of advec_diff_operation
     *  are overwritten.
     */
    set_level_set_parameters();
    /*
     *    initialize the reinit operation
     */
    if (level_set_data.do_reinitialization)
      {
        if ((base_in->parameters.reinit.implementation ==
             "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
          {
            reinit_operation = std::make_shared<Reinitialization::ReinitializationOperation<dim>>(
              scratch_data,
              base_in->parameters.reinit,
              base_in->parameters.normal_vec,
              reinit_time_iterator,
              reinit_dof_idx_in,
              ls_quad_idx_in,
              ls_dof_idx,
              normal_dof_idx_in);
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if ((base_in->parameters.reinit.implementation == "adaflo") ||
                 (base_in->parameters.ls.implementation == "adaflo"))
          {
            AssertThrow(base_in->parameters.reinit.linear_solver.do_matrix_free,
                        ExcNotImplemented());
            reinit_operation =
              std::make_shared<Reinitialization::ReinitializationOperationAdaflo<dim>>(
                scratch_data,
                reinit_time_iterator,
                reinit_dof_idx_in,
                ls_quad_idx_in,
                normal_dof_idx_in,
                base_in->parameters);
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());
      }
    /*
     *    initialize the curvature operation class
     */
    if ((base_in->parameters.curv.implementation ==
         "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
      {
        curvature_operation =
          std::make_shared<Curvature::CurvatureOperation<dim>>(scratch_data,
                                                               base_in->parameters.curv,
                                                               base_in->parameters.normal_vec,
                                                               curv_dof_idx_in,
                                                               ls_quad_idx_in,
                                                               normal_dof_idx_in,
                                                               ls_dof_idx);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if ((base_in->parameters.curv.implementation == "adaflo") ||
             (base_in->parameters.ls.implementation == "adaflo"))
      {
        AssertThrow(base_in->parameters.curv.linear_solver.do_matrix_free, ExcNotImplemented());
        curvature_operation = std::make_shared<Curvature::CurvatureOperationAdaflo<dim>>(
          scratch_data_in,
          ls_dof_idx_in,
          normal_dof_idx_in,
          curv_dof_idx_in,
          ls_quad_idx,
          advec_diff_operation->get_advected_field(),
          base_in->parameters);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
  }

  /**
   * set initial condition
   */
  template <int dim>
  void
  LevelSetOperation<dim>::set_initial_condition(
    const Function<dim> &initial_field_function,
    const VectorType &   initial_velocity_in,
    const bool is_signed_distance_initial_field_function) //@todo: provide separate function for
                                                          // this argument
  {
    advec_diff_operation->set_initial_condition(initial_field_function, initial_velocity_in);

    // optional: if the provided function is a signed distance compute a corresponding
    // level set field
    //
    // @todo: create separate function
    if (is_signed_distance_initial_field_function)
      {
        // setup DoF vector holding distances
        scratch_data.initialize_dof_vector(distance_to_level_set, ls_hanging_nodes_dof_idx);
        dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                         scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
                                         initial_field_function,
                                         distance_to_level_set);

        // transform distance to level set function
        transform_distance_to_level_set();

        // set the values in the advection operation
        advec_diff_operation->get_advected_field() = get_level_set();
      }
    // do reinitialization of the initial field if requested
    if (reinit_operation)
      {
        reinit_time_iterator.reset_max_n_time_steps(level_set_data.n_initial_reinit_steps);
        do_reinitialization();
        reinit_time_iterator.reset_max_n_time_steps(reinit_data.max_n_steps);
      }
    /*
     *    compute the localized heaviside function
     */
    transform_level_set_to_smooth_heaviside();
    /*
     *    compute the curvature of the initial level set field
     */
    curvature_operation->solve(advec_diff_operation->get_advected_field());
    /*
     *    correct the curvature value far away from the zero level set
     */
    if (level_set_data.do_curvature_correction)
      correct_curvature_values();
  }

  template <int dim>
  void
  LevelSetOperation<dim>::reinit()
  {
    advec_diff_operation->reinit();
    if (reinit_operation)
      reinit_operation->reinit();
    curvature_operation->reinit();

    scratch_data.initialize_dof_vector(level_set_as_heaviside, ls_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(distance_to_level_set, ls_hanging_nodes_dof_idx);

    reinit_time_iterator.set_current_time_increment(level_set_data.reinit_time_step_size > 0.0 ?
                                                      level_set_data.reinit_time_step_size :
                                                      scratch_data.get_min_cell_size() *
                                                        reinit_data.scale_factor_epsilon /
                                                        scratch_data.get_degree(ls_dof_idx));
  }

  template <int dim>
  void
  LevelSetOperation<dim>::distribute_constraints()
  {
    //@todo:
    // advec_diff_operation->distribute_constraints();
    // reinit_operation->distribute_constraints();
    // curvature_operation->distribute_constraintst();

    scratch_data.get_constraint(ls_dof_idx).distribute(advec_diff_operation->get_advected_field());
    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(level_set_as_heaviside);
    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(distance_to_level_set);

    for (unsigned int d = 0; d < dim; ++d)
      scratch_data.get_constraint(ls_hanging_nodes_dof_idx)
        .distribute(get_normal_vector().block(d));
  }

  template <int dim>
  void
  LevelSetOperation<dim>::set_level_set_user_rhs(const VectorType &level_set_user_rhs)
  {
    advec_diff_operation->get_user_rhs() = level_set_user_rhs;
  }

  template <int dim>
  void
  LevelSetOperation<dim>::init_time_advance()
  {
    // TODO: activate
    // advec_diff_operation->init_time_advance(time_stepping.get_current_time_increment());
    // transform_level_set_to_smooth_heaviside();
  }

  template <int dim>
  void
  LevelSetOperation<dim>::solve(const VectorType &advection_velocity)
  {
    /*
     *  1) solve the advection step of the level set function
     */
    {
      TimerOutput::Scope scope(scratch_data.get_timer(), "LevelSet::advect");
      advect_level_set(advection_velocity);
    }
    /*
     *  2) solve the reinitialization problem of the level set equation
     */
    if (reinit_operation)
      {
        TimerOutput::Scope scope(scratch_data.get_timer(), "LevelSet::reinit");
        do_reinitialization();
      }
    /*
     *  3) compute the smoothened heaviside function ...
     */
    transform_level_set_to_smooth_heaviside();
    /*
     *    ... the curvature
     */
    {
      TimerOutput::Scope scope(scratch_data.get_timer(), "LevelSet::curvature");
      curvature_operation->solve(advec_diff_operation->get_advected_field());
    }
    /*
     *    ... and correct the curvature value far away from the zero level set
     */
    if (level_set_data.do_curvature_correction)
      {
        TimerOutput::Scope scope(scratch_data.get_timer(), "LevelSet::curvature_correction");
        correct_curvature_values();
      }
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  LevelSetOperation<dim>::get_curvature() const
  {
    return curvature_operation->get_curvature();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  LevelSetOperation<dim>::get_curvature()
  {
    return curvature_operation->get_curvature();
  }

  template <int dim>
  const LinearAlgebra::distributed::BlockVector<double> &
  LevelSetOperation<dim>::get_normal_vector() const
  {
    return curvature_operation->get_normal_vector();
  }

  template <int dim>
  LinearAlgebra::distributed::BlockVector<double> &
  LevelSetOperation<dim>::get_normal_vector()
  {
    return curvature_operation->get_normal_vector();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  LevelSetOperation<dim>::get_level_set() const
  {
    return advec_diff_operation->get_advected_field();
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  LevelSetOperation<dim>::get_level_set()
  {
    return advec_diff_operation->get_advected_field();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  LevelSetOperation<dim>::get_level_set_as_heaviside() const
  {
    return level_set_as_heaviside;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  LevelSetOperation<dim>::get_level_set_as_heaviside()
  {
    return level_set_as_heaviside;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  LevelSetOperation<dim>::get_distance_to_level_set() const
  {
    return distance_to_level_set;
  }

  template <int dim>
  const LevelSetOperation<dim>::SurfaceMeshInfo &
  LevelSetOperation<dim>::get_surface_mesh_info() const
  {
    return surface_mesh_info;
  }
  /**
   * register vectors for adaptive mesh refinement
   */
  template <int dim>
  void
  LevelSetOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    advec_diff_operation->attach_vectors(vectors);

    // needed for evaporation
    vectors.push_back(&level_set_as_heaviside);

    reinit_operation->attach_vectors(vectors);
    curvature_operation->attach_vectors(vectors);
  }

  template <int dim>
  void
  LevelSetOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    /*
     * output advected field
     *
     * @todo: advected_field duplicates level_set
     */
    advec_diff_operation->attach_output_vectors(data_out);
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             get_level_set(),
                             "level_set");
    /*
     *  output normal vector field
     */
    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
    /*
     *  output curvature
     *
     *  @todo: move to operation
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_dof_idx),
                             get_curvature(),
                             "curvature");
    /*
     *  output heaviside
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
                             level_set_as_heaviside,
                             "heaviside");
    /*
     *  output distance function
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
                             distance_to_level_set,
                             "distance");
  }

  template <int dim>
  void
  LevelSetOperation<dim>::do_reinitialization()
  {
    // compute the change in the level set since the last reinit
    VectorType temp;
    scratch_data.initialize_dof_vector(temp, ls_dof_idx);
    temp.copy_locally_owned_data_from(get_level_set());
    temp -= reinit_operation->get_level_set();
    max_d_level_set_since_last_reinit = temp.linfty_norm();

    // do reinitialization only if the level set has changed more than a certain tolerance
    if (max_d_level_set_since_last_reinit > level_set_data.tol_reinit)
      {
        reinit_operation->set_initial_condition(advec_diff_operation->get_advected_field());

        Journal::print_decoration_line(scratch_data.get_pcout());
        while (!reinit_time_iterator.is_finished())
          {
            reinit_time_iterator.compute_next_time_increment();

            std::ostringstream str;
            str << " τ = " << std::setw(10) << std::left << reinit_time_iterator.get_current_time();
            Journal::print_line(scratch_data.get_pcout(), str.str(), "reinitialization", 1);

            reinit_operation->solve();

            // Check how much the level set changed due to reinitialization
            if (reinit_operation->get_max_change_level_set() < level_set_data.tol_reinit)
              break;

            /*
             *  reset the solution of the level set field to the reinitialized solution ...
             */
            advec_diff_operation->get_advected_field().copy_locally_owned_data_from(
              reinit_operation->get_level_set());
            /*
             *  @todo
             *
             *  ... and distribute the constraints;
             *
             *  Should constraints between advec diff operation
             *  and reinitialization operation be synched?
             */
            // scratch_data.get_constraint(ls_dof_idx).distribute(advec_diff_operation->get_advected_field());

            // If it is the first reinitialization cycle, the normal vector
            // field might not be computed very accurately from the initial level set
            // field. Thus, in this case we update the normal vector in every reinitialization
            // step.
            if (very_first_step)
              reinit_operation->set_initial_condition(get_level_set());
          }
        reinit_time_iterator.reset();

        very_first_step = false;

        Journal::print_decoration_line(scratch_data.get_pcout());
      }
    else
      {
        std::ostringstream str;
        str << " skipped reinit since max(|ΔΦ|) = " << std::setw(10) << std::setprecision(5)
            << std::scientific << std::left << max_d_level_set_since_last_reinit
            << " < level_set_data.reinit_tol";
        Journal::print_line(scratch_data.get_pcout(), str.str(), "reinitialization", 2);
      }
  }

  template <int dim>
  void
  LevelSetOperation<dim>::advect_level_set(const VectorType &advection_velocity)
  {
    advec_diff_operation->solve(advection_velocity);
  }

  template <int dim>
  void
  LevelSetOperation<dim>::transform_distance_to_level_set()
  {
    distance_to_level_set.update_ghost_values();
    scratch_data.initialize_dof_vector(get_level_set(), ls_dof_idx);

    VectorType multiplicity;
    scratch_data.initialize_dof_vector(multiplicity, ls_hanging_nodes_dof_idx);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(ls_hanging_nodes_dof_idx);

    FEValues<dim> distance_eval(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).get_fe(),
      Quadrature<dim>(
        scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).get_fe().get_unit_support_points()),
      update_values);

    std::vector<double> distance_at_q(distance_eval.n_quadrature_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell :
         scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            distance_eval.reinit(cell);
            distance_eval.get_function_values(distance_to_level_set, distance_at_q);

            const double epsilon_cell =
              reinit_data.constant_epsilon > 0.0 ?
                reinit_data.constant_epsilon :
                UtilityFunctions::compute_cell_size_dependent_interface_thickness<dim>(
                  cell, reinit_data.scale_factor_epsilon / scratch_data.get_degree(ls_dof_idx));

            Vector<double> level_set_local(dofs_per_cell);
            Vector<double> multiplicity_local(dofs_per_cell);

            for (const auto q : distance_eval.quadrature_point_indices())
              {
                multiplicity_local[q] = 1;
                level_set_local[q] =
                  UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
                    distance_at_q[q], epsilon_cell);
              }
            scratch_data.get_constraint(ls_dof_idx)
              .distribute_local_to_global(level_set_local, local_dof_indices, get_level_set());
            scratch_data.get_constraint(ls_dof_idx)
              .distribute_local_to_global(multiplicity_local, local_dof_indices, multiplicity);
          }
      }

    multiplicity.compress(VectorOperation::add);
    get_level_set().compress(VectorOperation::add);
    /*
     * average the nodally assembled values
     */
    for (unsigned int i = 0; i < multiplicity.locally_owned_size(); ++i)
      if (multiplicity.local_element(i) > 1.0)
        get_level_set().local_element(i) /= multiplicity.local_element(i);

    scratch_data.get_constraint(ls_dof_idx).distribute(get_level_set());
    distance_to_level_set.zero_out_ghost_values();
  }

  template <int dim>
  void
  LevelSetOperation<dim>::transform_level_set_to_smooth_heaviside()
  {
    scratch_data.initialize_dof_vector(level_set_as_heaviside, ls_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(distance_to_level_set, ls_hanging_nodes_dof_idx);

    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const double cut_off_level_set = std::tanh(2);

    for (const auto &cell :
         scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);

          const double epsilon_cell =
            reinit_data.constant_epsilon > 0.0 ?
              reinit_data.constant_epsilon :
              UtilityFunctions::compute_cell_size_dependent_interface_thickness<dim>(
                cell, reinit_data.scale_factor_epsilon / scratch_data.get_degree(ls_dof_idx));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double distance = approximate_distance_from_level_set(
                advec_diff_operation->get_advected_field()[local_dof_indices[i]],
                epsilon_cell,
                cut_off_level_set);
              distance_to_level_set(local_dof_indices[i]) = distance;
              if (level_set_data.do_localized_heaviside)
                level_set_as_heaviside(local_dof_indices[i]) =
                  smooth_heaviside_from_distance_value(2 * distance / (3 * epsilon_cell));
              else
                level_set_as_heaviside(local_dof_indices[i]) =
                  (get_level_set()(local_dof_indices[i]) + 1.) * 0.5;
            }
        }
    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(level_set_as_heaviside);
    scratch_data.get_constraint(ls_hanging_nodes_dof_idx).distribute(distance_to_level_set);
  }

  template <int dim>
  void
  LevelSetOperation<dim>::correct_curvature_values()
  {
    // new approach: use curvature values at the interface
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(
      1e-6 /*tolerance*/, false /*unique mapping*/);

    LevelSet::Tools::broadcast_interface_value_to_vector<dim>(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_hanging_nodes_dof_idx),
      scratch_data.get_dof_handler(curv_dof_idx),
      level_set_as_heaviside,
      distance_to_level_set,
      get_normal_vector(),
      curvature_operation->get_curvature(),
      curvature_operation->get_curvature(),
      remote_point_evaluation,
      5 /*n_iterations*/);

    /*
     * old approach --> only kept as back-up [MS]
     */
    // for (unsigned int i = 0; i < curvature_operation->get_curvature().locally_owned_size(); ++i)
    //// if (std::abs(solution_curvature.local_element(i)) > 1e-4)
    // if (1. - advec_diff_operation->get_advected_field().local_element(i) *
    // advec_diff_operation->get_advected_field().local_element(i) >
    // 1e-2)
    // curvature_operation->get_curvature().local_element(i) =
    // 1. / (1. / curvature_operation->get_curvature().local_element(i) +
    // distance_to_level_set.local_element(i) / (dim - 1));
  }

  template <int dim>
  void
  LevelSetOperation<dim>::set_level_set_parameters()
  {
    advec_diff_operation->advec_diff_data.diffusivity = level_set_data.artificial_diffusivity;
    advec_diff_operation->advec_diff_data.time_integration_scheme =
      level_set_data.time_integration_scheme;
  }

  template <int dim>
  void
  LevelSetOperation<dim>::update_surface_mesh()
  {
    surface_mesh_info.clear();
    surface_mesh_info = Tools::generate_surface_mesh_info(scratch_data.get_dof_handler(ls_dof_idx),
                                                          scratch_data.get_mapping(),
                                                          level_set_as_heaviside,
                                                          /*contour of surface*/ 0.5,
                                                          /*n_subdivisions*/ 1,
                                                          /*use_mca*/ true);

    std::ostringstream str;
    str << "Surface mesh generated, "
        << Utilities::MPI::sum(surface_mesh_info.size(), scratch_data.get_mpi_comm())
        << " cut cells found.";
    Journal::print_line(scratch_data.get_pcout(), str.str(), "level set", 1);
  }

  template class LevelSetOperation<1>;
  template class LevelSetOperation<2>;
  template class LevelSetOperation<3>;
} // namespace MeltPoolDG::LevelSet
