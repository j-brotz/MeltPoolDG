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
  void
  LevelSetOperation<dim>::initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                                     std::shared_ptr<SimulationBase<dim>>           base_in,
                                     const unsigned int                             ls_dof_idx_in,
                                     const unsigned int ls_hanging_nodes_dof_idx_in,
                                     const unsigned int ls_quad_idx_in,
                                     const unsigned int reinit_dof_idx_in,
                                     const unsigned int reinit_hanging_nodes_dof_idx_in,
                                     const unsigned int curv_dof_idx_in,
                                     const unsigned int normal_dof_idx_in,
                                     const unsigned int vel_dof_idx,
                                     const unsigned int ls_zero_bc_idx)
  {
    scratch_data             = scratch_data_in;
    ls_dof_idx               = ls_dof_idx_in;
    ls_hanging_nodes_dof_idx = ls_hanging_nodes_dof_idx_in;
    ls_quad_idx              = ls_quad_idx_in;
    curv_dof_idx             = curv_dof_idx_in;
    reinit_dof_idx           = reinit_dof_idx_in;
    max_reinit_steps         = base_in->parameters.reinit.max_n_steps;
    /*
     *  set the level set data
     */
    level_set_data = base_in->parameters.ls;
    /*
     *    initialize the advection diffusion operation and the reinitialization operation class
     */

    if ((base_in->parameters.advec_diff.implementation ==
         "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
      {
        (void)ls_zero_bc_idx;
        advec_diff_operation =
          std::make_shared<AdvectionDiffusion::AdvectionDiffusionOperation<dim>>();
        advec_diff_operation->initialize(scratch_data,
                                         base_in->parameters,
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
            *scratch_data,
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
     *  set the parameters for the levelset problem; already determined parameters
     *  from the initialize call of advec_diff_operation are overwritten.
     */
    set_level_set_parameters(base_in->parameters);

    if ((base_in->parameters.reinit.implementation ==
         "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
      {
        reinit_operation = std::make_shared<Reinitialization::ReinitializationOperation<dim>>();
        reinit_operation->initialize(scratch_data,
                                     base_in->parameters,
                                     reinit_hanging_nodes_dof_idx_in,
                                     ls_quad_idx_in,
                                     ls_dof_idx,
                                     normal_dof_idx_in);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    else if ((base_in->parameters.reinit.implementation == "adaflo") ||
             (base_in->parameters.ls.implementation == "adaflo"))
      {
        AssertThrow(base_in->parameters.reinit.linear_solver.do_matrix_free, ExcNotImplemented());
        reinit_operation = std::make_shared<Reinitialization::ReinitializationOperationAdaflo<dim>>(
          *scratch_data,
          reinit_hanging_nodes_dof_idx_in,
          ls_quad_idx_in,
          normal_dof_idx_in,
          base_in->parameters);
      }
#endif
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     *    initialize the curvature operation class
     */
    if ((base_in->parameters.curv.implementation ==
         "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
      {
        curvature_operation = std::make_shared<Curvature::CurvatureOperation<dim>>();

        curvature_operation->initialize(scratch_data,
                                        base_in->parameters,
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
          *scratch_data_in,
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
    const Function<dim> &initial_field_function_level_set,
    const VectorType &   initial_velocity_in)
  {
    advec_diff_operation->set_initial_condition(initial_field_function_level_set,
                                                initial_velocity_in);
    /*
     * 1) The initial solution of the level set equation will be reinitialized first WITHOUT
     *    dirichlet constraints of the reinitialization.
     */
    do_reinitialization();
    reinit_time_iterator.reset_max_n_time_steps(max_reinit_steps);
    /*
     * 2) From now on, the initial solution of the level set equation will be reinitialized
     *    with dirichlet constraints of the reinitialization.
     *
     * MS:
     * This is needed when the initial level set field should be reinitializated WITHOUT
     * dirichlet constraints and subsequently certain (constrained) dofs of the level set
     * should not change anymore due to reinitialization. This is needed in the
     * MeltPoolOperation, where the level set values in the solid domain should not change
     * anymore after initial reinitialization. This is up to now the only case where dirichlet
     * constraints for reinitialization are set.
     *
     *    @todo: check if this really needed for the melt pool simulations
     */
    reinit_operation->update_dof_idx(reinit_dof_idx);
    /*
     *    compute the smoothened function
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
    reinit_operation->reinit();
    curvature_operation->reinit();

    scratch_data->initialize_dof_vector(level_set_as_heaviside, ls_hanging_nodes_dof_idx);
    scratch_data->initialize_dof_vector(distance_to_level_set, ls_hanging_nodes_dof_idx);
  }

  template <int dim>
  void
  LevelSetOperation<dim>::distribute_constraints()
  {
    //@todo:
    // advec_diff_operation->distribute_constraints();
    // reinit_operation->distribute_constraints();
    // curvature_operation->distribute_constraintst();

    scratch_data->get_constraint(ls_dof_idx).distribute(advec_diff_operation->get_advected_field());
    scratch_data->get_constraint(ls_hanging_nodes_dof_idx).distribute(level_set_as_heaviside);
    scratch_data->get_constraint(ls_hanging_nodes_dof_idx).distribute(distance_to_level_set);
  }
  /**
   *  this function may be called to recompute the normal vector with the
   *  current level set.
   */
  template <int dim>
  void
  LevelSetOperation<dim>::update_normal_vector()
  {
    reinit_operation->set_initial_condition(get_level_set());
  }

  template <int dim>
  void
  LevelSetOperation<dim>::solve(const double dt, const VectorType &advection_velocity)
  {
    /*
     *  1) solve the advection step of the level set function
     */
    {
      TimerOutput::Scope scope(scratch_data->get_timer(), "LevelSet::advect");
      advect_level_set(dt, advection_velocity);
    }
    /*
     *  2) solve the reinitialization problem of the level set equation
     */
    {
      TimerOutput::Scope scope(scratch_data->get_timer(), "LevelSet::reinit");
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
      TimerOutput::Scope scope(scratch_data->get_timer(), "LevelSet::curvature");
      curvature_operation->solve(advec_diff_operation->get_advected_field());
    }
    /*
     *    ... and correct the curvature value far away from the zero level set
     */
    if (level_set_data.do_curvature_correction)
      correct_curvature_values();
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
    if (level_set_data.do_reinitialization)
      return reinit_operation->get_normal_vector();
    else
      return curvature_operation->get_normal_vector();
  }

  template <int dim>
  LinearAlgebra::distributed::BlockVector<double> &
  LevelSetOperation<dim>::get_normal_vector()
  {
    if (level_set_data.do_reinitialization)
      return reinit_operation->get_normal_vector();
    else
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
  /**
   * register vectors for adaptive mesh refinement
   */
  template <int dim>
  void
  LevelSetOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    advec_diff_operation->attach_vectors(vectors);

    level_set_as_heaviside.update_ghost_values();
    vectors.push_back(&level_set_as_heaviside);

    distance_to_level_set.update_ghost_values();
    vectors.push_back(&distance_to_level_set);
  }

  template <int dim>
  void
  LevelSetOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    /*
     *  output advected field
     */
    data_out.add_data_vector(scratch_data->get_dof_handler(ls_dof_idx),
                             get_level_set(),
                             "level_set");

    /*
     *  output normal vector field
     */
    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data->get_dof_handler(ls_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
    /*
     *  output curvature
     */
    data_out.add_data_vector(scratch_data->get_dof_handler(ls_dof_idx),
                             get_curvature(),
                             "curvature");
    /*
     *  output heaviside
     */
    data_out.add_data_vector(scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx),
                             level_set_as_heaviside,
                             "heaviside");
    /*
     *  output distance function
     */
    data_out.add_data_vector(scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx),
                             distance_to_level_set,
                             "distance");
  }

  template <int dim>
  void
  LevelSetOperation<dim>::do_reinitialization()
  {
    if (level_set_data.do_reinitialization)
      {
        reinit_operation->set_initial_condition(advec_diff_operation->get_advected_field());

        Journal::print_decoration_line(scratch_data->get_pcout());
        while (!reinit_time_iterator.is_finished())
          {
            const double       d_tau = reinit_time_iterator.get_next_time_increment();
            std::ostringstream str;
            str << " τ = " << std::setw(10) << std::left << reinit_time_iterator.get_current_time();
            Journal::print_line(scratch_data->get_pcout(), str.str(), "reinitialization", 1);
            reinit_operation->solve(d_tau);
            /*
             *  reset the solution of the level set field to the reinitialized solution ...
             */
            advec_diff_operation->get_advected_field() = reinit_operation->get_level_set();
            /*
             *  @todo
             *
             *  ... and distribute the constraints;
             *
             *  Should constraints between advec diff operation
             *  and reinitialization operation be synched?
             */
            // scratch_data->get_constraint(ls_dof_idx).distribute(advec_diff_operation->get_advected_field());


            // If it is the first reinitialization cycle, the normal vector
            // field might not be computed very accurately from the initial level set
            // field. Thus, in this case we update the normal vector in every reinitialization
            // step.
            if (very_first_step)
              update_normal_vector();
          }
        reinit_time_iterator.reset();

        very_first_step = false;
      }
    Journal::print_decoration_line(scratch_data->get_pcout());
  }

  template <int dim>
  void
  LevelSetOperation<dim>::advect_level_set(const double dt, const VectorType &advection_velocity)
  {
    advec_diff_operation->solve(dt, advection_velocity);
  }

  template <int dim>
  void
  LevelSetOperation<dim>::transform_level_set_to_smooth_heaviside()
  {
    scratch_data->initialize_dof_vector(level_set_as_heaviside, ls_hanging_nodes_dof_idx);
    scratch_data->initialize_dof_vector(distance_to_level_set, ls_hanging_nodes_dof_idx);

    const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const double cut_off_level_set = std::tanh(2);

    for (const auto &cell :
         scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);

          const double epsilon_cell =
            reinit_constant_epsilon > 0.0 ?
              reinit_constant_epsilon :
              UtilityFunctions::compute_cell_size_dependent_interface_thickness<dim>(
                cell, reinit_scale_factor_epsilon / level_set_data.n_subdivisions);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double distance = approximate_distance_from_level_set(
                advec_diff_operation->get_advected_field()[local_dof_indices[i]],
                epsilon_cell,
                cut_off_level_set);
              distance_to_level_set(local_dof_indices[i]) = distance;
              level_set_as_heaviside(local_dof_indices[i]) =
                smooth_heaviside_from_distance_value(2 * distance / (3 * epsilon_cell));
            }
        }
    scratch_data->get_constraint(ls_hanging_nodes_dof_idx).distribute(level_set_as_heaviside);
    scratch_data->get_constraint(ls_hanging_nodes_dof_idx).distribute(distance_to_level_set);
  }

  template <int dim>
  void
  LevelSetOperation<dim>::correct_curvature_values()
  {
    // new approach: use curvature values at the interface
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(
      1e-6 /*tolerance*/, false /*unique mapping*/);

    LevelSet::Tools::broadcast_interface_value_to_vector<dim>(
      scratch_data->get_mapping(),
      scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx),
      scratch_data->get_dof_handler(curv_dof_idx),
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
  LevelSetOperation<dim>::set_level_set_parameters(const Parameters<double> &data_in)
  {
    level_set_data.do_reinitialization                = data_in.ls.do_reinitialization;
    advec_diff_operation->advec_diff_data.diffusivity = data_in.ls.artificial_diffusivity;
    advec_diff_operation->advec_diff_data.time_integration_scheme =
      data_in.ls.time_integration_scheme;
    /*
     *  setup the time iterator for the reinitialization problem
     */
    reinit_time_iterator.initialize(
      TimeIteratorData<double>{0.0,
                               100000.,
                               data_in.reinit.dtau > 0.0 ?
                                 data_in.reinit.dtau :
                                 scratch_data->get_min_cell_size() *
                                   data_in.reinit.scale_factor_epsilon / data_in.ls.n_subdivisions,
                               (unsigned int)data_in.ls.n_initial_reinit_steps,
                               false});

    reinit_constant_epsilon     = data_in.reinit.constant_epsilon;     //@todo: better solution
    reinit_scale_factor_epsilon = data_in.reinit.scale_factor_epsilon; //@todo: better solution
  }

  template class LevelSetOperation<1>;
  template class LevelSetOperation<2>;
  template class LevelSetOperation<3>;
} // namespace MeltPoolDG::LevelSet
