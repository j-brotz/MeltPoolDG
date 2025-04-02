#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/numerics/vector_tools_interpolate.h>

#  include <meltpooldg/level_set/advection_diffusion_adaflo_wrapper.hpp>
#  include <meltpooldg/utilities/journal.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/level_set_okz_preconditioner.h>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  AdvectionDiffusionOperationAdaflo<dim, number>::AdvectionDiffusionOperationAdaflo(
    const ScratchData<dim, dim, number>  &scratch_data,
    const TimeIterator<number>           &time_iterator,
    const VectorType                     &advection_velocity,
    const int                             advec_diff_zero_dirichlet_dof_idx,
    const int                             advec_diff_dirichlet_dof_idx,
    const int                             advec_diff_quad_idx,
    const int                             velocity_dof_idx,
    const TimeSteppingData<number>       &time_stepping,
    const AdvectionDiffusionData<number> &advec_diff_data,
    const BoundaryConditionManager<dim>  &bc)
    : scratch_data(scratch_data)
    , time_iterator(time_iterator)
    , advection_velocity(advection_velocity)
    , pcout(scratch_data.get_pcout(2))
    , dirichlet_dof_idx(advec_diff_dirichlet_dof_idx)
  {
    /**
     * set parameters of adaflo
     */
    set_adaflo_parameters(time_stepping,
                          advec_diff_data,
                          advec_diff_zero_dirichlet_dof_idx,
                          advec_diff_quad_idx,
                          velocity_dof_idx);
    /*
     * Boundary conditions for the advected field
     */
    for (const auto &[symmetry_id, dummy] : bc.get_bc_of_type("symmetry"))
      bcs.symmetry.insert(symmetry_id);
    for (const auto &dirichlet_bc : bc.get_bc_of_type("dirichlet"))
      bcs.dirichlet[dirichlet_bc.first] = dirichlet_bc.second;
    /*
     * initialize adaflo operation
     */
    advec_diff_operation = std::make_shared<LevelSetOKZSolverAdvanceConcentration<dim>>(
      advected_field,
      advected_field_old,
      advected_field_old_old,
      increment,
      rhs,
      velocity_vec,
      velocity_vec_old,
      velocity_vec_old_old,
      scratch_data.get_cell_sizes(),
      scratch_data.get_constraint(advec_diff_zero_dirichlet_dof_idx),
      pcout,
      bcs,
      scratch_data.get_matrix_free(),
      adaflo_params,
      preconditioner);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::reinit()
  {
    /**
     *  initialize the dof vectors
     */
    initialize_vectors();

    /**
     * initialize the preconditioner
     */
    initialize_mass_matrix_diagonal<dim, number>(scratch_data.get_matrix_free(),
                                                 scratch_data.get_constraint(
                                                   adaflo_params.dof_index_ls),
                                                 adaflo_params.dof_index_ls,
                                                 adaflo_params.quad_index,
                                                 preconditioner);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    initialize_vectors();
    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(adaflo_params.dof_index_ls),
                                     initial_field_function,
                                     advected_field);

    scratch_data.get_constraint(dirichlet_dof_idx).distribute(advected_field);
    advected_field_old     = advected_field;
    advected_field_old_old = advected_field;
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::init_time_advance()
  {
    advected_field_old_old.reinit(advected_field_old);
    advected_field_old_old.swap(advected_field_old);
    advected_field_old.swap(advected_field);

    set_velocity(time_iterator.get_current_time_step_number() == 1 /* is initial step */);

    this->ready_for_time_advance = true;
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::solve(const bool do_finish_time_step)
  {
    if (!this->ready_for_time_advance)
      init_time_advance();

    advected_field.update_ghost_values();
    advected_field_old.update_ghost_values();
    advected_field_old_old.update_ghost_values();

    //@todo -- extrapolation (?)
    // if (step_size_old > 0)
    // solution_update.sadd((step_size + step_size_old) / step_size_old,
    //-step_size / step_size_old,
    // solution_old);
    advec_diff_operation->advance_concentration(time_iterator.get_current_time_increment());

    std::ostringstream str;
    str << "|phi| = " << std::setw(11) << std::setprecision(10) << std::left
        << VectorTools::compute_norm<dim, number>(get_advected_field(),
                                                  scratch_data,
                                                  adaflo_params.dof_index_ls,
                                                  adaflo_params.quad_index)
        << " |phi_n-1| = " << std::setw(11) << std::setprecision(10) << std::left
        << VectorTools::compute_norm<dim, number>(get_advected_field_old(),
                                                  scratch_data,
                                                  adaflo_params.dof_index_ls,
                                                  adaflo_params.quad_index)
        << " |phi_n-2| = " << std::setw(11) << std::setprecision(10) << std::left
        << VectorTools::compute_norm<dim, number>(get_advected_field_old_old(),
                                                  scratch_data,
                                                  adaflo_params.dof_index_ls,
                                                  adaflo_params.quad_index);
    Journal::print_line(scratch_data.get_pcout(1), str.str(), "advection_diffusion_adaflo");

    if (do_finish_time_step)
      this->finish_time_advance();
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperationAdaflo<dim, number>::get_advected_field() const
  {
    return advected_field;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperationAdaflo<dim, number>::get_advected_field()
  {
    return advected_field;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperationAdaflo<dim, number>::get_user_rhs() const
  {
    static const LinearAlgebra::distributed::Vector<number> no_vector;
    AssertThrow(false, ExcNotImplemented());
    return no_vector;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperationAdaflo<dim, number>::get_user_rhs()
  {
    static LinearAlgebra::distributed::Vector<number> no_vector;
    AssertThrow(false, ExcNotImplemented());
    return no_vector;
  }


  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    vectors.push_back(&advected_field);
    vectors.push_back(&advected_field_old);
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(adaflo_params.dof_index_ls),
                             advected_field,
                             "advected_field");
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperationAdaflo<dim, number>::get_advected_field_old() const
  {
    return advected_field_old;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperationAdaflo<dim, number>::get_advected_field_old()
  {
    return advected_field_old;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  AdvectionDiffusionOperationAdaflo<dim, number>::get_advected_field_old_old() const
  {
    return advected_field_old_old;
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::set_adaflo_parameters(
    const TimeSteppingData<number>       &time_stepping,
    const AdvectionDiffusionData<number> &advec_diff,
    const int                             advec_diff_dof_idx,
    const int                             advec_diff_quad_idx,
    const int                             velocity_dof_idx)
  {
    adaflo_params.time.start_time           = time_stepping.start_time;
    adaflo_params.time.end_time             = time_stepping.end_time;
    adaflo_params.time.time_step_size_start = time_stepping.time_step_size;
    adaflo_params.time.time_step_size_min   = time_stepping.time_step_size;
    adaflo_params.time.time_step_size_max   = time_stepping.time_step_size;

    if (advec_diff.time_integrator_data.integrator_type == TimeIntegratorSchemes::implicit_euler)
      adaflo_params.time.time_step_scheme = TimeSteppingParameters::Scheme::implicit_euler;
    else if (advec_diff.time_integrator_data.integrator_type ==
             TimeIntegratorSchemes::explicit_euler)
      adaflo_params.time.time_step_scheme = TimeSteppingParameters::Scheme::explicit_euler;
    else if (advec_diff.time_integrator_data.integrator_type ==
             TimeIntegratorSchemes::crank_nicolson)
      adaflo_params.time.time_step_scheme = TimeSteppingParameters::Scheme::crank_nicolson;
    else if (advec_diff.time_integrator_data.integrator_type == TimeIntegratorSchemes::bdf_2)
      adaflo_params.time.time_step_scheme = TimeSteppingParameters::Scheme::bdf_2;
    else
      AssertThrow(false, ExcMessage("Requested time stepping scheme not supported."));
    adaflo_params.time.time_stepping_cfl   = 0.8;  //@ todo
    adaflo_params.time.time_stepping_coef2 = 10.0; //@ todo capillary number

    adaflo_params.dof_index_ls  = advec_diff_dof_idx;
    adaflo_params.dof_index_vel = velocity_dof_idx;
    adaflo_params.quad_index    = advec_diff_quad_idx;

    adaflo_params.convection_stabilization = false; //@ todo
    adaflo_params.do_iteration             = false; //@ todo
    adaflo_params.tol_nl_iteration         = 1e-8;  //@ todo
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::set_velocity(bool initial_step)
  {
    velocity_vec_old_old.zero_out_ghost_values();
    velocity_vec_old.zero_out_ghost_values();
    velocity_vec.zero_out_ghost_values();

    if (initial_step)
      {
        velocity_vec_old_old = advection_velocity;
        velocity_vec_old     = advection_velocity;
        velocity_vec         = advection_velocity;
      }
    else
      {
        velocity_vec_old_old = velocity_vec_old;
        velocity_vec_old     = velocity_vec;
        velocity_vec         = advection_velocity;
      }

    velocity_vec_old_old.update_ghost_values();
    velocity_vec_old.update_ghost_values();
    velocity_vec.update_ghost_values();
  }

  template <int dim, typename number>
  void
  AdvectionDiffusionOperationAdaflo<dim, number>::initialize_vectors()
  {
    /**
     * initialize advected field dof vectors
     */
    scratch_data.initialize_dof_vector(advected_field, adaflo_params.dof_index_ls);
    scratch_data.initialize_dof_vector(advected_field_old, adaflo_params.dof_index_ls);
    scratch_data.initialize_dof_vector(advected_field_old_old, adaflo_params.dof_index_ls);
    /**
     * initialize vectors for the solution of the linear system
     */
    scratch_data.initialize_dof_vector(rhs, adaflo_params.dof_index_ls);
    scratch_data.initialize_dof_vector(increment, adaflo_params.dof_index_ls);
    /**
     *  initialize the velocity vector for adaflo
     */
    scratch_data.initialize_dof_vector(velocity_vec, adaflo_params.dof_index_vel);
    scratch_data.initialize_dof_vector(velocity_vec_old, adaflo_params.dof_index_vel);
    scratch_data.initialize_dof_vector(velocity_vec_old_old, adaflo_params.dof_index_vel);
  }

  template class AdvectionDiffusionOperationAdaflo<1, double>;
  template class AdvectionDiffusionOperationAdaflo<2, double>;
  template class AdvectionDiffusionOperationAdaflo<3, double>;
} // namespace MeltPoolDG::LevelSet

#endif
