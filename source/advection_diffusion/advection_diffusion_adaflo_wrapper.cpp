#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <meltpooldg/advection_diffusion/advection_diffusion_adaflo_wrapper.hpp>
#  include <meltpooldg/utilities/journal.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  AdvectionDiffusionOperationAdaflo<dim>::AdvectionDiffusionOperationAdaflo(
    const ScratchData<dim>              &scratch_data,
    const TimeIterator<double>          &time_iterator,
    const VectorType                    &advection_velocity,
    const int                            advec_diff_zero_dirichlet_dof_idx,
    const int                            advec_diff_dirichlet_dof_idx,
    const int                            advec_diff_quad_idx,
    const int                            velocity_dof_idx,
    std::shared_ptr<SimulationBase<dim>> base_in,
    std::string                          operation_name)
    : scratch_data(scratch_data)
    , time_iterator(time_iterator)
    , advection_velocity(advection_velocity)
    , pcout(scratch_data.get_pcout(1))
    , dirichlet_dof_idx(advec_diff_dirichlet_dof_idx)
  {
    /**
     * set parameters of adaflo
     */
    set_adaflo_parameters(base_in->parameters,
                          advec_diff_zero_dirichlet_dof_idx,
                          advec_diff_quad_idx,
                          velocity_dof_idx);
    /*
     * Boundary conditions for the advected field
     */
    for (const auto &symmetry_id : base_in->get_symmetry_id(operation_name))
      bcs.symmetry.insert(symmetry_id);
    for (const auto &dirichlet_bc : base_in->get_dirichlet_bc(operation_name).get_data())
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

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::reinit()
  {
    /**
     *  initialize the dof vectors
     */
    initialize_vectors();

    /**
     * initialize the preconditioner
     */
    initialize_mass_matrix_diagonal<dim, double>(scratch_data.get_matrix_free(),
                                                 scratch_data.get_constraint(
                                                   adaflo_params.dof_index_ls),
                                                 adaflo_params.dof_index_ls,
                                                 adaflo_params.quad_index,
                                                 preconditioner);
  }

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::set_initial_condition(
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

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::init_time_advance()
  {
    advected_field_old_old.reinit(advected_field_old);
    advected_field_old_old.swap(advected_field_old);
    advected_field_old.swap(advected_field);

    set_velocity(time_iterator.get_current_time_step_number() == 1 /* is initial step */);

    this->ready_for_time_advance = true;
  }

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::solve(const bool do_finish_time_step)
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
        << VectorTools::compute_norm<dim>(get_advected_field(),
                                          scratch_data,
                                          adaflo_params.dof_index_ls,
                                          adaflo_params.quad_index)
        << " |phi_n-1| = " << std::setw(11) << std::setprecision(10) << std::left
        << VectorTools::compute_norm<dim>(get_advected_field_old(),
                                          scratch_data,
                                          adaflo_params.dof_index_ls,
                                          adaflo_params.quad_index)
        << " |phi_n-2| = " << std::setw(11) << std::setprecision(10) << std::left
        << VectorTools::compute_norm<dim>(get_advected_field_old_old(),
                                          scratch_data,
                                          adaflo_params.dof_index_ls,
                                          adaflo_params.quad_index);
    Journal::print_line(scratch_data.get_pcout(), str.str(), "advection_diffusion_adaflo");

    if (do_finish_time_step)
      this->finish_time_advance();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperationAdaflo<dim>::get_advected_field() const
  {
    return advected_field;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperationAdaflo<dim>::get_advected_field()
  {
    return advected_field;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperationAdaflo<dim>::get_user_rhs() const
  {
    static const LinearAlgebra::distributed::Vector<double> no_vector;
    AssertThrow(false, ExcNotImplemented());
    return no_vector;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperationAdaflo<dim>::get_user_rhs()
  {
    static LinearAlgebra::distributed::Vector<double> no_vector;
    AssertThrow(false, ExcNotImplemented());
    return no_vector;
  }


  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&advected_field);
    vectors.push_back(&advected_field_old);
  }

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(adaflo_params.dof_index_ls),
                             advected_field,
                             "advected_field");
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperationAdaflo<dim>::get_advected_field_old() const
  {
    return advected_field_old;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperationAdaflo<dim>::get_advected_field_old()
  {
    return advected_field_old;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdvectionDiffusionOperationAdaflo<dim>::get_advected_field_old_old() const
  {
    return advected_field_old_old;
  }

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::set_adaflo_parameters(
    const Parameters<double> &parameters,
    const int                 advec_diff_dof_idx,
    const int                 advec_diff_quad_idx,
    const int                 velocity_dof_idx)
  {
    adaflo_params.time.start_time           = parameters.time_stepping.start_time;
    adaflo_params.time.end_time             = parameters.time_stepping.end_time;
    adaflo_params.time.time_step_size_start = parameters.time_stepping.time_step_size;
    adaflo_params.time.time_step_size_min   = parameters.time_stepping.time_step_size;
    adaflo_params.time.time_step_size_max   = parameters.time_stepping.time_step_size;
    if (parameters.ls.advec_diff.time_integration_scheme == "implicit_euler")
      adaflo_params.time.time_step_scheme = TimeSteppingParameters::Scheme::implicit_euler;
    else if (parameters.ls.advec_diff.time_integration_scheme == "explicit_euler")
      adaflo_params.time.time_step_scheme = TimeSteppingParameters::Scheme::explicit_euler;
    else if (parameters.ls.advec_diff.time_integration_scheme == "crank_nicolson")
      adaflo_params.time.time_step_scheme = TimeSteppingParameters::Scheme::crank_nicolson;
    else if (parameters.ls.advec_diff.time_integration_scheme == "bdf_2")
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

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::set_velocity(bool initial_step)
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

  template <int dim>
  void
  AdvectionDiffusionOperationAdaflo<dim>::initialize_vectors()
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

  template class AdvectionDiffusionOperationAdaflo<1>;
  template class AdvectionDiffusionOperationAdaflo<2>;
  template class AdvectionDiffusionOperationAdaflo<3>;
} // namespace MeltPoolDG::LevelSet

#endif
