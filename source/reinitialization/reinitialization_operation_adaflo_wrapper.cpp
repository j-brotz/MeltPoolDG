#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/reinitialization/reinitialization_operation_adaflo_wrapper.hpp>
#  include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::Reinitialization
{
  template <int dim>
  ReinitializationOperationAdaflo<dim>::ReinitializationOperationAdaflo(
    const ScratchData<dim> &  scratch_data,
    const int                 reinit_dof_idx,
    const int                 reinit_quad_idx,
    const int                 normal_dof_idx,
    const Parameters<double> &parameters)
    : scratch_data(scratch_data)
    , pcout(scratch_data.get_pcout(1))
  {
    /**
     * set parameters of adaflo
     */
    set_adaflo_parameters(parameters, reinit_dof_idx, reinit_quad_idx, normal_dof_idx);
    /*
     * initialize normal_vector_operation from adaflo
     */
    normal_vector_operation_adaflo =
      std::make_shared<NormalVector::NormalVectorOperationAdaflo<dim>>(
        scratch_data, reinit_dof_idx, normal_dof_idx, reinit_quad_idx, level_set, parameters);

    /*
     * setup lambda function to compute the normal vector
     */
    compute_normal = [&](bool do_compute_normal) {
      if (do_compute_normal && force_compute_normal)
        normal_vector_operation_adaflo->solve(level_set);
    };

    /*
     * initialize reinitialization operation from adaflo
     */
    reinit_operation_adaflo = std::make_shared<LevelSetOKZSolverReinitialization<dim>>(
      normal_vector_operation_adaflo->get_solution_normal_vector(),
      scratch_data.get_cell_diameters(),
      cell_diameter_max,
      cell_diameter_min,
      scratch_data.get_constraint(reinit_dof_idx),
      increment,
      level_set,
      rhs,
      pcout,
      preconditioner,
      last_concentration_range, // @todo
      reinit_params_adaflo,
      first_reinit_step,
      scratch_data.get_matrix_free());
    /**
     *  initialize the dof vectors and compute the preconditioner
     */
    reinit();
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::update_dof_idx(const unsigned int &reinit_dof_idx)
  {
    reinit_params_adaflo.dof_index_ls = reinit_dof_idx;

    reinit_operation_adaflo = std::make_shared<LevelSetOKZSolverReinitialization<dim>>(
      normal_vector_operation_adaflo->get_solution_normal_vector(),
      scratch_data.get_cell_diameters(),
      cell_diameter_max,
      cell_diameter_min,
      scratch_data.get_constraint(reinit_dof_idx),
      increment,
      level_set,
      rhs,
      pcout,
      preconditioner,
      last_concentration_range, // @todo
      reinit_params_adaflo,
      first_reinit_step,
      scratch_data.get_matrix_free());
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::reinit()
  {
    initialize_vectors();

    compute_cell_diameters<dim>(scratch_data.get_matrix_free(),
                                reinit_params_adaflo.dof_index_ls,
                                cell_diameters,
                                cell_diameter_min,
                                cell_diameter_max);

    /**
     * initialize the preconditioner
     */
    initialize_mass_matrix_diagonal<dim, double>(scratch_data.get_matrix_free(),
                                                 scratch_data.get_constraint(
                                                   reinit_params_adaflo.dof_index_ls),
                                                 reinit_params_adaflo.dof_index_ls,
                                                 reinit_params_adaflo.quad_index,
                                                 preconditioner);

    normal_vector_operation_adaflo->reinit();
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::solve(const double dt)
  {
    reinit_operation_adaflo->reinitialize(dt,
                                          1 /*stab_steps @todo*/,
                                          0 /*diff_steps @todo*/,
                                          compute_normal);
    Journal::print_formatted_norm(scratch_data.get_pcout(1),
                                  increment.linfty_norm(),
                                  "delta phi",
                                  "reinitialization",
                                  10 /*precision*/,
                                  "∞ ",
                                  2);
    Journal::print_formatted_norm(
      scratch_data.get_pcout(0),
      VectorTools::compute_L2_norm<dim>(increment,
                                        scratch_data,
                                        reinit_params_adaflo.dof_index_ls,
                                        reinit_params_adaflo.quad_index),
      "delta phi",
      "reinitialization_adaflo",
      10 /*precision*/
    );
    force_compute_normal = false;
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  ReinitializationOperationAdaflo<dim>::get_level_set() const
  {
    return level_set;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  ReinitializationOperationAdaflo<dim>::get_level_set()
  {
    return level_set;
  }

  template <int dim>
  const LinearAlgebra::distributed::BlockVector<double> &
  ReinitializationOperationAdaflo<dim>::get_normal_vector() const
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim>
  LinearAlgebra::distributed::BlockVector<double> &
  ReinitializationOperationAdaflo<dim>::get_normal_vector()
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&level_set);
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_params_adaflo.dof_index_ls),
                             get_level_set(),
                             "psi");

    //@todo: attach_output_vectors from normal_vector_operation
    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(reinit_params_adaflo.dof_index_ls),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::set_initial_condition(const VectorType &level_set_in)
  {
    /**
     * initialize advected field dof vectors
     */
    scratch_data.initialize_dof_vector(level_set, reinit_params_adaflo.dof_index_ls);
    level_set.copy_locally_owned_data_from(level_set_in);
    force_compute_normal = true;
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::set_adaflo_parameters(const Parameters<double> &parameters,
                                                              const int reinit_dof_idx,
                                                              const int reinit_quad_idx,
                                                              const int normal_dof_idx)
  {
    reinit_params_adaflo.time.start_time           = 0.0;
    reinit_params_adaflo.time.end_time             = 1e8;
    reinit_params_adaflo.time.time_step_size_start = parameters.reinit.dtau;
    reinit_params_adaflo.time.time_step_size_min   = parameters.reinit.dtau;
    reinit_params_adaflo.time.time_step_size_max   = parameters.reinit.dtau;

    //@todo?
    // if (parameters.reinit.time_integration_scheme == "implicit_euler")
    // reinit_params_adaflo.time.time_step_scheme =
    // TimeSteppingParameters::Scheme::implicit_euler;
    //
    reinit_params_adaflo.time.time_stepping_cfl   = 0.8;  //@ todo
    reinit_params_adaflo.time.time_stepping_coef2 = 10.0; //@ todo capillary number

    reinit_params_adaflo.dof_index_ls     = reinit_dof_idx;
    reinit_params_adaflo.dof_index_normal = normal_dof_idx;
    reinit_params_adaflo.quad_index       = reinit_quad_idx;
    reinit_params_adaflo.do_iteration     = false; //@ todo
  }

  template <int dim>
  void
  ReinitializationOperationAdaflo<dim>::initialize_vectors()
  {
    /**
     * initialize advected field dof vectors
     */
    scratch_data.initialize_dof_vector(level_set, reinit_params_adaflo.dof_index_ls);
    /**
     * initialize vectors for the solution of the linear system
     */
    scratch_data.initialize_dof_vector(rhs, reinit_params_adaflo.dof_index_ls);
    scratch_data.initialize_dof_vector(increment, reinit_params_adaflo.dof_index_ls);
  }


  template class ReinitializationOperationAdaflo<1>;
  template class ReinitializationOperationAdaflo<2>;
  template class ReinitializationOperationAdaflo<3>;
} // namespace MeltPoolDG::Reinitialization
#endif
