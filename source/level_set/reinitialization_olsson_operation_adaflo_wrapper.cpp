#include <deal.II/numerics/vector_tools_interpolate.h>
#ifdef MPDG_ENABLE_ADAFLO
#  include <meltpooldg/level_set/reinitialization_olsson_operation_adaflo_wrapper.hpp>
#  include <meltpooldg/utilities/journal.hpp>

#  include <adaflo/level_set_okz_preconditioner.h>
#  include <adaflo/util.h>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationOlssonOperationAdaflo<dim, number>::ReinitializationOlssonOperationAdaflo(
    const ScratchData<dim, dim, number>             &scratch_data,
    const TimeIntegration::TimeIterator<number>     &time_iterator,
    const int                                        reinit_dof_idx,
    const int                                        reinit_quad_idx,
    const int                                        normal_dof_idx,
    const TimeIntegration::TimeSteppingData<number> &time_stepping,
    const NormalVectorData<number>                  &normal_vec_data,
    const number                                     interface_thickness_parameter_value,
    const unsigned int                               n_subdivisions)
    : scratch_data(scratch_data)
    , time_iterator(time_iterator)
    , pcout(scratch_data.get_pcout(2))
    , normal_vector_data(normal_vec_data)
    , eps_cell_factor(interface_thickness_parameter_value / n_subdivisions)
  {
    // set parameters of adaflo
    set_adaflo_parameters(time_stepping, reinit_dof_idx, reinit_quad_idx, normal_dof_idx);

    // setup lambda function to compute the normal vector
    compute_normal = [&](bool do_compute_normal) {
      if (do_compute_normal && force_compute_normal)
        normal_vector_operation_adaflo->solve();
    };
  }

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::create_operator()
  {
    // std::cout <<
    reinit_operation_adaflo = std::make_shared<adaflo::LevelSetOKZSolverReinitialization<dim>>(
      normal_vector_operation_adaflo->get_solution_normal_vector(),
      scratch_data.get_cell_sizes(),
      epsilon_used,
      cell_diameter_min,
      scratch_data.get_constraint(reinit_params_adaflo.dof_index_ls),
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

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::create_normal_vector_operator()
  {
    normal_vector_operation_adaflo =
      std::make_shared<LevelSet::NormalVectorOperationAdaflo<dim, number>>(
        scratch_data,
        normal_vector_data,
        reinit_params_adaflo.dof_index_ls,
        reinit_params_adaflo.dof_index_normal,
        reinit_params_adaflo.quad_index,
        level_set,
        eps_cell_factor);
  }

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::reinit()
  {
    initialize_vectors();

    adaflo::compute_cell_diameters<dim>(scratch_data.get_matrix_free(),
                                        reinit_params_adaflo.dof_index_ls,
                                        cell_diameters,
                                        cell_diameter_min,
                                        cell_diameter_max);

    if (!normal_vector_operation_adaflo)
      create_normal_vector_operator();
    if (!reinit_operation_adaflo)
      {
        epsilon_used = cell_diameter_max * eps_cell_factor;
        create_operator();
      }

    /**
     * initialize the preconditioner
     */
    initialize_mass_matrix_diagonal<dim, number>(scratch_data.get_matrix_free(),
                                                 scratch_data.get_constraint(
                                                   reinit_params_adaflo.dof_index_ls),
                                                 reinit_params_adaflo.dof_index_ls,
                                                 reinit_params_adaflo.quad_index,
                                                 preconditioner);

    normal_vector_operation_adaflo->reinit();
  }

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::solve()
  {
    reinit_operation_adaflo->reinitialize(
      time_iterator.get_current_time_increment(),
      1 /*stab_steps --> we only solve one increment of reinitialization*/,
      0 /*additional diffusion steps --> no input parameter provided; would be useful for highly
           distorted solutions*/
      ,
      compute_normal);

    max_change_level_set = increment.linfty_norm();

    Journal::print_formatted_norm<number>(scratch_data.get_pcout(2),
                                          max_change_level_set,
                                          "delta phi",
                                          "reinitialization",
                                          10 /*precision*/,
                                          "∞ ",
                                          2);
    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(1),
      [&]() -> number {
        return VectorTools::compute_norm<dim, number>(increment,
                                                      scratch_data,
                                                      reinit_params_adaflo.dof_index_ls,
                                                      reinit_params_adaflo.quad_index);
      },
      "delta phi",
      "reinitialization_adaflo",
      10 /*precision*/
    );
    force_compute_normal = false;
  }

  template <int dim, typename number>
  number
  ReinitializationOlssonOperationAdaflo<dim, number>::get_max_change_level_set() const
  {
    return max_change_level_set;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  ReinitializationOlssonOperationAdaflo<dim, number>::get_level_set() const
  {
    return level_set;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  ReinitializationOlssonOperationAdaflo<dim, number>::get_level_set()
  {
    return level_set;
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::BlockVector<number> &
  ReinitializationOlssonOperationAdaflo<dim, number>::get_normal_vector() const
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::BlockVector<number> &
  ReinitializationOlssonOperationAdaflo<dim, number>::get_normal_vector()
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    vectors.push_back(&level_set);
  }

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
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

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::set_initial_condition(
    const VectorType &level_set_in)
  {
    /**
     * initialize advected field dof vectors
     */
    scratch_data.initialize_dof_vector(level_set, reinit_params_adaflo.dof_index_ls);
    level_set.copy_locally_owned_data_from(level_set_in);
    force_compute_normal = true;
  }

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    scratch_data.initialize_dof_vector(level_set, reinit_params_adaflo.dof_index_ls);

    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(
                                       reinit_params_adaflo.dof_index_ls),
                                     initial_field_function,
                                     level_set);

    scratch_data.get_constraint(reinit_params_adaflo.dof_index_ls).distribute(level_set);
    force_compute_normal = true;
  }

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::set_adaflo_parameters(
    const TimeIntegration::TimeSteppingData<number> &time_stepping,
    const int                                        reinit_dof_idx,
    const int                                        reinit_quad_idx,
    const int                                        normal_dof_idx)
  {
    reinit_params_adaflo.time.time_step_scheme =
      adaflo::TimeSteppingParameters::Scheme::implicit_euler;
    reinit_params_adaflo.time.start_time           = 0.0;
    reinit_params_adaflo.time.end_time             = 1e8;
    reinit_params_adaflo.time.time_step_size_start = time_stepping.time_step_size;
    reinit_params_adaflo.time.time_step_size_min   = time_stepping.time_step_size;
    reinit_params_adaflo.time.time_step_size_max   = time_stepping.time_step_size;

    //@todo?
    // if (parameters.ls.reinit.time_integration_scheme == "implicit_euler")
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

  template <int dim, typename number>
  void
  ReinitializationOlssonOperationAdaflo<dim, number>::initialize_vectors()
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


  template class ReinitializationOlssonOperationAdaflo<1, double>;
  template class ReinitializationOlssonOperationAdaflo<2, double>;
  template class ReinitializationOlssonOperationAdaflo<3, double>;
} // namespace MeltPoolDG::LevelSet
#endif
