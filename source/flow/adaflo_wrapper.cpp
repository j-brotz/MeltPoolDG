#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/flow/adaflo_wrapper.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim>
  AdafloWrapper<dim>::AdafloWrapper(
    ScratchData<dim, dim, double, VectorizedArray<double>> &scratch_data,
    std::shared_ptr<SimulationBase<dim>>                    base_in)
    : scratch_data(scratch_data)
    , timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
    , navier_stokes(base_in->parameters.adaflo_params.get_parameters(),
                    *const_cast<Triangulation<dim> *>(&scratch_data.get_triangulation()),
                    &timer)
    , adaflo_params(base_in->parameters.adaflo_params.get_parameters())
  {
    /*
     * Boundary conditions for the velocity field
     */
    if (base_in->get_bc("navier_stokes_u"))
      {
        for (const auto &symmetry_id : base_in->get_symmetry_id("navier_stokes_u"))
          navier_stokes.set_symmetry_boundary(symmetry_id);
        for (const auto &no_slip_id : base_in->get_no_slip_id("navier_stokes_u"))
          navier_stokes.set_no_slip_boundary(no_slip_id);
        for (const auto &dirichlet_bc : base_in->get_dirichlet_bc("navier_stokes_u"))
          navier_stokes.set_velocity_dirichlet_boundary(dirichlet_bc.first, dirichlet_bc.second);
        for (const auto &open_id : base_in->get_open_boundary_id("navier_stokes_u"))
          navier_stokes.set_open_boundary(open_id);
      }
    /*
     * Boundary conditions for the pressure field
     */
    if (base_in->get_bc("navier_stokes_p"))
      {
        for (const auto &neumann_bc : base_in->get_neumann_bc("navier_stokes_p"))
          navier_stokes.set_open_boundary_with_normal_flux(neumann_bc.first, neumann_bc.second);
        for (const auto &fix_pressure_constant_id :
             base_in->get_fix_pressure_constant_id("navier_stokes_p"))
          navier_stokes.fix_pressure_constant(fix_pressure_constant_id);
      }
    /*
     * Periodic boundary conditions
     */
    for (const auto &periodic_bc : base_in->get_periodic_bc())
      {
        const auto [id_in, id_out, direction] = periodic_bc;
        navier_stokes.set_periodic_direction(direction, id_in, id_out);
      }
    /*
     * Initial conditions of the navier stokes problem
     */
    AssertThrow(
      base_in->get_initial_condition("navier_stokes_u"),
      ExcMessage(
        "It seems that your SimulationBase object does not contain "
        "a valid initial field function for the level set field. A shared_ptr to your initial field "
        "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
        "  this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), 'navier_stokes_u') "));

    this->dof_index_u = scratch_data.attach_dof_handler(navier_stokes.get_dof_handler_u());
    this->dof_index_p = scratch_data.attach_dof_handler(navier_stokes.get_dof_handler_p());
    scratch_data.attach_dof_handler(navier_stokes.get_dof_handler_u());

    scratch_data.attach_constraint_matrix(navier_stokes.get_constraints_u());
    scratch_data.attach_constraint_matrix(navier_stokes.get_constraints_p());
    this->dof_index_hanging_nodes_u =
      scratch_data.attach_constraint_matrix(navier_stokes.get_hanging_node_constraints_u());

    this->quad_index_u =
      adaflo_params.use_simplex_mesh ?
        scratch_data.attach_quadrature(QGaussSimplex<dim>(adaflo_params.velocity_degree + 1)) :
        scratch_data.attach_quadrature(QGauss<dim>(adaflo_params.velocity_degree + 1));
    this->quad_index_p =
      adaflo_params.use_simplex_mesh ?
        scratch_data.attach_quadrature(QGaussSimplex<dim>(adaflo_params.velocity_degree)) :
        scratch_data.attach_quadrature(QGauss<dim>(adaflo_params.velocity_degree));

    // dof handler for output of densities and viscosities
    dof_handler_parameters.reinit(*base_in->triangulation);
    dof_index_parameters = scratch_data.attach_dof_handler(dof_handler_parameters);
    scratch_data.attach_constraint_matrix(constraints_parameters);
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_initial_condition(const Function<dim> &initial_field_function_velocity)
  {
    navier_stokes.solution.zero_out_ghosts();
    navier_stokes.solution_old.zero_out_ghosts();
    dealii::VectorTools::interpolate(navier_stokes.mapping,
                                     navier_stokes.get_dof_handler_u(),
                                     initial_field_function_velocity,
                                     navier_stokes.solution.block(0));

    navier_stokes.get_constraints_u().distribute(navier_stokes.solution.block(0));
    navier_stokes.solution.update_ghost_values();
    navier_stokes.solution_old.update_ghost_values();
  }

  template <int dim>
  void
  AdafloWrapper<dim>::reinit_1()
  {
    // clear constraints and setup hanging node constraints
    navier_stokes.distribute_dofs();

    if (adaflo_params.use_simplex_mesh)
      dof_handler_parameters.distribute_dofs(
        FE_SimplexP<dim>(navier_stokes.get_dof_handler_u().get_fe().tensor_degree()));
    else
      dof_handler_parameters.distribute_dofs(
        FE_Q<dim>(navier_stokes.get_dof_handler_u().get_fe().tensor_degree()));

    // fill constraints_u and constraints_p
    navier_stokes.initialize_data_structures();
  }

  template <int dim>
  void
  AdafloWrapper<dim>::reinit_2()
  {
    navier_stokes.initialize_matrix_free(
      &scratch_data.get_matrix_free(), dof_index_u, dof_index_p, quad_index_u, quad_index_p);
  }

  template <int dim>
  void
  AdafloWrapper<dim>::solve()
  {
    navier_stokes.get_constraints_u().set_zero(navier_stokes.user_rhs.block(0));
    navier_stokes.get_constraints_p().set_zero(navier_stokes.user_rhs.block(1));

    navier_stokes.advance_time_step();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_velocity() const
  {
    return navier_stokes.solution.block(0);
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_velocity()
  {
    return navier_stokes.solution.block(0);
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_velocity_old()
  {
    return navier_stokes.solution_old.block(0);
  }

  template <int dim>
  const DoFHandler<dim> &
  AdafloWrapper<dim>::get_dof_handler_velocity() const
  {
    return navier_stokes.get_dof_handler_u();
  }

  template <int dim>
  const unsigned int &
  AdafloWrapper<dim>::get_dof_handler_idx_velocity() const
  {
    return dof_index_u;
  }

  template <int dim>
  const unsigned int &
  AdafloWrapper<dim>::get_dof_handler_idx_hanging_nodes_velocity() const
  {
    return dof_index_hanging_nodes_u;
  }

  template <int dim>
  const unsigned int &
  AdafloWrapper<dim>::get_quad_idx_velocity() const
  {
    return quad_index_u;
  }

  template <int dim>
  const unsigned int &
  AdafloWrapper<dim>::get_quad_idx_pressure() const
  {
    return quad_index_p;
  }

  template <int dim>
  const AffineConstraints<double> &
  AdafloWrapper<dim>::get_constraints_velocity() const
  {
    return navier_stokes.get_constraints_u();
  }

  template <int dim>
  AffineConstraints<double> &
  AdafloWrapper<dim>::get_constraints_velocity()
  {
    return navier_stokes.modify_constraints_u();
  }

  template <int dim>
  const AffineConstraints<double> &
  AdafloWrapper<dim>::get_hanging_node_constraints_velocity() const
  {
    return navier_stokes.get_hanging_node_constraints_u();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_pressure() const
  {
    return navier_stokes.solution.block(1);
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_pressure()
  {
    return navier_stokes.solution.block(1);
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_pressure_old()
  {
    return navier_stokes.solution_old.block(1);
  }

  template <int dim>
  const DoFHandler<dim> &
  AdafloWrapper<dim>::get_dof_handler_pressure() const
  {
    return navier_stokes.get_dof_handler_p();
  }

  template <int dim>
  const unsigned int &
  AdafloWrapper<dim>::get_dof_handler_idx_pressure() const
  {
    return dof_index_p;
  }

  template <int dim>
  const AffineConstraints<double> &
  AdafloWrapper<dim>::get_constraints_pressure() const
  {
    return navier_stokes.get_constraints_p();
  }

  template <int dim>
  AffineConstraints<double> &
  AdafloWrapper<dim>::get_constraints_pressure()
  {
    return navier_stokes.modify_constraints_p();
  }

  template <int dim>
  const AffineConstraints<double> &
  AdafloWrapper<dim>::get_hanging_node_constraints_pressure() const
  {
    return navier_stokes.get_hanging_node_constraints_p();
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_force_rhs(const LinearAlgebra::distributed::Vector<double> &vec)
  {
    navier_stokes.user_rhs.block(0) = vec;
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_mass_balance_rhs(const LinearAlgebra::distributed::Vector<double> &vec)
  {
    navier_stokes.user_rhs.block(1) = vec;
  }

  template <int dim>
  VectorizedArray<double> &
  AdafloWrapper<dim>::get_density(const unsigned int cell, const unsigned int q)
  {
    return navier_stokes.get_matrix().begin_densities(cell)[q];
  }

  template <int dim>
  const VectorizedArray<double> &
  AdafloWrapper<dim>::get_density(const unsigned int cell, const unsigned int q) const
  {
    return navier_stokes.get_matrix().begin_densities(cell)[q];
  }

  template <int dim>
  VectorizedArray<double> &
  AdafloWrapper<dim>::get_viscosity(const unsigned int cell, const unsigned int q)
  {
    return navier_stokes.get_matrix().begin_viscosities(cell)[q];
  }

  template <int dim>
  const VectorizedArray<double> &
  AdafloWrapper<dim>::get_viscosity(const unsigned int cell, const unsigned int q) const
  {
    return navier_stokes.get_matrix().begin_viscosities(cell)[q];
  }

  template <int dim>
  void
  AdafloWrapper<dim>::attach_vectors_u(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    navier_stokes.solution.block(0).update_ghost_values();
    navier_stokes.solution_old.block(0).update_ghost_values();
    vectors.push_back(&navier_stokes.solution.block(0));
    vectors.push_back(&navier_stokes.solution_old.block(0));
  }

  template <int dim>
  void
  AdafloWrapper<dim>::attach_vectors_p(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    navier_stokes.solution.block(1).update_ghost_values();
    navier_stokes.solution_old.block(1).update_ghost_values();
    vectors.push_back(&navier_stokes.solution.block(1));
    vectors.push_back(&navier_stokes.solution_old.block(1));
  }

  template <int dim>
  void
  AdafloWrapper<dim>::distribute_constraints()
  {
    navier_stokes.get_constraints_u().distribute(navier_stokes.solution.block(0));
    navier_stokes.get_hanging_node_constraints_u().distribute(navier_stokes.solution_old.block(0));
    navier_stokes.get_constraints_p().distribute(navier_stokes.solution.block(1));
    navier_stokes.get_hanging_node_constraints_p().distribute(navier_stokes.solution_old.block(1));
  }

  template <int dim>
  void
  AdafloWrapper<dim>::attach_output_vectors(GenericDataOut<dim> &data_out)
  {
    MeltPoolDG::VectorTools::update_ghost_values(get_velocity(),
                                                 get_pressure(),
                                                 navier_stokes.user_rhs.block(0),
                                                 navier_stokes.user_rhs.block(1));

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      vector_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);
    /**
     *  velocity
     */
    data_out.add_data_vector(get_dof_handler_velocity(),
                             get_velocity(),
                             std::vector<std::string>(dim, "velocity"),
                             vector_component_interpretation);
    /**
     *  pressure
     */
    data_out.add_data_vector(get_dof_handler_pressure(), get_pressure(), "pressure");
    /**
     *  force rhs
     */
    data_out.add_data_vector(get_dof_handler_velocity(),
                             navier_stokes.user_rhs.block(0),
                             std::vector<std::string>(dim, "force_rhs_velocity"),
                             vector_component_interpretation);
    /**
     *  mass balance rhs
     */
    data_out.add_data_vector(get_dof_handler_pressure(),
                             navier_stokes.user_rhs.block(1),
                             "force_rhs_pressure");
    /**
     *  density
     */
    scratch_data.initialize_dof_vector(density, dof_index_parameters);
    UtilityFunctions::fill_dof_vector_from_cell_operation<dim, 1>(
      density,
      scratch_data.get_matrix_free(),
      dof_index_parameters,
      quad_index_u,
      scratch_data.get_fe(dof_index_parameters).tensor_degree(),     // fe_degree,
      scratch_data.get_fe(dof_index_parameters).tensor_degree() + 1, // fe_degree,
      [&](const unsigned int cell, const unsigned int quad) -> const VectorizedArray<double> & {
        return get_density(cell, quad);
      });

    density.update_ghost_values();
    data_out.add_data_vector(dof_handler_parameters, density, "density");
    /**
     *  viscosity
     */
    scratch_data.initialize_dof_vector(viscosity, dof_index_parameters);
    UtilityFunctions::fill_dof_vector_from_cell_operation<dim, 1>(
      viscosity,
      scratch_data.get_matrix_free(),
      dof_index_parameters,
      quad_index_u,
      scratch_data.get_fe(dof_index_parameters).tensor_degree(),     // fe_degree,
      scratch_data.get_fe(dof_index_parameters).tensor_degree() + 1, // fe_degree,
      [&](const unsigned int cell, const unsigned int quad) -> const VectorizedArray<double> & {
        return get_viscosity(cell, quad);
      });
    viscosity.update_ghost_values();
    data_out.add_data_vector(dof_handler_parameters, viscosity, "viscosity");
  }

  template class AdafloWrapper<1>;
  template class AdafloWrapper<2>;
  template class AdafloWrapper<3>;
} // namespace MeltPoolDG::Flow
#endif
