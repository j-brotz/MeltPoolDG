#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/geometry_info.h>
#  include <deal.II/base/index_set.h>
#  include <deal.II/base/point.h>

#  include <deal.II/dofs/dof_accessor.h>
#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/fe/fe_q.h>
#  include <deal.II/fe/fe_simplex_p.h>
#  include <deal.II/fe/fe_values.h>

#  include <deal.II/grid/grid_generator.h>
#  include <deal.II/grid/tria_iterator.h>

#  include <deal.II/numerics/data_component_interpretation.h>
#  include <deal.II/numerics/vector_tools_interpolate.h>

#  include <meltpooldg/flow/adaflo_wrapper.hpp>
#  include <meltpooldg/material/material.templates.hpp>
#  include <meltpooldg/utilities/constraints.hpp>
#  include <meltpooldg/utilities/dof_monitor.hpp>
#  include <meltpooldg/utilities/iteration_monitor.hpp>
#  include <meltpooldg/utilities/journal.hpp>
#  include <meltpooldg/utilities/scoped_name.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim>
  AdafloWrapper<dim>::AdafloWrapper(
    ScratchData<dim, dim, double, VectorizedArray<double>> &scratch_data,
    std::shared_ptr<SimulationBase<dim>>                    base_in,
    const TimeIterator<double>                             &time_iterator,
    const bool                                              do_evaporative_mass_flux)
    : scratch_data(scratch_data)
    , timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
    , adaflo_params(base_in->parameters.adaflo_params.get_parameters())
    , do_evaporative_mass_flux(do_evaporative_mass_flux)
    , time_iterator(time_iterator)
  {
    /*
     *  create input parameters for adaflo
     */
    create_parameters(base_in->parameters, base_in->parameter_file);
    /*
     * setup Navier-Stokes solver
     */
    navier_stokes = std::make_unique<NavierStokes<dim>>(
      adaflo_params, *const_cast<Triangulation<dim> *>(&scratch_data.get_triangulation()), &timer);
    /*
     * Boundary conditions for the velocity field
     */
    base_in->attach_boundary_condition("navier_stokes_u");
    for (const auto &symmetry_id : base_in->get_symmetry_id("navier_stokes_u"))
      navier_stokes->set_symmetry_boundary(symmetry_id);
    for (const auto &no_slip_id : base_in->get_no_slip_id("navier_stokes_u"))
      navier_stokes->set_no_slip_boundary(no_slip_id);
    for (const auto &dirichlet_bc : base_in->get_dirichlet_bc("navier_stokes_u").get_data())
      navier_stokes->set_velocity_dirichlet_boundary(dirichlet_bc.first, dirichlet_bc.second);
    for (const auto &open_bc : base_in->get_open_boundary_bc("navier_stokes_u"))
      navier_stokes->set_open_boundary(open_bc.first, open_bc.second);
    /*
     * Boundary conditions for the pressure field
     */
    base_in->attach_boundary_condition("navier_stokes_p");
    for (const auto &neumann_bc : base_in->get_neumann_bc("navier_stokes_p"))
      navier_stokes->set_open_boundary_with_normal_flux(neumann_bc.first, neumann_bc.second);
    for (const auto &fix_pressure_constant_id :
         base_in->get_fix_pressure_constant_id("navier_stokes_p"))
      navier_stokes->fix_pressure_constant(fix_pressure_constant_id);
    /*
     * Periodic boundary conditions
     */
    for (const auto &periodic_bc : base_in->get_periodic_bc().get_data())
      {
        const auto [id_in, id_out, direction] = periodic_bc;
        navier_stokes->set_periodic_direction(direction, id_in, id_out);
      }
    /*
     * Attach DoFHandler and constraints
     */
    this->dof_index_u = scratch_data.attach_dof_handler(navier_stokes->get_dof_handler_u());
    this->dof_index_p = scratch_data.attach_dof_handler(navier_stokes->get_dof_handler_p());
    scratch_data.attach_dof_handler(navier_stokes->get_dof_handler_u());

    scratch_data.attach_constraint_matrix(navier_stokes->get_constraints_u());
    scratch_data.attach_constraint_matrix(navier_stokes->get_constraints_p());
    this->dof_index_hanging_nodes_u =
      scratch_data.attach_constraint_matrix(navier_stokes->get_hanging_node_constraints_u());

    this->quad_index_u =
      adaflo_params.use_simplex_mesh ?
        scratch_data.attach_quadrature(QGaussSimplex<dim>(adaflo_params.velocity_degree + 1)) :
        scratch_data.attach_quadrature(QGauss<dim>(adaflo_params.velocity_degree + 1));
    this->quad_index_p =
      adaflo_params.use_simplex_mesh ?
        scratch_data.attach_quadrature(QGaussSimplex<dim>(adaflo_params.velocity_degree)) :
        scratch_data.attach_quadrature(QGauss<dim>(adaflo_params.velocity_degree));

    // dof handler for output of densities and viscosities
    // @todo: introduce only if do output == true
    dof_handler_parameters.reinit(*base_in->triangulation);
    dof_index_parameters = scratch_data.attach_dof_handler(dof_handler_parameters);
    scratch_data.attach_constraint_matrix(constraints_parameters);
  }

  template <int dim>
  void
  AdafloWrapper<dim>::create_parameters(Parameters<double> &parameters,
                                        const std::string   parameter_file)
  {
    parameters.adaflo_params.parse_parameters(parameter_file);

    AssertThrow(parameters.adaflo_params.params.density ==
                  1.0, // 1.0 is the default value from adaflo
                ExcMessage("It seems that you specified the density parameter "
                           "within the adaflo section, which is ignored by MeltPoolDG. "
                           "Please use the >material: material first density:< section instead. "));

    AssertThrow(parameters.adaflo_params.params.viscosity ==
                  1.0, // 1.0 is the default value from adaflo
                ExcMessage("It seems that you specified the viscosity parameter "
                           "within the adaflo section, which is ignored by MeltPoolDG. "
                           "Please use the >material: material first density:< section instead. "));

    // TODO: move to check in evaporation data
    if (do_evaporative_mass_flux)
      {
        if (parameters.evapor.evaporative_dilation_rate.enable)
          {
            AssertThrow(
              parameters.adaflo_params.params.beta_convective_term_momentum_balance == 0,
              ExcMessage(
                "For the consideration of phase change, the convective "
                "formulation of the momentum balance in the Navier-Stokes equations "
                "must be chosen: Navier-Stokes: adaflo: Navier-Stokes: {formulation convective "
                "term momentum balance: convective }"));

            //@todo: the following is kept as back-up

            // AssertThrow(material.two_phase_properties_transition_type ==
            // TwoPhasePropertiesTransitionType::consistent_with_evaporation,
            // ExcMessage(
            //"For the consideration of phase change, the density "
            //"has to be interpolated consistently with the continuity equation "
            //"including phase change."));
          }
      }
    // WARNING: by setting the differences to a non-zero value we force
    //   adaflo to assume that we are running a simulation with variable
    //   coefficients, i.e., it allocates memory for the data structures
    //   variable_densities and variable_viscosities, which are accessed
    //   during NavierStokesMatrix::begin_densities() and
    //   NavierStokesMatrix::begin_viscosity(). However, we do not actually
    //   use these values, since we fill the density and viscosity
    //   differently.
    parameters.adaflo_params.params.density_diff   = 1.0;
    parameters.adaflo_params.params.viscosity_diff = 1.0;

    if (parameters.material.gas.density > 0.0)
      {
        // adaflo assumes the parameter density to be the one of heaviside == 0
        parameters.adaflo_params.params.density = parameters.material.gas.density;
      }
    if (parameters.material.gas.dynamic_viscosity > 0.0)
      {
        // adaflo assumes the parameter viscosity to be the one of heaviside == 0
        parameters.adaflo_params.params.viscosity = parameters.material.gas.dynamic_viscosity;
      }

    /// synchronize time stepping schemes
    parameters.adaflo_params.params.start_time           = parameters.time_stepping.start_time;
    parameters.adaflo_params.params.end_time             = parameters.time_stepping.end_time;
    parameters.adaflo_params.params.time_step_size_start = parameters.time_stepping.time_step_size;
    parameters.adaflo_params.params.time_step_size_min   = 1e-16;
    parameters.adaflo_params.params.time_step_size_max   = 1e10;

    parameters.adaflo_params.params.use_simplex_mesh =
      parameters.base.fe.type == FiniteElementType::FE_SimplexP;
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_initial_condition(const Function<dim> &initial_field_function_velocity)
  {
    navier_stokes->solution.zero_out_ghost_values();
    navier_stokes->solution_old.zero_out_ghost_values();
    navier_stokes->solution_old_old.zero_out_ghost_values();
    dealii::VectorTools::interpolate(navier_stokes->mapping,
                                     navier_stokes->get_dof_handler_u(),
                                     initial_field_function_velocity,
                                     navier_stokes->solution.block(0));

    // the hanging node constraints contain the inhomogeneity
    navier_stokes->get_hanging_node_constraints_u().distribute(navier_stokes->solution.block(0));
    navier_stokes->get_hanging_node_constraints_p().distribute(navier_stokes->solution.block(1));

    navier_stokes->solution.update_ghost_values();
    navier_stokes->solution_old.update_ghost_values();
    navier_stokes->solution_old_old.update_ghost_values();
  }

  template <int dim>
  void
  AdafloWrapper<dim>::reinit_1()
  {
    // clear constraints and setup hanging node constraints
    navier_stokes->distribute_dofs();

    if (adaflo_params.use_simplex_mesh)
      dof_handler_parameters.distribute_dofs(
        FE_SimplexP<dim>(navier_stokes->get_dof_handler_u().get_fe().tensor_degree()));
    else
      dof_handler_parameters.distribute_dofs(
        FE_Q<dim>(navier_stokes->get_dof_handler_u().get_fe().tensor_degree()));

    // fill constraints_u and constraints_p
    navier_stokes->initialize_data_structures();

    // only for output purposes
    constraints_parameters.clear();
    IndexSet locally_relevant_dofs_temp;
    DoFTools::extract_locally_relevant_dofs(dof_handler_parameters, locally_relevant_dofs_temp);

    constraints_parameters.reinit(locally_relevant_dofs_temp);
    DoFTools::make_hanging_node_constraints(dof_handler_parameters, constraints_parameters);
    constraints_parameters.close();
    Constraints::check_constraints(dof_handler_parameters, constraints_parameters);

    {
      ScopedName sc("ns::n_dofs_u");
      DoFMonitor::add_n_dofs(sc, get_dof_handler_velocity().n_dofs());
    }
    {
      ScopedName sc2("ns::n_dofs_p");
      DoFMonitor::add_n_dofs(sc2, get_dof_handler_pressure().n_dofs());
    }
  }

  template <int dim>
  void
  AdafloWrapper<dim>::reinit_2()
  {
    navier_stokes->initialize_matrix_free(
      &scratch_data.get_matrix_free(), dof_index_u, dof_index_p, quad_index_u, quad_index_p);
  }

  template <int dim>
  void
  AdafloWrapper<dim>::reinit_3()
  {
    VectorType vel_temp, vel_temp_old, vel_temp_old_old, pressure_temp, pressure_temp_old,
      pressure_temp_old_old;

    scratch_data.initialize_dof_vector(vel_temp, dof_index_u);
    scratch_data.initialize_dof_vector(vel_temp_old, dof_index_u);
    scratch_data.initialize_dof_vector(vel_temp_old_old, dof_index_u);

    scratch_data.initialize_dof_vector(pressure_temp, dof_index_p);
    scratch_data.initialize_dof_vector(pressure_temp_old, dof_index_p);
    scratch_data.initialize_dof_vector(pressure_temp_old_old, dof_index_p);

    vel_temp.copy_locally_owned_data_from(navier_stokes->solution.block(0));
    vel_temp_old.copy_locally_owned_data_from(navier_stokes->solution_old.block(0));
    vel_temp_old_old.copy_locally_owned_data_from(navier_stokes->solution_old_old.block(0));

    pressure_temp.copy_locally_owned_data_from(navier_stokes->solution.block(1));
    pressure_temp_old.copy_locally_owned_data_from(navier_stokes->solution_old.block(1));
    pressure_temp_old_old.copy_locally_owned_data_from(navier_stokes->solution_old_old.block(1));

    navier_stokes->initialize_matrix_free(
      &scratch_data.get_matrix_free(), dof_index_u, dof_index_p, quad_index_u, quad_index_p);

    navier_stokes->solution.block(0).copy_locally_owned_data_from(vel_temp);
    navier_stokes->solution_old.block(0).copy_locally_owned_data_from(vel_temp_old);
    navier_stokes->solution_old_old.block(0).copy_locally_owned_data_from(vel_temp_old_old);

    navier_stokes->solution.block(1).copy_locally_owned_data_from(pressure_temp);
    navier_stokes->solution_old.block(1).copy_locally_owned_data_from(pressure_temp_old);
    navier_stokes->solution_old_old.block(1).copy_locally_owned_data_from(pressure_temp_old_old);
  }

  template <int dim>
  void
  AdafloWrapper<dim>::init_time_advance()
  {
    navier_stokes->time_stepping.set_time_step(time_iterator.get_current_time_increment());

    navier_stokes->init_time_advance(false);

    ready_for_time_advance = true;

    Assert(time_stepping_synchronized(),
           ExcMessage("Adaflo and MeltPoolDG time steppers are not aligned."));
  }

  template <int dim>
  void
  AdafloWrapper<dim>::solve()
  {
    if (!ready_for_time_advance)
      init_time_advance();

    navier_stokes->get_constraints_u().set_zero(navier_stokes->user_rhs.block(0));
    navier_stokes->get_constraints_p().set_zero(navier_stokes->user_rhs.block(1));

    const auto iter =
      navier_stokes->solve_nonlinear_system(navier_stokes->compute_initial_residual(true));

    {
      ScopedName sc("nonlinear_solve");
      IterationMonitor::add_linear_iterations(sc, iter.first);
    }

    {
      ScopedName sc("linear_solve");
      IterationMonitor::add_linear_iterations(sc, iter.second);
    }

    distribute_constraints();

    Journal::print_formatted_norm(
      scratch_data.get_pcout(0),
      [&]() -> double {
        return VectorTools::compute_norm<dim>(get_velocity(),
                                              scratch_data,
                                              get_dof_handler_idx_velocity(),
                                              get_quad_idx_velocity());
      },
      "velocity",
      "navier_stokes_adaflo",
      15 /*precision*/
    );

    Journal::print_formatted_norm(
      scratch_data.get_pcout(0),
      [&]() -> double {
        return VectorTools::compute_norm<dim>(get_pressure(),
                                              scratch_data,
                                              get_dof_handler_idx_pressure(),
                                              get_quad_idx_pressure());
      },
      "pressure",
      "navier_stokes_adaflo",
      15 /*precision*/
    );

    ready_for_time_advance = false;
  }



  template <int dim>
  void
  AdafloWrapper<dim>::set_face_average_density_augmented_taylor_hood(
    const Material<double> &material,
    const VectorType       &ls_as_heaviside,
    const unsigned int      ls_dof_idx,
    const VectorType       *temperature,
    const unsigned int      temp_dof_idx)
  {
    if (adaflo_params.augmented_taylor_hood == false)
      return;

    FEValues<dim> ls_values(scratch_data.get_mapping(),
                            scratch_data.get_fe(ls_dof_idx),
                            get_face_center_quad(),
                            update_values);

    std::unique_ptr<FEValues<dim>> temp_values;
    if (temperature && material.has_dependency(Material<double>::FieldType::temperature))
      temp_values = std::make_unique<FEValues<dim>>(scratch_data.get_mapping(),
                                                    scratch_data.get_fe(temp_dof_idx),
                                                    get_face_center_quad(),
                                                    update_values);

    std::vector<double> hs(ls_values.n_quadrature_points);
    std::vector<double> temp(ls_values.n_quadrature_points);

    for (const auto &cell : scratch_data.get_triangulation().active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(ls_dof_idx));

            ls_values.reinit(ls_dof_cell);
            ls_values.get_function_values(ls_as_heaviside, hs);

            std::vector<double> rho_ls(ls_values.n_quadrature_points);

            TriaIterator<DoFCellAccessor<dim, dim, false>> temp_dof_cell(
              &scratch_data.get_triangulation(),
              cell->level(),
              cell->index(),
              &scratch_data.get_dof_handler(temp_dof_idx));

            if (temp_values)
              {
                temp_values->reinit(temp_dof_cell);
                temp_values->get_function_values(*temperature, temp);
              }

            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                if (temp_values)
                  {
                    const auto material_values =
                      material.template compute_parameters<double>(hs[f],
                                                                   temp[f],
                                                                   MaterialUpdateFlags::density);
                    set_face_average_density(cell, f, material_values.density);
                  }
                else
                  {
                    const auto material_values =
                      material.template compute_parameters<double>(hs[f],
                                                                   MaterialUpdateFlags::density);

                    set_face_average_density(cell, f, material_values.density);
                  }
              }
          }
      }
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_velocity() const
  {
    return navier_stokes->solution.block(0);
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_velocity()
  {
    return navier_stokes->solution.block(0);
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_velocity_old() const
  {
    return navier_stokes->solution_old.block(0);
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_velocity_old_old() const
  {
    return navier_stokes->solution_old_old.block(0);
  }

  template <int dim>
  const DoFHandler<dim> &
  AdafloWrapper<dim>::get_dof_handler_velocity() const
  {
    return navier_stokes->get_dof_handler_u();
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
    return navier_stokes->get_constraints_u();
  }

  template <int dim>
  AffineConstraints<double> &
  AdafloWrapper<dim>::get_constraints_velocity()
  {
    return navier_stokes->modify_constraints_u();
  }

  template <int dim>
  const AffineConstraints<double> &
  AdafloWrapper<dim>::get_hanging_node_constraints_velocity() const
  {
    return navier_stokes->get_hanging_node_constraints_u();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_pressure() const
  {
    return navier_stokes->solution.block(1);
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_pressure()
  {
    return navier_stokes->solution.block(1);
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_pressure_old() const
  {
    return navier_stokes->solution_old.block(1);
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  AdafloWrapper<dim>::get_pressure_old_old() const
  {
    return navier_stokes->solution_old_old.block(1);
  }

  template <int dim>
  const DoFHandler<dim> &
  AdafloWrapper<dim>::get_dof_handler_pressure() const
  {
    return navier_stokes->get_dof_handler_p();
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
    return navier_stokes->get_constraints_p();
  }

  template <int dim>
  AffineConstraints<double> &
  AdafloWrapper<dim>::get_constraints_pressure()
  {
    return navier_stokes->modify_constraints_p();
  }

  template <int dim>
  const AffineConstraints<double> &
  AdafloWrapper<dim>::get_hanging_node_constraints_pressure() const
  {
    return navier_stokes->get_hanging_node_constraints_p();
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_force_rhs(const LinearAlgebra::distributed::Vector<double> &vec)
  {
    navier_stokes->user_rhs.block(0) = vec;
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_mass_balance_rhs(const LinearAlgebra::distributed::Vector<double> &vec)
  {
    navier_stokes->user_rhs.block(1) = vec;
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_user_defined_material(
    std::function<
      Tensor<2, dim, VectorizedArray<double>>(const Tensor<2, dim, VectorizedArray<double>> &,
                                              const unsigned int,
                                              const unsigned int,
                                              const bool)> my_user_defined_material)
  {
    navier_stokes->set_user_defined_material(my_user_defined_material);
  }

  template <int dim>
  VectorizedArray<double> &
  AdafloWrapper<dim>::get_density(const unsigned int cell, const unsigned int q)
  {
    return navier_stokes->get_matrix().begin_densities(cell)[q];
  }

  template <int dim>
  const VectorizedArray<double> &
  AdafloWrapper<dim>::get_density(const unsigned int cell, const unsigned int q) const
  {
    return navier_stokes->get_matrix().begin_densities(cell)[q];
  }

  template <int dim>
  VectorizedArray<double> &
  AdafloWrapper<dim>::get_viscosity(const unsigned int cell, const unsigned int q)
  {
    return navier_stokes->get_matrix().begin_viscosities(cell)[q];
  }

  template <int dim>
  const VectorizedArray<double> &
  AdafloWrapper<dim>::get_viscosity(const unsigned int cell, const unsigned int q) const
  {
    return navier_stokes->get_matrix().begin_viscosities(cell)[q];
  }

  template <int dim>
  VectorizedArray<double> &
  AdafloWrapper<dim>::get_damping(const unsigned int cell, const unsigned int q)
  {
    return navier_stokes->get_matrix().begin_damping_coeff(cell)[q];
  }

  template <int dim>
  const VectorizedArray<double> &
  AdafloWrapper<dim>::get_damping(const unsigned int cell, const unsigned int q) const
  {
    return navier_stokes->get_matrix().begin_damping_coeff(cell)[q];
  }


  template <int dim>
  void
  AdafloWrapper<dim>::attach_vectors_u(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&navier_stokes->solution.block(0));
    vectors.push_back(&navier_stokes->solution_old.block(0));
    vectors.push_back(&navier_stokes->solution_old_old.block(0));
  }

  template <int dim>
  void
  AdafloWrapper<dim>::attach_vectors_p(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    vectors.push_back(&navier_stokes->solution.block(1));
    vectors.push_back(&navier_stokes->solution_old.block(1));
    vectors.push_back(&navier_stokes->solution_old_old.block(1));
  }

  template <int dim>
  void
  AdafloWrapper<dim>::distribute_constraints()
  {
    navier_stokes->get_hanging_node_constraints_u().distribute(navier_stokes->solution.block(0));
    navier_stokes->get_hanging_node_constraints_u().distribute(
      navier_stokes->solution_old.block(0));
    navier_stokes->get_hanging_node_constraints_u().distribute(
      navier_stokes->solution_old_old.block(0));

    navier_stokes->get_hanging_node_constraints_p().distribute(navier_stokes->solution.block(1));
    navier_stokes->get_hanging_node_constraints_p().distribute(
      navier_stokes->solution_old.block(1));
    navier_stokes->get_hanging_node_constraints_p().distribute(
      navier_stokes->solution_old_old.block(1));

    navier_stokes->get_constraints_u().distribute(navier_stokes->user_rhs.block(0));
    navier_stokes->get_constraints_p().distribute(navier_stokes->user_rhs.block(1));
  }

  template <int dim>
  void
  AdafloWrapper<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
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
     *  force (raw)
     */
    data_out.add_data_vector(get_dof_handler_velocity(),
                             navier_stokes->user_rhs.block(0),
                             std::vector<std::string>(dim, "force_rhs_velocity"),
                             vector_component_interpretation);

    /**
     *  force (projected)
     */
    if (data_out.is_requested("force_rhs_velocity_projected"))
      {
        scratch_data.initialize_dof_vector(force_rhs_velocity_projected, dof_index_u);
        VectorTools::project_vector<dim>(scratch_data.get_mapping(),
                                         get_dof_handler_velocity(),
                                         scratch_data.get_constraint(dof_index_u),
                                         scratch_data.get_quadrature(quad_index_u),
                                         navier_stokes->user_rhs.block(0),
                                         force_rhs_velocity_projected);
        data_out.add_data_vector(get_dof_handler_velocity(),
                                 force_rhs_velocity_projected,
                                 std::vector<std::string>(dim, "force_rhs_velocity_projected"),
                                 vector_component_interpretation);
      }

    /**
     *  mass balance source term (raw)
     */
    data_out.add_data_vector(get_dof_handler_pressure(),
                             navier_stokes->user_rhs.block(1),
                             "mass_balance_source_term");

    /**
     *  mass balance source term (projected)
     */
    if (data_out.is_requested("mass_balance_source_term_projected"))
      {
        scratch_data.initialize_dof_vector(mass_balance_source_term_projected, dof_index_p);
        VectorTools::project_vector<1>(scratch_data.get_mapping(),
                                       get_dof_handler_pressure(),
                                       scratch_data.get_constraint(dof_index_p),
                                       scratch_data.get_quadrature(quad_index_p),
                                       navier_stokes->user_rhs.block(1),
                                       mass_balance_source_term_projected);
        data_out.add_data_vector(get_dof_handler_pressure(),
                                 mass_balance_source_term_projected,
                                 "mass_balance_source_term_projected");
      }

    /**
     *  density
     */
    if (data_out.is_requested("density"))
      {
        scratch_data.initialize_dof_vector(density, dof_index_parameters);

        if (scratch_data.is_hex_mesh())
          {
            MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
              density,
              scratch_data.get_matrix_free(),
              dof_index_parameters,
              quad_index_u,
              [&](const unsigned int cell, const unsigned int quad)
                -> const VectorizedArray<double> & { return get_density(cell, quad); });
            scratch_data.get_constraint(dof_index_parameters).distribute(density);
          }

        data_out.add_data_vector(dof_handler_parameters, density, "density");
      }

    /**
     *  viscosity
     */
    if (data_out.is_requested("viscosity"))
      {
        scratch_data.initialize_dof_vector(viscosity, dof_index_parameters);
        if (scratch_data.is_hex_mesh())
          {
            MeltPoolDG::VectorTools::fill_dof_vector_from_cell_operation<dim, 1>(
              viscosity,
              scratch_data.get_matrix_free(),
              dof_index_parameters,
              quad_index_u,
              [&](const unsigned int cell, const unsigned int quad)
                -> const VectorizedArray<double> & { return get_viscosity(cell, quad); });
            scratch_data.get_constraint(dof_index_parameters).distribute(viscosity);
          }

        data_out.add_data_vector(dof_handler_parameters, viscosity, "viscosity");
      }
  }



  template <int dim>
  void
  AdafloWrapper<dim>::attach_output_vectors_failed_step(GenericDataOut<dim> &data_out) const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      vector_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(scratch_data.get_dof_handler(get_dof_handler_idx_velocity()),
                             navier_stokes->solution_update.block(0),
                             std::vector<std::string>(dim, "velocity_last_solution_update"),
                             vector_component_interpretation,
                             true /* force output */);
    data_out.add_data_vector(scratch_data.get_dof_handler(get_dof_handler_idx_pressure()),
                             navier_stokes->solution_update.block(1),
                             "pressure_last_solution_update",
                             true /* force output */);
    data_out.add_data_vector(scratch_data.get_dof_handler(get_dof_handler_idx_velocity()),
                             navier_stokes->get_system_rhs().block(0),
                             std::vector<std::string>(dim, "velocity_newton_failed_residual"),
                             vector_component_interpretation,
                             true /* force output */);
    data_out.add_data_vector(scratch_data.get_dof_handler(get_dof_handler_idx_pressure()),
                             navier_stokes->get_system_rhs().block(1),
                             "pressure_newton_failed_residual",
                             true /* force output */);
  }

  template <int dim>
  void
  AdafloWrapper<dim>::set_face_average_density(
    const typename Triangulation<dim>::cell_iterator &cell,
    const unsigned int                                face,
    const double                                      density)
  {
    navier_stokes->set_face_average_density(cell, face, density);
  }

  template <int dim>
  const Quadrature<dim> &
  AdafloWrapper<dim>::get_face_center_quad()
  {
    if (!face_center_quad)
      {
        // quadrature rule of face centers
        std::vector<Point<dim>> face_centers;
        Triangulation<dim>      tria;
        GridGenerator::hyper_cube(tria, 0, 1);
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          face_centers.push_back(tria.begin()->face(f)->center());

        face_center_quad = std::make_unique<Quadrature<dim>>(face_centers);
      }

    return *face_center_quad;
  }

  template <int dim>
  bool
  AdafloWrapper<dim>::time_stepping_synchronized()
  {
    return std::abs(navier_stokes->time_stepping.step_size() -
                    time_iterator.get_current_time_increment()) < 1e-16 &&
           navier_stokes->time_stepping.step_no() == time_iterator.get_current_time_step_number() &&
           std::abs(navier_stokes->time_stepping.old_step_size() -
                    time_iterator.get_old_time_increment()) < 1e-16 &&
           std::abs(navier_stokes->time_stepping.now() - time_iterator.get_current_time()) <
             1e-16 &&
           std::abs(navier_stokes->time_stepping.previous() - time_iterator.get_old_time()) < 1e-16;
  }

  template class AdafloWrapper<1>;
  template class AdafloWrapper<2>;
  template class AdafloWrapper<3>;
} // namespace MeltPoolDG::Flow
#endif
