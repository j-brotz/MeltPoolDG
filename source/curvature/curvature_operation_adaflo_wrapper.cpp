#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/curvature/curvature_operation_adaflo_wrapper.hpp>
#  include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  CurvatureOperationAdaflo<dim>::CurvatureOperationAdaflo(const ScratchData<dim> &scratch_data,
                                                          const int         advec_diff_dof_idx,
                                                          const int         normal_vec_dof_idx,
                                                          const int         curv_dof_idx,
                                                          const int         curv_quad_idx,
                                                          const VectorType &advected_field,
                                                          const Parameters<double> &data_in)
    : scratch_data(scratch_data)
    , advected_field(advected_field)
    , normal_vector_data(data_in.ls.normal_vec)
  {
    (void)normal_vec_dof_idx;

    /**
     * set parameters of adaflo
     */
    set_adaflo_parameters(data_in, advec_diff_dof_idx, curv_dof_idx, curv_quad_idx);

    /**
     * initialize the projection matrix
     */
    projection_matrix     = std::make_shared<BlockMatrixExtension>();
    ilu_projection_matrix = std::make_shared<BlockILUExtension>();
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::create_operator()
  {
    curvature_operation = std::make_shared<LevelSetOKZSolverComputeCurvature<dim>>(
      scratch_data.get_cell_sizes(),
      normal_vector_operation_adaflo->get_solution_normal_vector(),
      scratch_data.get_constraint(curv_adaflo_params.dof_index_curvature),
      scratch_data.get_constraint(
        curv_adaflo_params
          .dof_index_curvature), // @todo -- check adaflo --> hanging node constraints??
      epsilon_used,
      rhs,
      curv_adaflo_params,
      curvature_field,
      advected_field,
      scratch_data.get_matrix_free(),
      preconditioner,
      projection_matrix,
      ilu_projection_matrix);
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::create_normal_vector_operator()
  {
    normal_vector_operation_adaflo = std::make_shared<LevelSet::NormalVectorOperationAdaflo<dim>>(
      scratch_data,
      curv_adaflo_params.dof_index_ls,
      curv_adaflo_params.dof_index_normal,
      curv_adaflo_params.quad_index,
      advected_field,
      normal_vector_data,
      curv_adaflo_params.epsilon);
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::reinit()
  {
    /**
     *  initialize the dof vectors
     */
    initialize_vectors();

    compute_cell_diameters<dim>(scratch_data.get_matrix_free(),
                                curv_adaflo_params.dof_index_ls,
                                cell_diameters,
                                cell_diameter_min,
                                cell_diameter_max);

    epsilon_used = cell_diameter_max * curv_adaflo_params.epsilon;

    if (!normal_vector_operation_adaflo)
      create_normal_vector_operator();
    if (!curvature_operation)
      create_operator();

    /**
     * initialize the preconditioner -->  @todo: currently not used in adaflo
     */
    initialize_mass_matrix_diagonal<dim, double>(scratch_data.get_matrix_free(),
                                                 scratch_data.get_constraint(
                                                   curv_adaflo_params.dof_index_curvature),
                                                 curv_adaflo_params.dof_index_curvature,
                                                 curv_adaflo_params.quad_index,
                                                 preconditioner);

    initialize_projection_matrix<dim, double, VectorizedArray<double>>(
      scratch_data.get_matrix_free(),
      scratch_data.get_constraint(curv_adaflo_params.dof_index_curvature),
      curv_adaflo_params.dof_index_curvature,
      curv_adaflo_params.quad_index,
      epsilon_used,
      curv_adaflo_params.epsilon,
      cell_diameters,
      *projection_matrix,
      *ilu_projection_matrix);

    normal_vector_operation_adaflo->reinit();
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::update_normal_vector()
  {
    advected_field.update_ghost_values();
    normal_vector_operation_adaflo->solve();
    advected_field.zero_out_ghost_values();
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::solve()
  {
    advected_field.update_ghost_values();
    normal_vector_operation_adaflo->solve();
    curvature_operation->compute_curvature(
      false); // @todo: adaflo does not use the boolean function argument

    const int verbosity_l2_norm = dim > 1 ? 0 : 1;
    Journal::print_formatted_norm(
      scratch_data.get_pcout(verbosity_l2_norm),
      [&]() -> double {
        return VectorTools::compute_norm<dim>(get_curvature(),
                                              scratch_data,
                                              curv_adaflo_params.dof_index_curvature,
                                              curv_adaflo_params.quad_index);
      },
      "curvature",
      "curvature_adaflo",
      10 /*precision*/
    );
    advected_field.zero_out_ghost_values();
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  CurvatureOperationAdaflo<dim>::get_curvature() const
  {
    return curvature_field;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  CurvatureOperationAdaflo<dim>::get_curvature()
  {
    return curvature_field;
  }

  template <int dim>
  const LinearAlgebra::distributed::BlockVector<double> &
  CurvatureOperationAdaflo<dim>::get_normal_vector() const
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim>
  LinearAlgebra::distributed::BlockVector<double> &
  CurvatureOperationAdaflo<dim>::get_normal_vector()
  {
    return normal_vector_operation_adaflo->get_solution_normal_vector();
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    normal_vector_operation_adaflo->attach_vectors(vectors);
    vectors.push_back(&curvature_field);
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::set_adaflo_parameters(const Parameters<double> &parameters,
                                                       const int                 advec_diff_dof_idx,
                                                       const int                 curv_dof_idx,
                                                       const int                 curv_quad_idx)
  {
    curv_adaflo_params.dof_index_ls        = advec_diff_dof_idx;
    curv_adaflo_params.dof_index_curvature = curv_dof_idx; //@ todo
    curv_adaflo_params.dof_index_normal    = curv_dof_idx;
    curv_adaflo_params.quad_index          = curv_quad_idx;
    curv_adaflo_params.epsilon =
      parameters.ls.reinit.interface_thickness_parameter.value / parameters.ls.get_n_subdivisions();
    curv_adaflo_params.approximate_projections = false; //@ todo
    curv_adaflo_params.curvature_correction    = parameters.ls.curv.do_curvature_correction;
    verbosity_level                            = parameters.ls.curv.verbosity_level;
    // curv_adaflo_params.filter_parameter = parameters.ls.normal_vec.filter_parameter; //@
    // todo
  }

  template <int dim>
  void
  CurvatureOperationAdaflo<dim>::initialize_vectors()
  {
    /**
     * initialize advected field dof vectors
     */
    scratch_data.initialize_dof_vector(curvature_field, curv_adaflo_params.dof_index_curvature);
    scratch_data.initialize_dof_vector(normal_vec_dummy, curv_adaflo_params.dof_index_curvature);
    /**
     * initialize vectors for the solution of the linear system
     */
    scratch_data.initialize_dof_vector(rhs, curv_adaflo_params.dof_index_curvature);
  }

  template class CurvatureOperationAdaflo<1>;
  template class CurvatureOperationAdaflo<2>;
  template class CurvatureOperationAdaflo<3>;
} // namespace MeltPoolDG::LevelSet
#endif
