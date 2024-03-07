#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>
#  include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim>
  NormalVectorOperationAdaflo<dim>::NormalVectorOperationAdaflo(
    const ScratchData<dim>         &scratch_data,
    const int                       advec_diff_dof_idx,
    const int                       normal_vec_dof_idx,
    const int                       normal_vec_quad_idx,
    const VectorType               &advected_field_in,
    const NormalVectorData<double> &data_in,
    const double                    reinit_scale_factor_epsilon)
    : scratch_data(scratch_data)
    , advected_field(advected_field_in)
    , normal_vector_field(dim)
    , rhs(dim)
  {
    /**
     * set parameters of adaflo
     */
    set_adaflo_parameters(data_in,
                          reinit_scale_factor_epsilon,
                          advec_diff_dof_idx,
                          normal_vec_dof_idx,
                          normal_vec_quad_idx);

    /**
     * initialize the projection matrix
     */
    projection_matrix     = std::make_shared<BlockMatrixExtension>();
    ilu_projection_matrix = std::make_shared<BlockILUExtension>();
  }

  template <int dim>
  void
  NormalVectorOperationAdaflo<dim>::create_operator()
  {
    normal_vec_operation = std::make_shared<LevelSetOKZSolverComputeNormal<dim>>(
      normal_vector_field,
      rhs,
      advected_field,
      scratch_data.get_cell_sizes(),
      epsilon_used,
      cell_diameter_min,
      scratch_data.get_constraint(normal_vec_adaflo_params.dof_index_normal),
      normal_vec_adaflo_params,
      scratch_data.get_matrix_free(),
      preconditioner,
      projection_matrix,
      ilu_projection_matrix);
  }

  template <int dim>
  void
  NormalVectorOperationAdaflo<dim>::reinit()
  {
    /**
     *  initialize the dof vectors
     */
    initialize_vectors();

    compute_cell_diameters<dim>(scratch_data.get_matrix_free(),
                                normal_vec_adaflo_params.dof_index_ls,
                                cell_diameters,
                                cell_diameter_min,
                                cell_diameter_max);

    epsilon_used = cell_diameter_max * normal_vec_adaflo_params.epsilon;

    if (!normal_vec_operation)
      create_operator();

    /**
     * initialize the preconditioner -->  @todo: currently not used in adaflo
     */
    initialize_mass_matrix_diagonal<dim, double>(scratch_data.get_matrix_free(),
                                                 scratch_data.get_constraint(
                                                   normal_vec_adaflo_params.dof_index_normal),
                                                 normal_vec_adaflo_params.dof_index_normal,
                                                 normal_vec_adaflo_params.quad_index,
                                                 preconditioner);


    initialize_projection_matrix<dim, double, VectorizedArray<double>>(
      scratch_data.get_matrix_free(),
      scratch_data.get_constraint(normal_vec_adaflo_params.dof_index_normal),
      normal_vec_adaflo_params.dof_index_normal,
      normal_vec_adaflo_params.quad_index,
      epsilon_used,
      normal_vec_adaflo_params.epsilon,
      scratch_data.get_cell_sizes(),
      *projection_matrix,
      *ilu_projection_matrix);
  }

  template <int dim>
  void
  NormalVectorOperationAdaflo<dim>::solve()
  {
    initialize_vectors();
    normal_vec_operation->compute_normal(false /* fast computation*/);

    const int verbosity_l2_norm = dim > 1 ? 0 : 1;

    for (unsigned int d = 0; d < dim; ++d)
      Journal::print_formatted_norm(
        scratch_data.get_pcout(verbosity_l2_norm),
        [&]() -> double {
          return MeltPoolDG::VectorTools::compute_norm<dim>(
            get_solution_normal_vector().block(d),
            scratch_data,
            normal_vec_adaflo_params.dof_index_normal,
            normal_vec_adaflo_params.quad_index);
        },
        "normal_" + std::to_string(d),
        "normal_vector_adaflo",
        10 /*precision*/
      );
  }

  template <int dim>
  const LinearAlgebra::distributed::BlockVector<double> &
  NormalVectorOperationAdaflo<dim>::get_solution_normal_vector() const
  {
    return normal_vector_field;
  }

  template <int dim>
  LinearAlgebra::distributed::BlockVector<double> &
  NormalVectorOperationAdaflo<dim>::get_solution_normal_vector()
  {
    return normal_vector_field;
  }

  template <int dim>
  LevelSetOKZSolverComputeNormal<dim> &
  NormalVectorOperationAdaflo<dim>::get_adaflo_obj()
  {
    return *normal_vec_operation;
  }

  template <int dim>
  void
  NormalVectorOperationAdaflo<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    for (unsigned int d = 0; d < dim; ++d)
      vectors.push_back(&normal_vector_field.block(d));
  }

  template <int dim>
  void
  NormalVectorOperationAdaflo<dim>::set_adaflo_parameters(
    const NormalVectorData<double> &normal_vec_data,
    const double                    epsilon,
    const int                       advec_diff_dof_idx,
    const int                       normal_vec_dof_idx,
    const int                       normal_vec_quad_idx)
  {
    normal_vec_adaflo_params.dof_index_ls            = advec_diff_dof_idx;
    normal_vec_adaflo_params.dof_index_normal        = normal_vec_dof_idx;
    normal_vec_adaflo_params.quad_index              = normal_vec_quad_idx;
    normal_vec_adaflo_params.damping_scale_factor    = normal_vec_data.filter_parameter;
    normal_vec_adaflo_params.epsilon                 = epsilon;
    normal_vec_adaflo_params.approximate_projections = false; // not used in adaflo
  }

  template <int dim>
  void
  NormalVectorOperationAdaflo<dim>::initialize_vectors()
  {
    /**
     * initialize advected field dof vectors
     */
    scratch_data.initialize_dof_vector(normal_vector_field,
                                       normal_vec_adaflo_params.dof_index_normal);
    /**
     * initialize vectors for the solution of the linear system
     */
    scratch_data.initialize_dof_vector(rhs, normal_vec_adaflo_params.dof_index_normal);
  }


  template class NormalVectorOperationAdaflo<1>;
  template class NormalVectorOperationAdaflo<2>;
  template class NormalVectorOperationAdaflo<3>;
} // namespace MeltPoolDG::LevelSet
#endif
