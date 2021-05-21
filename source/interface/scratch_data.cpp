#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG
{
  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  ScratchData<dim, spacedim, number, VectorizedArrayType>::ScratchData(
    const MPI_Comm     mpi_communicator,
    const unsigned int max_verbosity_level,
    const bool         do_matrix_free)
    : do_matrix_free(do_matrix_free)
  {
    this->pcout.clear();

    for (unsigned int i = 0; i <= 10; ++i)
      this->pcout.push_back(
        ConditionalOStream(std::cout,
                           (Utilities::MPI::this_mpi_process(mpi_communicator)) == 0 &&
                             (i <= max_verbosity_level)));
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  ScratchData<dim, spacedim, number, VectorizedArrayType>::ScratchData(
    const ScratchData &scratch_data)
  {
    do_matrix_free = scratch_data.do_matrix_free;

    //@todo: check if mapping is deleted
    this->reinit(scratch_data.get_mapping(),
                 scratch_data.get_dof_handlers(),
                 scratch_data.get_constraints(),
                 scratch_data.get_quadratures());
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::reinit(
    const Mapping<dim, spacedim> &                        mapping,
    const std::vector<const DoFHandler<dim, spacedim> *> &dof_handler,
    const std::vector<const AffineConstraints<number> *> &constraint,
    const std::vector<Quadrature<dim>> &                  quad)
  {
    this->clear();

    set_mapping(mapping);

    for (unsigned int i = 0; i < dof_handler.size(); ++i)
      this->attach_dof_handler(*dof_handler[i]);

    for (unsigned int i = 0; i < constraint.size(); ++i)
      this->attach_constraint_matrix(*constraint[i]);

    for (unsigned int i = 0; i < quad.size(); ++i)
      this->attach_quadrature(quad[i]);

    this->create_partitioning();

    this->build();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::set_mapping(
    const Mapping<dim, spacedim> &mapping)
  {
    this->mapping = mapping.clone();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  unsigned int
  ScratchData<dim, spacedim, number, VectorizedArrayType>::attach_dof_handler(
    const DoFHandler<dim, spacedim> &dof_handler)
  {
    this->dof_handler.emplace_back(&dof_handler);
    return this->dof_handler.size() - 1;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  unsigned int
  ScratchData<dim, spacedim, number, VectorizedArrayType>::attach_constraint_matrix(
    const AffineConstraints<number> &constraint)
  {
    this->constraint.emplace_back(&constraint);
    return this->constraint.size() - 1;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::create_partitioning()
  {
    /*
     *  recreate DoF-dependent partitioning data
     */
    this->min_cell_size.clear();
    this->diameter = 0.0;
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();
    this->partitioner.clear();

    int dof_idx = 0;
    for (const auto &dof : dof_handler)
      {
        /*
         *  create vector of minimum cell sizes
         */
        this->min_cell_size.push_back(GridTools::minimal_cell_diameter(dof->get_triangulation()) /
                                      std::sqrt(dim));
        /*
         *  create diameter of the object
         */
        if (dof_idx == 0)
          // @todo: this should actually mean GridTools::diameter; however dealii does not
          // support diameter for parallel triangulations atm
          this->diameter = GridTools::minimal_cell_diameter(dof->get_triangulation());
        /*
         *  create partitioning
         */
        this->locally_owned_dofs.push_back(dof->locally_owned_dofs());

        IndexSet locally_relevant_dofs_temp;
        DoFTools::extract_locally_relevant_dofs(*dof, locally_relevant_dofs_temp);
        this->locally_relevant_dofs.push_back(locally_relevant_dofs_temp);

        this->partitioner.push_back(
          std::make_shared<Utilities::MPI::Partitioner>(this->get_locally_owned_dofs(dof_idx),
                                                        this->get_locally_relevant_dofs(dof_idx),
                                                        this->get_mpi_comm(dof_idx)));
        dof_idx += 1;
      }
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::build()
  {
    AssertThrow(this->constraint.size() == this->dof_handler.size(),
                ExcMessage(
                  "The number of DoFHandlers and AffineConstraints attached to ScratchData<dim>"
                  " must be equal."));

    if (do_matrix_free)
      {
        this->matrix_free.clear();

        typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;
        additional_data.mapping_update_flags =
          update_values | update_gradients | update_JxW_values | dealii::update_quadrature_points;

        additional_data.mapping_update_flags_boundary_faces =
          update_values | update_gradients | update_JxW_values | dealii::update_quadrature_points;

        this->matrix_free.reinit(
          *this->mapping, this->dof_handler, this->constraint, this->quad, additional_data);

        this->cell_diameters.clear();

        /*
         *  create vector of cell diameters for matrix free
         */
        int dof_idx = 0;
        for (const auto &dof : dof_handler)
          {
            (void)dof;
            AlignedVector<VectorizedArray<double>> cell_diameters_temp;
            cell_diameters_temp.resize(this->matrix_free.n_cell_batches());

            FullMatrix<double> mat(dim, dim);

            for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
              {
                VectorizedArray<double> diameter = VectorizedArray<double>();
                for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
                  diameter[v] = this->matrix_free.get_cell_iterator(cell, v, dof_idx)->diameter();
                cell_diameters_temp[cell] = make_vectorized_array<double>(0.0); // diameter;
              }
            this->cell_diameters.push_back(cell_diameters_temp);
            dof_idx += 1;
          }
      }
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::initialize_dof_vector(
    VectorType &       vec,
    const unsigned int dof_idx) const
  {
    if (do_matrix_free)
      matrix_free.initialize_dof_vector(vec, dof_idx);
    else
      vec.reinit(get_locally_owned_dofs(dof_idx),
                 get_locally_relevant_dofs(dof_idx),
                 get_mpi_comm(dof_idx));
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::initialize_dof_vector(
    BlockVectorType &  vec,
    const unsigned int dof_idx) const
  {
    vec.reinit(dim);
    for (unsigned int d = 0; d < dim; ++d)
      this->initialize_dof_vector(vec.block(d), dof_idx);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::initialize_bc_vector(
    VectorType &       vec,
    const unsigned int dof_idx) const
  {
    this->initialize_dof_vector(vec, dof_idx);
    this->get_constraint(dof_idx).distribute(vec);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::initialize_bc_vector(
    BlockVectorType &  vec,
    const unsigned int dof_idx) const
  {
    vec.reinit(dim);
    for (unsigned int d = 0; d < dim; ++d)
      this->initialize_bc_vector(vec.block(d), dof_idx);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::clear()
  {
    this->matrix_free.clear();
    this->quad.clear();
    this->constraint.clear();
    this->dof_handler.clear();
    this->mapping.reset();
    this->min_cell_size.clear();
    this->diameter = 0.0;
    this->cell_diameters.clear();
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();
    this->pcout.clear();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const Mapping<dim, spacedim> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_mapping() const
  {
    return *this->mapping;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const FiniteElement<dim, spacedim> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_fe(const unsigned int fe_index) const
  {
    return this->dof_handler[fe_index]->get_fe(0);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const AffineConstraints<number> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_constraint(
    const unsigned int constraint_index) const
  {
    return *this->constraint[constraint_index];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  AffineConstraints<number> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_constraint(
    const unsigned int constraint_index)
  {
    return const_cast<AffineConstraints<number> &>(*this->constraint[constraint_index]);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const std::vector<const AffineConstraints<number> *> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_constraints() const
  {
    return this->constraint;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const Quadrature<dim> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_quadrature(
    const unsigned int quad_index) const
  {
    return this->quad[quad_index];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const std::vector<Quadrature<dim>> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_quadratures() const
  {
    return this->quad;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const Quadrature<dim - 1> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_face_quadrature(
    const unsigned int quad_index) const
  {
    return this->face_quad[quad_index];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const std::vector<Quadrature<dim - 1>> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_face_quadratures() const
  {
    return this->face_quad;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  MatrixFree<dim, number, VectorizedArrayType> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_matrix_free()
  {
    return this->matrix_free;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const MatrixFree<dim, number, VectorizedArrayType> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_matrix_free() const
  {
    return this->matrix_free;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const DoFHandler<dim, spacedim> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_dof_handler(
    const unsigned int dof_idx) const
  {
    return *this->dof_handler[dof_idx];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const std::vector<const DoFHandler<dim, spacedim> *> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_dof_handlers() const
  {
    return this->dof_handler;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const Triangulation<dim> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_triangulation(
    const unsigned int dof_idx) const
  {
    return this->get_dof_handler(dof_idx).get_triangulation();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  unsigned int
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_n_dofs_per_cell(
    const unsigned int dof_idx) const
  {
    return get_dof_handler(dof_idx).get_fe().n_dofs_per_cell();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  unsigned int
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_degree(
    const unsigned int dof_idx) const
  {
    return get_dof_handler(dof_idx).get_fe().tensor_degree();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const double &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_min_cell_size(
    const unsigned int dof_idx) const
  {
    return this->min_cell_size[dof_idx];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const double &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_diameter() const
  {
    return this->diameter;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const AlignedVector<VectorizedArray<double>> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_cell_diameters(
    const unsigned int dof_idx) const
  {
    return this->cell_diameters[dof_idx];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  MPI_Comm
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_mpi_comm(
    const unsigned int dof_idx) const
  {
    return UtilityFunctions::get_mpi_comm(*this->dof_handler[dof_idx]);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const IndexSet &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_locally_owned_dofs(
    const unsigned int dof_idx) const
  {
    return this->locally_owned_dofs[dof_idx];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const IndexSet &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_locally_relevant_dofs(
    const unsigned int dof_idx) const
  {
    return this->locally_relevant_dofs[dof_idx];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const std::shared_ptr<Utilities::MPI::Partitioner> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_partitioner(
    const unsigned int dof_idx) const
  {
    return this->partitioner[dof_idx];
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const ConditionalOStream
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_pcout(const unsigned int level) const
  {
    AssertIndexRange(level, pcout.size());
    return pcout[level];
  }

  template class ScratchData<1>;
  template class ScratchData<2>;
  template class ScratchData<3>;
} // namespace MeltPoolDG
