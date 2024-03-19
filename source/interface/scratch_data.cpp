#include <deal.II/fe/fe_q_iso_q1.h>

#include <deal.II/matrix_free/util.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/journal.hpp>

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
        dealii::ConditionalOStream(std::cout,
                                   (Utilities::MPI::this_mpi_process(mpi_communicator)) == 0 &&
                                     (i <= max_verbosity_level)));

    timer = std::make_shared<TimerOutput>(pcout[0], TimerOutput::never, TimerOutput::wall_times);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::reinit(
    const Mapping<dim, spacedim>                         &mapping,
    const std::vector<const DoFHandler<dim, spacedim> *> &dof_handler,
    const std::vector<const AffineConstraints<number> *> &constraint,
    const std::vector<Quadrature<dim>>                   &quad,
    const bool                                            enable_boundary_face_loops,
    const bool                                            enable_inner_face_loops)
  {
    enable_inner_faces    = enable_inner_face_loops;
    enable_boundary_faces = enable_boundary_face_loops;

    this->clear();

    set_mapping(mapping);

    for (unsigned int i = 0; i < dof_handler.size(); ++i)
      this->attach_dof_handler(*dof_handler[i]);

    for (unsigned int i = 0; i < constraint.size(); ++i)
      this->attach_constraint_matrix(*constraint[i]);

    for (unsigned int i = 0; i < quad.size(); ++i)
      this->attach_quadrature(quad[i]);

    this->create_partitioning();

    this->build(enable_boundary_face_loops, enable_inner_face_loops);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::set_mapping(
    const Mapping<dim, spacedim> &mapping)
  {
    this->mapping = mapping.clone();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::set_mapping(
    const std::shared_ptr<Mapping<dim, spacedim>> mapping)
  {
    this->mapping = mapping;
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
  unsigned int
  ScratchData<dim, spacedim, number, VectorizedArrayType>::attach_quadrature(
    const Quadrature<dim> &quadrature)
  {
    this->quad.emplace_back(Quadrature<dim>(quadrature));

    // determine face quadrature like it is done in MatrixFree
    // https://github.com/dealii/dealii/blob/2946051880b5c674f141397219349fbfa579a6ac/include/deal.II/matrix_free/mapping_info.templates.h#L382-L439
    bool flag = quadrature.is_tensor_product();

    if (flag)
      for (unsigned int i = 1; i < dim; ++i)
        flag &= quadrature.get_tensor_basis()[0] == quadrature.get_tensor_basis()[i];

    if (flag) // hex element
      {
        this->face_quad.emplace_back(quadrature.get_tensor_basis()[0]);
      }
    else // simplex element
      {
        const auto unique_face_quadratures =
          internal::MatrixFreeFunctions::get_unique_face_quadratures(quadrature);

        // make sure we have not got wedges or pyramids
        AssertDimension(unique_face_quadratures.second.size(), 0);

        this->face_quad.emplace_back(unique_face_quadratures.first);
      }

    return this->quad.size() - 1;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::create_partitioning()
  {
    /*
     *  recreate DoF-dependent partitioning data
     */
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();
    this->partitioner.clear();

    /*
     *  create diameter of the object
     */
    this->min_diameter = GridTools::minimal_cell_diameter(get_triangulation());

    /*
     *  compute minimum cell size; this corresponds to the edge length in case of cubic elements
     */
    this->min_cell_size = this->min_diameter / std::sqrt(dim);
    this->max_cell_size = GridTools::maximal_cell_diameter(get_triangulation()) / std::sqrt(dim);

    int dof_idx = 0;
    for (const auto &dof : dof_handler)
      {
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
  ScratchData<dim, spacedim, number, VectorizedArrayType>::build(
    const bool enable_boundary_face_loops,
    const bool enable_inner_face_loops)
  {
    enable_inner_faces    = enable_inner_face_loops;
    enable_boundary_faces = enable_boundary_face_loops;

    AssertThrow(this->constraint.size() == this->dof_handler.size(),
                ExcMessage(
                  "The number of DoFHandlers and AffineConstraints attached to ScratchData<dim>"
                  " must be equal."));

    if (do_matrix_free)
      {
        this->matrix_free.clear();

        typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;

        additional_data.overlap_communication_computation = false;

        additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values | dealii::update_quadrature_points);


        if (enable_inner_face_loops)
          {
            Journal::print_line(get_pcout(2),
                                "Matrix-free: set update flags for inner face loops",
                                "ScratchData",
                                0);
            additional_data.mapping_update_flags_inner_faces =
              (update_values | update_gradients | update_JxW_values |
               dealii::update_quadrature_points);
          }

        if (enable_boundary_face_loops)
          {
            Journal::print_line(get_pcout(2),
                                "Matrix-free: set update flags for boundary face loops",
                                "ScratchData",
                                0);
            additional_data.mapping_update_flags_boundary_faces =
              (update_values | update_gradients | update_JxW_values |
               dealii::update_quadrature_points);
          }

        this->matrix_free.reinit(
          *this->mapping, this->dof_handler, this->constraint, this->quad, additional_data);

        this->cell_sizes.clear();

        /*
         *  create vector of cell sizes for matrix free
         */
        this->cell_sizes.resize(this->matrix_free.n_cell_batches());

        for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
          {
            VectorizedArray<double> cell_size = VectorizedArray<double>();
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
              {
                // the diameter is subdivided by sqrt(dim) to get the edge length for quadratic
                // elements
                cell_size[v] =
                  this->matrix_free.get_cell_iterator(cell, v, 0 /*dof_idx*/)->diameter() /
                  sqrt(dim);

                Assert(cell_size[v] > 0.0,
                       ExcMessage("The calculated diameter should be larger than zero."));
              }
            this->cell_sizes[cell] = cell_size;
          }
      }
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::initialize_dof_vector(
    VectorType        &vec,
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
    BlockVectorType   &vec,
    const unsigned int dof_idx) const
  {
    vec.reinit(dim);
    for (unsigned int d = 0; d < dim; ++d)
      this->initialize_dof_vector(vec.block(d), dof_idx);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::initialize_bc_vector(
    VectorType        &vec,
    const unsigned int dof_idx) const
  {
    this->initialize_dof_vector(vec, dof_idx);
    this->get_constraint(dof_idx).distribute(vec);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::initialize_bc_vector(
    BlockVectorType   &vec,
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
    this->min_cell_size = 0.0;
    this->min_diameter  = 0.0;
    this->cell_sizes.clear();
    this->locally_owned_dofs.clear();
    this->locally_relevant_dofs.clear();
    this->pcout.clear();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  void
  ScratchData<dim, spacedim, number, VectorizedArrayType>::create_remote_point_evaluation(
    const unsigned int                        dof_idx,
    const std::function<std::vector<bool>()> &marked_vertices)
  {
    if (!rpe.contains(dof_idx))
      rpe.insert({dof_idx,
                  std::make_shared<Utilities::MPI::RemotePointEvaluation<dim, dim>>(
                    1e-6 /*tolerance*/, true /*unique mapping*/, 0, marked_vertices)});
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
  unsigned int
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_n_q_points(
    const unsigned int quad_idx) const
  {
    return get_quadrature(quad_idx).size();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const double &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_min_cell_size() const
  {
    return this->min_cell_size;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  double
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_min_cell_size(
    const unsigned int dof_idx) const
  {
    return is_FE_Q_iso_Q_1(dof_idx) ? min_cell_size : min_cell_size / get_degree(dof_idx);
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const double &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_max_cell_size() const
  {
    return this->max_cell_size;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const double &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_min_diameter() const
  {
    return this->min_diameter;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  const AlignedVector<VectorizedArray<double>> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_cell_sizes() const
  {
    return this->cell_sizes;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  MPI_Comm
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_mpi_comm(
    const unsigned int dof_idx) const
  {
    return this->dof_handler[dof_idx]->get_communicator();
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

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  bool
  ScratchData<dim, spacedim, number, VectorizedArrayType>::is_hex_mesh(
    const unsigned int dof_idx) const
  {
    return get_triangulation(dof_idx).all_reference_cells_are_hyper_cube();
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  bool
  ScratchData<dim, spacedim, number, VectorizedArrayType>::is_FE_Q_iso_Q_1(
    const unsigned int dof_idx) const
  {
    return dynamic_cast<const FE_Q_iso_Q1<dim> *>(&this->get_fe(dof_idx)) != nullptr;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  TimerOutput &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_timer() const
  {
    return *timer;
  }

  template <int dim, int spacedim, typename number, typename VectorizedArrayType>
  Utilities::MPI::RemotePointEvaluation<dim, dim> &
  ScratchData<dim, spacedim, number, VectorizedArrayType>::get_remote_point_evaluation(
    const unsigned int dof_idx) const
  {
    return *rpe.at(dof_idx);
  }

  template class ScratchData<1>;
  template class ScratchData<2>;
  template class ScratchData<3>;
} // namespace MeltPoolDG
