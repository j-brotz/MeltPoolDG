#include <meltpooldg/level_set/normal_vector_operation.hpp>
#include <meltpooldg/level_set/normal_vector_operator.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  NormalVectorOperation<dim, number>::NormalVectorOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const NormalVectorData<number>      &normal_vector_data,
    const VectorType                    &solution_level_set,
    const std::array<unsigned int, dim> &normal_dof_indices_per_block_in,
    const unsigned int                   normal_no_bc_dof_idx_in,
    const unsigned int                   normal_quad_idx_in,
    const unsigned int                   ls_dof_idx_in)
    : scratch_data(scratch_data_in)
    , normal_vector_data(normal_vector_data)
    , solution_level_set(solution_level_set)
    , normal_no_bc_dof_idx(normal_no_bc_dof_idx_in)
    , normal_dof_indices_per_block(normal_dof_indices_per_block_in)
    , normal_quad_idx(normal_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , solution_history(normal_vector_data.predictor.n_old_solution_vectors)
  {
    if (!normal_vector_operator)
      create_operator();
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::reinit()
  {
    // const std::array<unsigned int, dim> dof_indices_per_block = [this]() {
    //   if constexpr (dim == 1)
    //     return std::array<unsigned int, 1>{{normal_dirichlet_x_dof_idx}};
    //   else if constexpr (dim == 2)
    //     return std::array<unsigned int, 2>{
    //       {normal_dirichlet_x_dof_idx, normal_dirichlet_y_dof_idx}};
    //   else if constexpr (dim == 3)
    //     return std::array<unsigned int, 3>{
    //       {normal_dirichlet_x_dof_idx, normal_dirichlet_y_dof_idx, normal_dirichlet_z_dof_idx}};
    // }();

    solution_history.apply([this](BlockVectorType &v) {
      scratch_data.initialize_dof_vector(v, normal_dof_indices_per_block);
    });

    scratch_data.initialize_dof_vector(solution_normal_vector_predictor,
                                       normal_dof_indices_per_block);
    scratch_data.initialize_dof_vector(rhs, normal_dof_indices_per_block);

    normal_vector_operator->reinit();

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        // precompute preconditioner
        const bool update_ghosts = !solution_history.get_current_solution().has_ghost_elements();
        if (update_ghosts)
          solution_history.get_current_solution().update_ghost_values();

        preconditioner.reinit(scratch_data, normal_dof_indices_per_block[0]);
        preconditioner.update();

        if (update_ghosts)
          solution_history.get_current_solution().zero_out_ghost_values();
      }
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::solve()
  {
    const ScopedName   sc("normal::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    // Create wetting constraints if requested
    this->create_wetting_constraints();
    this->create_contact_angle_constraints();

    const bool update_ghosts = !solution_level_set.has_ghost_elements();
    if (update_ghosts)
      solution_level_set.update_ghost_values();

    // compute predictor
    if (!predictor)
      predictor = std::make_unique<Predictor<BlockVectorType, number>>(normal_vector_data.predictor,
                                                                       solution_history);

    // const std::array<unsigned int, dim> dof_indices_per_block = [this]() {
    //   if constexpr (dim == 1)
    //     return std::array<unsigned int, 1>{{normal_dirichlet_x_dof_idx}};
    //   else if constexpr (dim == 2)
    //     return std::array<unsigned int, 2>{
    //       {normal_dirichlet_x_dof_idx, normal_dirichlet_y_dof_idx}};
    //   else if constexpr (dim == 3)
    //     return std::array<unsigned int, 3>{
    //       {normal_dirichlet_x_dof_idx, normal_dirichlet_y_dof_idx, normal_dirichlet_z_dof_idx}};
    // }();

    if (normal_vector_data.linear_solver.do_matrix_free &&
        normal_vector_data.predictor.type == PredictorType::least_squares_projection)
      {
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree<dim, number>(
          *normal_vector_operator,
          rhs,
          solution_level_set,
          scratch_data,
          normal_dof_indices_per_block,
          normal_no_bc_dof_idx,
          false,
          wetting_constraints_indices_and_values);
      }

    predictor->vmult(*normal_vector_operator, solution_normal_vector_predictor, rhs);

    for (unsigned int d = 0; d < dim; ++d)
      scratch_data.get_constraint(normal_dof_indices_per_block[d])
        .distribute(solution_history.get_current_solution().block(d));

    unsigned int iter = 0;

    if (normal_vector_data.linear_solver.do_matrix_free)
      {
        //  Apply wetting boundary conditions
        Utilities::MatrixFree::create_rhs_and_apply_dirichlet_matrixfree<dim, number>(
          *normal_vector_operator,
          rhs,
          solution_level_set,
          scratch_data,
          normal_dof_indices_per_block,
          normal_no_bc_dof_idx,
          false,
          wetting_constraints_indices_and_values);

        normal_vector_operator->enable_pre_post();

        iter = LinearSolver::solve<BlockVectorType>(*normal_vector_operator,
                                                    solution_history.get_current_solution(),
                                                    rhs,
                                                    normal_vector_data.linear_solver,
                                                    preconditioner,
                                                    "normal_vector_operation");
        normal_vector_operator->disable_pre_post();
      }
    else
      {
        normal_vector_operator->compute_system_matrix_and_rhs(solution_level_set, rhs);

        for (unsigned int d = 0; d < dim; ++d)
          iter = LinearSolver::solve<VectorType>(normal_vector_operator->get_system_matrix(),
                                                 solution_history.get_current_solution().block(d),
                                                 rhs.block(d),
                                                 normal_vector_data.linear_solver,
                                                 PreconditionIdentity(),
                                                 "normal_vector_operation");
      }

    // Wetting boundary condition
    if (not wetting_bc_map.empty())
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            for (unsigned int i = 0; i < wetting_constraints_indices_and_values.first.size(); ++i)
              {
                solution_history.get_current_solution().block(d).local_element(
                  wetting_constraints_indices_and_values.first[i]) =
                  wetting_constraints_indices_and_values.second[d][i];
              }
          }
      }

    if (update_ghosts)
      solution_level_set.zero_out_ghost_values();

    for (unsigned int d = 0; d < dim; ++d)
      {
        scratch_data.get_constraint(normal_dof_indices_per_block[d])
          .distribute(solution_history.get_current_solution().block(d));
      }
    constexpr int             verbosity_l2_norm = dim > 1 ? 1 : 2;
    const ConditionalOStream &pcout =
      scratch_data.get_pcout(std::max(normal_vector_data.verbosity_level, verbosity_l2_norm));

    for (unsigned int d = 0; d < dim; ++d)
      Journal::print_formatted_norm<number>(
        pcout,
        [&]() -> number {
          return MeltPoolDG::VectorTools::compute_norm<dim, number>(
            solution_history.get_current_solution().block(d),
            scratch_data,
            normal_dof_indices_per_block[d],
            normal_quad_idx);
        },
        "normal_" + std::to_string(d),
        "normal_vector",
        11 /*precision*/
      );

    Journal::print_line(scratch_data.get_pcout(2),
                        "     * CG: i = " + std::to_string(iter),
                        "normal_vector");

    IterationMonitor<number>::add_linear_iterations(sc, iter);

    // update ghost_values of solution
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::create_wetting_constraints()
  {
    if (not wetting_bc_map.empty())
      {
        AssertThrow(
          normal_vector_data.linear_solver.do_matrix_free,
          ExcMessage(
            "Wetting (Dirichlet) boundary conditions are only available for the matrix-free computation."));

        // Clear out previously stored constraint data
        this->wetting_constraints_indices_and_values.first.clear();
        this->wetting_constraints_indices_and_values.second.clear();

        // Resize outer vector to account for components
        wetting_constraints_indices_and_values.second.resize(dim);

        // Get partitioner for the first component of the normal vector block
        const auto &partitioner =
          this->solution_history.get_current_solution().block(0).get_partitioner();

        // Set up FEFaceValues to evaluate the DoF indices of the normal vector at support points on
        // faces
        FEFaceValues<dim> normal_eval(
          scratch_data.get_mapping(),
          scratch_data.get_fe(normal_dof_indices_per_block[0]),
          // Use support points of the first base element component as quadrature points
          Quadrature<dim - 1>(scratch_data.get_fe(normal_dof_indices_per_block[0])
                                .base_element(0)
                                .get_unit_face_support_points()),
          update_quadrature_points);

        // Loop over all active cells in the associated DoF handler
        for (const auto &cell :
             scratch_data.get_dof_handler(normal_dof_indices_per_block[0]).active_cell_iterators())
          {
            // Only consider locally owned or ghost cells
            if (cell->is_locally_owned() || cell->is_ghost())
              {
                unsigned int face_index = 0;
                // Loop over all faces of the cell
                for (const auto &face : cell->face_iterators())
                  {
                    // Check if the face is at a boundary with a prescribed wetting boundary
                    // condition
                    if (face->at_boundary() and wetting_bc_map.contains(face->boundary_id()))
                      {
                        // Update FEFaceValues
                        normal_eval.reinit(cell, face_index);

                        // Get all DoF indices of the face (all components are included here)
                        std::vector<types::global_dof_index> face_dof_indices(
                          normal_eval.get_fe().dofs_per_face);
                        face->get_dof_indices(face_dof_indices);

                        // Loop over quadrature points on the face
                        for (const auto &q : normal_eval.quadrature_point_indices())
                          {
                            // Get current global DoF index
                            const auto global_index = face_dof_indices[q];

                            // Only consider if the DoF is relevant (owned or ghost)
                            if (partitioner->in_local_range(global_index) or
                                partitioner->is_ghost_entry(global_index))
                              {
                                // Get current local DoF index
                                const auto local_index = partitioner->global_to_local(global_index);

                                // Store local DoF index
                                wetting_constraints_indices_and_values.first.emplace_back(
                                  local_index);

                                // Loop over all spatial components to evaluate store vector-value
                                // boundary condition
                                for (unsigned int d = 0; d < dim; ++d)
                                  {
                                    wetting_constraints_indices_and_values.second[d].emplace_back(
                                      wetting_bc_map[face->boundary_id()]->value(
                                        normal_eval.quadrature_point(q), d));
                                  }
                              }
                          }
                        ++face_index;
                      }
                  }
              }
          }
        // Delete duplicated entries
        for (unsigned int d = 0; d < dim; ++d)
          {
            UtilityFunctions::remove_duplicates(wetting_constraints_indices_and_values.first,
                                                wetting_constraints_indices_and_values.second[d]);
          }

        // Set local DoF indices in operator
        dynamic_cast<NormalVectorOperator<dim, number> *>(normal_vector_operator.get())
          ->set_wetting_bc_indices(wetting_constraints_indices_and_values.first);
      }
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::create_contact_angle_constraints()
  {
    if (not contact_angle_bc_map.empty())
      {
        AssertThrow(
          normal_vector_data.linear_solver.do_matrix_free,
          ExcMessage(
            "Contact angle boundary conditions are only available for the matrix-free computation."));

        if constexpr (dim == 3) // TODO AA 3D generalization
          AssertThrow(false,
                      dealii::ExcMessage(
                        "The contact angle constraints have not been generalized to 3D yet."));

        // Clear out previously stored constraint data
        this->wetting_constraints_indices_and_values.first.clear();
        this->wetting_constraints_indices_and_values.second.clear();

        // Resize outer vector to account for components
        wetting_constraints_indices_and_values.second.resize(dim);

        // Get partitioner for the first component of the normal vector block
        const auto &partitioner =
          this->solution_history.get_current_solution().block(0).get_partitioner();

        // Set up FEFaceValues to evaluate the DoF indices of the normal vector at support points on
        // faces
        FEFaceValues<dim> normal_eval(
          scratch_data.get_mapping(),
          scratch_data.get_fe(normal_dof_indices_per_block[0]),
          // Use support points of the first base element component as quadrature points
          Quadrature<dim - 1>(scratch_data.get_fe(normal_dof_indices_per_block[0])
                                .base_element(0)
                                .get_unit_face_support_points()),
          update_normal_vectors | update_quadrature_points);

        // Loop over all active cells in the associated DoF handler
        for (const auto &cell :
             scratch_data.get_dof_handler(normal_dof_indices_per_block[0]).active_cell_iterators())
          {
            // Only consider locally owned or ghost cells
            if (cell->is_locally_owned() || cell->is_ghost())
              {
                unsigned int face_index = 0;
                // Loop over all faces of the cell
                for (const auto &face : cell->face_iterators())
                  {
                    // Check if the face is at a boundary with a prescribed wetting boundary
                    // condition
                    if (face->at_boundary() and contact_angle_bc_map.contains(face->boundary_id()))
                      {
                        // Update FEFaceValues
                        normal_eval.reinit(cell, face_index);

                        // Get all DoF indices of the face (all components are included here)
                        std::vector<types::global_dof_index> face_dof_indices(
                          normal_eval.get_fe().dofs_per_face);
                        face->get_dof_indices(face_dof_indices);

                        // Retrieve normal vectors (pointing outwards)
                        const auto &wall_normals = normal_eval.get_normal_vectors();

                        // Loop over quadrature points on the face
                        for (const auto &q : normal_eval.quadrature_point_indices())
                          {
                            // Get current global DoF index
                            const auto global_index = face_dof_indices[q];

                            // Only consider if the DoF is relevant (owned or ghost)
                            if (partitioner->in_local_range(global_index) or
                                partitioner->is_ghost_entry(global_index))
                              {
                                // Get current local DoF index
                                const auto local_index = partitioner->global_to_local(global_index);

                                // Store local DoF index
                                contact_angle_constraints_indices_and_values.first.emplace_back(
                                  local_index);

                                // Get wall normal vector (pointing inwards)
                                const auto wall_normal = -wall_normals[q];

                                // Compute wall tangent vector
                                Tensor<1, dim, number>
                                  wall_tangent; /*= (dim == 2) ? Tensor<1, dim,
                                                   number>({-wall_normal[1], wall_normal[0]}) :
                                                   Tensor<1, dim, number>();*/
                                if constexpr (dim == 2)
                                  wall_tangent =
                                    Tensor<1, dim, number>({-wall_normal[1], wall_normal[0]});
                                if constexpr (dim == 3)
                                  wall_tangent = Tensor<1, dim, number>();

                                // Get contact angle (scalar)
                                auto contact_angle_rad =
                                  contact_angle_bc_map[face->boundary_id()]->value(
                                    normal_eval.quadrature_point(q));

                                // TODO AA Compute interface normal vector (in 2D)
                                const auto interface_normal =
                                  std::sin(contact_angle_rad) * wall_normal +
                                  std::cos(contact_angle_rad) * wall_tangent;


                                // Loop over all spatial components to evaluate store vector-value
                                // boundary condition
                                for (unsigned int d = 0; d < dim; ++d)
                                  {
                                    contact_angle_constraints_indices_and_values.second[d]
                                      .emplace_back(interface_normal[d]);
                                  }
                              }
                          }
                        ++face_index;
                      }
                  }
              }
          }
        // Delete duplicated entries
        for (unsigned int d = 0; d < dim; ++d)
          {
            UtilityFunctions::remove_duplicates(
              contact_angle_constraints_indices_and_values.first,
              contact_angle_constraints_indices_and_values.second[d]);
          }

        // Set local DoF indices in operator
        dynamic_cast<NormalVectorOperator<dim, number> *>(normal_vector_operator.get())
          ->set_contact_angle_bc_indices(contact_angle_constraints_indices_and_values.first);
      }
  }

  template <int dim, typename number>
  const typename NormalVectorOperation<dim, number>::BlockVectorType &
  NormalVectorOperation<dim, number>::get_solution_normal_vector() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  typename NormalVectorOperation<dim, number>::BlockVectorType &
  NormalVectorOperation<dim, number>::get_solution_normal_vector()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &vectors)
  {
    solution_history.apply([&](BlockVectorType &v) {
      for (unsigned int d = 0; d < dim; ++d)
        vectors.push_back(&v.block(d));
    });
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::set_wetting_bc_map(
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &p_wetting_bc_map)
  {
    this->wetting_bc_map = p_wetting_bc_map;
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::set_contact_angle_bc_map(
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &p_contact_angle_bc_map)
  {
    this->contact_angle_bc_map = p_contact_angle_bc_map;
  }

  template <int dim, typename number>
  void
  NormalVectorOperation<dim, number>::create_operator()
  {
    normal_vector_operator =
      std::make_unique<NormalVectorOperator<dim, number>>(scratch_data,
                                                          normal_vector_data,
                                                          normal_dof_indices_per_block,
                                                          normal_quad_idx,
                                                          ls_dof_idx,
                                                          &solution_level_set);

    preconditioner =
      make_preconditioner<dim, number, NormalVectorOperator<dim, number>, BlockVectorType>(
        normal_vector_data.linear_solver.preconditioner_type,
        normal_vector_operator.get(),
        normal_vector_data.linear_solver.do_matrix_free);
  }


  template class NormalVectorOperation<1, double>;
  template class NormalVectorOperation<2, double>;
  template class NormalVectorOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
