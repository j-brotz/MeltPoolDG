#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/cut/cut_util.hpp>
#include <meltpooldg/flow/cut_compressible_flow_operation.hpp>
#include <meltpooldg/flow/cut_compressible_flow_operator.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>


namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  CutCompressibleFlowOperation<dim, number>::CutCompressibleFlowOperation(
    const ScratchData<dim>     &scratch_data_in,
    const CompressibleFlowData &comp_flow_data_in,
    const TimeIterator<double> &time_iterator_in,
    const unsigned int          comp_flow_dof_idx_in,
    const unsigned int          level_set_dof_idx_in,
    const unsigned int          comp_flow_quad_idx_in,
    const VectorType           &level_set_in)
    : CompressibleFlowOperation<dim, number>(scratch_data_in,
                                             comp_flow_data_in,
                                             comp_flow_dof_idx_in,
                                             comp_flow_quad_idx_in)
    , time_iterator(time_iterator_in)
    , level_set_dof_idx(level_set_dof_idx_in)
    , level_set(level_set_in)
    , cut_solution_transfer(this->comp_flow_data_.cut.stabilization.ghost_penalty.gamma_M_degree_0,
                            this->comp_flow_data_.cut.stabilization.ghost_penalty.gamma_M_degree_1,
                            this->comp_flow_data_.cut.stabilization.ghost_penalty.gamma_M_degree_2,
                            false /* is_two_phase*/,
                            this->comp_flow_data_.verbosity_level /*verbosity level*/)
    , fe_point_temp(FE_DGQ<dim>(comp_flow_data_in.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
    , mapping_info_surface(this->scratch_data_.get_mapping(),
                           dealii::update_values | dealii::update_gradients |
                             dealii::update_JxW_values | dealii::update_normal_vectors)
  {
    // Currently, only explicit Euler time distretization with ghost-penalty stabilized mass matrix
    // is enabled for cutDG
    this->solution_history_.resize(1);

    // operator class for operator_type = "cut"
    this->comp_flow_operator_ =
      std::make_unique<CutCompressibleFlowOperator<dim, number>>(this->comp_flow_data_,
                                                                 scratch_data_in,
                                                                 this->solution_history_,
                                                                 mapping_info_surface,
                                                                 mapping_info_cells,
                                                                 mapping_info_faces,
                                                                 this->comp_flow_dof_idx,
                                                                 this->comp_flow_quad_idx);

    // old and new mesh classifier for representing the interface topologies in two subsequent time
    // levels
    mesh_classifier = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      this->scratch_data_.get_dof_handler(level_set_dof_idx), level_set);
    mesh_classifier_old = std::make_shared<dealii::NonMatching::MeshClassifier<dim>>(
      this->scratch_data_.get_dof_handler(level_set_dof_idx), level_set);

    // lambda function for vector reinitialization with the matrix-free object
    reinit_vector = [this](VectorType &vec, const DoFHandler<dim> & /*dof_handler*/) {
      this->scratch_data_.get_matrix_free().initialize_dof_vector(vec);
    };

    // mapping info objects for non-matching quadrature rules
    mapping_info_cells.push_back(
      std::make_shared<dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<double>>>(
        this->scratch_data_.get_mapping(),
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
          dealii::update_normal_vectors));

    UpdateFlags update_flags_faces =
      update_values | update_gradients | update_JxW_values | update_normal_vectors;
    if (this->comp_flow_data_.fe.degree == 2)
      update_flags_faces = update_flags_faces | update_hessians;
    mapping_info_faces.push_back(
      std::make_shared<NonMatching::MappingInfo<dim, dim, VectorizedArray<number>>>(
        this->scratch_data_.get_mapping(), update_flags_faces));
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::reinit()
  {
    // check if fe type is equal to FE_DGQ<dim>
    AssertThrow(
      dynamic_cast<const FE_DGQ<dim> *>(
        &(this->scratch_data_.get_fe(this->comp_flow_dof_idx).base_element(0))) != nullptr,
      dealii::ExcMessage(
        "The cutDG compressible flow solver only supports finite element types of FE_DGQ!"));

    this->scratch_data_.initialize_dof_vector(this->solution_history_.get_current_solution(),
                                              this->comp_flow_dof_idx);
    this->scratch_data_.initialize_dof_vector(rhs, this->comp_flow_dof_idx);
    this->comp_flow_operator_->reinit();
    compute_intersected_quadrature();
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::distribute_dofs(
    dealii::DoFHandler<dim> &dof_handler) const
  {
    classify_cells();

    FESystem<dim>   fe(FE_DGQ<dim>(this->comp_flow_data_.fe.degree), dim + 2);
    FE_Nothing<dim> fe_nothing;
    FESystem<dim>   fe_n(fe_nothing, dim + 2);

    dealii::hp::FECollection<dim> fe_collection;

    fe_collection.push_back(fe);   // liquid
    fe_collection.push_back(fe);   // intersected
    fe_collection.push_back(fe_n); // gas (not relevant for single-phase case)

    CutUtil::set_fe_index<dim>(dof_handler, *mesh_classifier, false /* set_future */);
    dof_handler.distribute_dofs(fe_collection);
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::solve(double current_time, double time_step)
  {
    // - setup new dof layout, reinit vectors
    // - compute new quadrature rules for intersected elements, intersected faces and phase surface
    // - reinit matrix-free object, rhs and solution vectors
    adapt_to_new_interface_position();

    // update inverse time step size
    this->comp_flow_operator_->compute_inverse_time_step(time_step);

    // compute rhs
    this->comp_flow_operator_->create_rhs(current_time,
                                          time_step,
                                          rhs,
                                          this->solution_history_.get_current_solution());

    // solve linear and symmetric system of equations with CG
    linear_solver.solve<VectorType>(*this->comp_flow_operator_,
                                    this->solution_history_.get_current_solution(),
                                    rhs,
                                    this->comp_flow_data_.time_integrator.linear_solver_data);
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::set_initial_condition(const Function<dim> &function)
  {
    FECellIntegrator<dim, dim + 2, number> phi(this->scratch_data_.get_matrix_free(),
                                               this->comp_flow_dof_idx,
                                               this->comp_flow_quad_idx);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, number> inverse(phi);
    this->solution_history_.get_current_solution().zero_out_ghost_values();

    for (unsigned int cell = 0; cell < this->scratch_data_.get_matrix_free().n_cell_batches();
         ++cell)
      {
        const auto active_fe_index = this->scratch_data_.get_matrix_free().get_cell_category(cell);
        if (active_fe_index == CutUtil::CellCategory::liquid or
            (active_fe_index == CutUtil::CellCategory::intersected))
          {
            phi.reinit(cell);
            for (const unsigned int q : phi.quadrature_point_indices())
              phi.submit_dof_value(
                VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
                  function, phi.quadrature_point(q)),
                q);
            inverse.transform_from_q_points_to_basis(dim + 2,
                                                     phi.begin_dof_values(),
                                                     phi.begin_dof_values());
            phi.set_dof_values(this->solution_history_.get_current_solution());
          }
      }
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::set_inflow_field_unfitted_boundary(
    std::shared_ptr<Function<dim>> &inflow_function)
  {
    this->comp_flow_operator_->set_inflow_field_unfitted_boundary(inflow_function);
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::set_unfitted_object_velocity(
    std::shared_ptr<Function<dim>> &velocity_function)
  {
    this->comp_flow_operator_->set_unfitted_object_velocity(velocity_function);
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::classify_cells() const
  {
    mesh_classifier->reclassify();
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::adapt_to_new_interface_position()
  {
    std::swap(mesh_classifier_old, mesh_classifier);
    classify_cells();

    Assert(reinit_matrix_free != nullptr,
           dealii::ExcMessage("You must register the reinit_matrix_free lambda function first!"));

    // transfer old solution according to the new interface position,
    // the matrix-free object is reinitialized within the reinit function
    cut_solution_transfer.reinit(const_cast<dealii::DoFHandler<dim> &>(
                                   this->scratch_data_.get_dof_handler(this->comp_flow_dof_idx)),
                                 const_cast<dealii::Triangulation<dim> &>(
                                   this->scratch_data_.get_triangulation()),
                                 this->solution_history_.get_current_solution(),
                                 *mesh_classifier_old,
                                 *mesh_classifier,
                                 reinit_vector,
                                 reinit_matrix_free);

    // reinit solution vector and rhs vector
    this->scratch_data_.initialize_dof_vector(this->solution_history_.get_current_solution(),
                                              this->comp_flow_dof_idx);
    this->scratch_data_.initialize_dof_vector(rhs, this->comp_flow_dof_idx);

    // reinit cut operator
    this->comp_flow_operator_->reinit();

    // compute non-matching quadrature rules
    compute_intersected_quadrature();

    // get extrapolated solution
    this->solution_history_.get_current_solution().swap(
      cut_solution_transfer.get_updated_solution());
  }

  template <int dim, typename number>
  number
  CutCompressibleFlowOperation<dim, number>::compute_minimum_density() const
  {
    FECellIntegrator<dim, 1, number> phi(this->scratch_data_.get_matrix_free(),
                                         this->comp_flow_dof_idx,
                                         this->comp_flow_quad_idx);
    this->solution_history_.get_current_solution().update_ghost_values();

    constexpr unsigned int n_lanes = VectorizedArray<number>::size();

    double min_density = std::numeric_limits<double>::max();

    for (unsigned int cell = 0; cell < this->scratch_data_.get_matrix_free().n_cell_batches();
         ++cell)
      {
        const auto active_fe_index = this->scratch_data_.get_matrix_free().get_cell_category(cell);

        if (active_fe_index == CutUtil::CellCategory::liquid)
          {
            phi.reinit(cell);
            phi.gather_evaluate(this->solution_history_.get_current_solution(),
                                EvaluationFlags::values);
            for (const unsigned int q : phi.quadrature_point_indices())
              {
                const auto density = phi.get_value(q);
                for (unsigned int lane = 0;
                     lane <
                     this->scratch_data_.get_matrix_free().n_active_entries_per_cell_batch(cell);
                     ++lane)
                  min_density = std::min(density[lane], min_density);
              }
          }
        else if (active_fe_index == CutUtil::CellCategory::intersected)
          {
            FEPointEvaluation<1, dim, dim, VectorizedArray<number>> fe_point_eval(
              *mapping_info_cells[0], fe_point_temp);

            phi.reinit(cell);
            phi.read_dof_values(this->solution_history_.get_current_solution());

            for (unsigned int lane = 0;
                 lane < this->scratch_data_.get_matrix_free().n_active_entries_per_cell_batch(cell);
                 ++lane)
              {
                fe_point_eval.reinit(cell * n_lanes + lane);
                fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                         &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       EvaluationFlags::values);

                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  {
                    const auto density = fe_point_eval.get_value(q);

                    for (unsigned int v = 0;
                         v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                         ++v)
                      min_density = std::min(density[v], min_density);
                  }
              }
          }
      }

    min_density = Utilities::MPI::min(min_density,
                                      this->scratch_data_.get_matrix_free()
                                        .get_dof_handler(this->comp_flow_dof_idx)
                                        .get_communicator());

    return min_density;
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::compute_intersected_quadrature()
  {
    level_set.update_ghost_values();

    CutUtil::compute_intersected_quadrature(mapping_info_cells,
                                            mapping_info_surface,
                                            this->scratch_data_.get_dof_handler(level_set_dof_idx),
                                            level_set,
                                            this->scratch_data_.get_matrix_free(),
                                            this->comp_flow_data_.fe.degree,
                                            this->comp_flow_data_.cut.two_phase,
                                            true /*is_dg*/,
                                            mapping_info_faces);
  }

  template <int dim, typename number>
  number
  CutCompressibleFlowOperation<dim, number>::compute_convective_time_step_limit() const
  {
    number                                 max_transport              = 0;
    number                                 convective_time_step_limit = 0.;
    FECellIntegrator<dim, dim + 2, number> phi(this->scratch_data_.get_matrix_free(),
                                               this->comp_flow_dof_idx,
                                               this->comp_flow_quad_idx);

    for (unsigned int cell = 0; cell < this->scratch_data_.get_matrix_free().n_cell_batches();
         ++cell)
      {
        const auto active_fe_index = this->scratch_data_.get_matrix_free().get_cell_category(cell);

        if (active_fe_index == CutUtil::CellCategory::liquid)
          {
            phi.reinit(cell);
            phi.gather_evaluate(this->solution_history_.get_current_solution(),
                                EvaluationFlags::values);
            VectorizedArray<number> local_max = 0.;
            for (const unsigned int q : phi.quadrature_point_indices())
              {
                const auto conserved_variables = phi.get_value(q);
                const auto velocity =
                  CompressibleFlowCalculators<dim, number>::calculate_velocity(conserved_variables);
                const auto pressure = CompressibleFlowCalculators<dim, number>::calculate_pressure(
                  conserved_variables, this->comp_flow_data_.gamma);

                const auto              inverse_jacobian = phi.inverse_jacobian(q);
                const auto              convective_speed = inverse_jacobian * velocity;
                VectorizedArray<number> convective_limit = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                  convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

                const auto speed_of_sound =
                  std::sqrt(this->comp_flow_data_.gamma * pressure / conserved_variables[0]);

                Tensor<1, dim, VectorizedArray<number>> eigenvector;
                for (unsigned int d = 0; d < dim; ++d)
                  eigenvector[d] = 1.;
                for (unsigned int i = 0; i < 5 /* number of iterations */; ++i)
                  {
                    eigenvector = transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
                    VectorizedArray<number> eigenvector_norm = 0.;
                    for (unsigned int d = 0; d < dim; ++d)
                      eigenvector_norm = std::max(eigenvector_norm, std::abs(eigenvector[d]));
                    eigenvector /= eigenvector_norm;
                  }
                const auto jac_times_ev = inverse_jacobian * eigenvector;
                const auto max_eigenvalue =
                  std::sqrt((jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
                local_max = std::max(local_max, max_eigenvalue * speed_of_sound + convective_limit);
              }

            for (unsigned int v = 0;
                 v < this->scratch_data_.get_matrix_free().n_active_entries_per_cell_batch(cell);
                 ++v)
              max_transport = std::max(max_transport, local_max[v]);
          }
        else if (active_fe_index == CutUtil::CellCategory::intersected)
          {
            // TODO: Rethink time-step size computation for cut elements. Do we have to consider the
            // inverse Jacobian of the physically relevant part of the cut element or the one of the
            // base element?
            FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval(
              *mapping_info_cells[0], fe_point_temp);

            VectorizedArray<number> local_max = 0.;
            constexpr unsigned int  n_lanes   = VectorizedArray<number>::size();

            phi.reinit(cell);
            phi.read_dof_values(this->solution_history_.get_current_solution());

            for (unsigned int lane = 0;
                 lane < this->scratch_data_.get_matrix_free().n_active_entries_per_cell_batch(cell);
                 ++lane)
              {
                fe_point_eval.reinit(cell * n_lanes + lane);
                fe_point_eval.evaluate(StridedArrayView<const number, n_lanes>(
                                         &phi.begin_dof_values()[0][lane], n_dofs_per_cell),
                                       EvaluationFlags::values);

                for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                  {
                    const auto conserved_variables = fe_point_eval.get_value(q);
                    const auto velocity =
                      CompressibleFlowCalculators<dim, number>::calculate_velocity(
                        conserved_variables);
                    const auto pressure =
                      CompressibleFlowCalculators<dim, number>::calculate_pressure(
                        conserved_variables, this->comp_flow_data_.gamma);

                    const Tensor<2, dim, VectorizedArray<number>> inverse_jacobian =
                      fe_point_eval.inverse_jacobian(q);
                    const auto              convective_speed = inverse_jacobian * velocity;
                    VectorizedArray<number> convective_limit = 0.;
                    for (unsigned int d = 0; d < dim; ++d)
                      convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

                    const auto speed_of_sound =
                      std::sqrt(this->comp_flow_data_.gamma * pressure / conserved_variables[0]);

                    Tensor<1, dim, VectorizedArray<number>> eigenvector;
                    for (unsigned int d = 0; d < dim; ++d)
                      eigenvector[d] = 1.;
                    for (unsigned int i = 0; i < 5 /* number of iterations */; ++i)
                      {
                        eigenvector =
                          transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
                        VectorizedArray<number> eigenvector_norm = 0.;
                        for (unsigned int d = 0; d < dim; ++d)
                          eigenvector_norm = std::max(eigenvector_norm, std::abs(eigenvector[d]));
                        eigenvector /= eigenvector_norm;
                      }
                    const auto jac_times_ev = inverse_jacobian * eigenvector;
                    const auto max_eigenvalue =
                      std::sqrt((jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
                    local_max =
                      std::max(local_max, max_eigenvalue * speed_of_sound + convective_limit);

                    for (unsigned int v = 0;
                         v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                         ++v)
                      max_transport = std::max(max_transport, local_max[v]);
                  }
              }
          }
      }

    max_transport = Utilities::MPI::max(max_transport, this->scratch_data_.get_mpi_comm());

    convective_time_step_limit =
      this->comp_flow_data_.courant_number /
      std::pow(this->scratch_data_.get_degree(this->comp_flow_dof_idx), 1.5) / max_transport;

    return convective_time_step_limit;
  }

  template <int dim, typename number>
  void
  CutCompressibleFlowOperation<dim, number>::register_reinit_matrix_free(
    const std::function<void(const dealii::DoFHandler<dim> &)> reinit_matrix_free_in)
  {
    reinit_matrix_free = reinit_matrix_free_in;
  }

  template class CutCompressibleFlowOperation<1, double>;
  template class CutCompressibleFlowOperation<2, double>;
  template class CutCompressibleFlowOperation<3, double>;
} // namespace MeltPoolDG::Flow
