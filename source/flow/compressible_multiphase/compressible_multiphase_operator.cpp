
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/cut/util.hpp>
#include <meltpooldg/flow/compressible_flow_explicit_utils.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_multiphase/compressible_multiphase_operator.hpp>
#include <meltpooldg/utilities/numbers.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>
#include <meltpooldg/flow/compressible_multiphase/compressible_multiphase_interface_kernels.hpp>

#include <boost/math/constants/constants.hpp>

namespace MeltPoolDG::Multiphase
{
  template <unsigned int dim, typename number>
  CompressibleMultiphaseOperator<dim, number>::CompressibleMultiphaseOperator(
    MeltPoolDG::Flow::CompressibleFlowScratchData<dim, number> &flow_scratch_data,
    const MappingInfoType                                      &mapping_info_surface_in,
    const MappingInfoVectorType                                &mapping_info_cells_in,
    const MappingInfoVectorType                                &mapping_info_faces_in)
    : flow_scratch_data(flow_scratch_data)
    , convective_terms(flow_scratch_data.flow_data)
    , viscous_terms(flow_scratch_data.flow_data)
    , mapping_info_surface(mapping_info_surface_in)
    , mapping_info_cells(mapping_info_cells_in)
    , mapping_info_faces(mapping_info_faces_in)
    , fe_point_temp(FE_DGQ<dim>(flow_scratch_data.flow_data.fe.degree), dim + 2)
    , n_dofs_per_cell(fe_point_temp.dofs_per_cell)
  {
    const double q_1 = 2.*flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity+
              flow_scratch_data.flow_data.material_data_gas_phase.thermal_conductivity/flow_scratch_data.flow_data.material_data_gas_phase.specific_gas_constant*
              (flow_scratch_data.flow_data.material_data_gas_phase.gamma-1.);

    const double q_2 = 2.*flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity+
              flow_scratch_data.flow_data.material_data_liquid_phase.thermal_conductivity/flow_scratch_data.flow_data.material_data_liquid_phase.specific_gas_constant*
              (flow_scratch_data.flow_data.material_data_liquid_phase.gamma-1.);

    alpha_1 = q_1/(q_1+q_2);
    alpha_2 = 1.-alpha_1;
  }

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::compute_inverse_time_step(const number &time_step)
  {
    Assert(time_step > 0., dealii::ExcMessage("Time step size must be larger than 0!"));
    inv_time_step = 1. / time_step;
  }

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::vmult(VectorType &dst, const VectorType &src) const
  {
    typedef std::function<void(const dealii::MatrixFree<dim, number> &,
                               dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                               const dealii::LinearAlgebra::distributed::Vector<number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    local_applier_type cell = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_cell_lhs);
    local_applier_type face = MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_face_lhs);
    local_applier_type boundary_face =
      MELT_POOL_DG_LAMBDA_WRAPPER(this->local_apply_boundary_face_lhs);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);
  }

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::create_rhs(const number     &time,
                                                          const number     &time_step,
                                                          VectorType       &dst,
                                                          const VectorType &src) const
  {
    AssertThrow(!dealii::numbers::is_invalid(inv_time_step),
                dealii::ExcMessage(
                  "Inverse time step size must be set to compute the rhs vector."));

    typedef std::function<void(const MatrixFree<dim, number> &,
                               LinearAlgebra::distributed::Vector<number>       &dst,
                               const LinearAlgebra::distributed::Vector<number> &src,
                               const std::pair<unsigned int, unsigned int> &)>
      local_applier_type;

    flow_scratch_data.boundary_conditions.update_boundary_conditions(time);
    local_applier_type cell          = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_cell_rhs);
    local_applier_type face          = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_face_rhs);
    local_applier_type boundary_face = MELT_POOL_DG_LAMBDA_WRAPPER(local_apply_boundary_face_rhs);
    flow_scratch_data.scratch_data.get_matrix_free().loop(
      cell,
      face,
      boundary_face,
      dst,
      src,
      true,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, number>::DataAccessOnFaces::gradients);

    dst *= time_step;
  }

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::local_apply_cell_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto active_fe_index =
     flow_scratch_data.scratch_data.get_matrix_free().get_cell_range_category(cell_range);

    FECellIntegrator<dim, dim + 2, number> phi_liquid =
      create_cell_integrator(CutUtil::CellCategory::liquid, 0);
    FECellIntegrator<dim, dim + 2, number> phi_gas =
      create_cell_integrator(CutUtil::CellCategory::gas, dim + 2);
    FECellIntegrator<dim, dim + 2, number> phi_liquid_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, 0);
    FECellIntegrator<dim, dim + 2, number> phi_gas_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, dim + 2);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim>                 *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(flow_scratch_data.body_force.get());

    if (constant_function)
      constant_body_force = VectorTools::evaluate_function_at_vectorized_points(
        *constant_function, dealii::Point<dim, dealii::VectorizedArray<number>>());

    // lambda function for cell integral
    auto process_cell = [&]<bool is_gas_phase, typename T0>(T0 &phi) {
      for (const unsigned int q : phi.quadrature_point_indices())
        {
          auto [flux, grad_flux] = MeltPoolDG::Flow::
            rhs_cell_integral_kernel<dim, number, T0, is_gas_phase>(
              phi,
              q,
              constant_function ? &constant_body_force : nullptr,
              convective_terms,
              viscous_terms,
              flow_scratch_data);

          // consider mass term
          if (flow_scratch_data.body_force.get() != nullptr)
            phi.submit_value(flux + phi.get_value(q) * inv_time_step, q);
          else
            phi.submit_value(phi.get_value(q) * inv_time_step, q);
          phi.submit_gradient(grad_flux, q);
        }
    };

    switch (active_fe_index) {
        case CutUtil::CellCategory::liquid: {
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              phi_liquid.reinit(cell);
              phi_liquid.gather_evaluate(src,
                                  EvaluationFlags::values |
                                    ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                       EvaluationFlags::gradients :
                                       EvaluationFlags::nothing));
              process_cell.template operator()<false>(phi_liquid);
              phi_liquid.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
            }
          break;
        }
        case CutUtil::CellCategory::gas: {
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              phi_gas.reinit(cell);
              phi_gas.gather_evaluate(src,
                                  EvaluationFlags::values |
                                    ((flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
                                       EvaluationFlags::gradients :
                                       EvaluationFlags::nothing));
              process_cell.template operator()<true>(phi_gas);
              phi_gas.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
            }
          break;
        }
        case CutUtil::CellCategory::intersected: {
          constexpr unsigned int n_lanes = VectorizedArray<number>::size();

        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> phi_point_liquid(
          *mapping_info_cells[0], fe_point_temp);
        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
          phi_point_surface_liquid(mapping_info_surface, fe_point_temp);
        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> phi_point_gas(
          *mapping_info_cells[1], fe_point_temp);
        dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
          phi_point_surface_gas(mapping_info_surface, fe_point_temp);

        for (unsigned int cell_batch = cell_range.first; cell_batch < cell_range.second;
             ++cell_batch)
          {
            phi_liquid_intersected.reinit(cell_batch);
            phi_liquid_intersected.read_dof_values(src);

            phi_gas_intersected.reinit(cell_batch);
            phi_gas_intersected.read_dof_values(src);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                   cell_batch);
                 ++lane)
              {
                // evaluate for domain integral in liquid phase
                phi_point_liquid.reinit(cell_batch * n_lanes + lane);
                phi_point_liquid.evaluate(
                  StridedArrayView<const number, n_lanes>(&phi_liquid_intersected.begin_dof_values()[0][lane],
                                                          n_dofs_per_cell),
                  EvaluationFlags::values | ((flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
                                               EvaluationFlags::gradients :
                                               EvaluationFlags::nothing));

                // evaluate for surface integral in liquid phase
                phi_point_surface_liquid.reinit(cell_batch * n_lanes + lane);
                phi_point_surface_liquid.evaluate(StridedArrayView<const number, n_lanes>(
                                            &phi_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                          EvaluationFlags::values | EvaluationFlags::gradients);

                // evaluate for domain integral in gas phase
                phi_point_gas.reinit(cell_batch * n_lanes + lane);
                phi_point_gas.evaluate(
                  StridedArrayView<const number, n_lanes>(&phi_gas_intersected.begin_dof_values()[0][lane],
                                                          n_dofs_per_cell),
                  EvaluationFlags::values | ((flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
                                               EvaluationFlags::gradients :
                                               EvaluationFlags::nothing));

                // evaluate for surface integral in gas phase
                phi_point_surface_gas.reinit(cell_batch * n_lanes + lane);
                phi_point_surface_gas.evaluate(StridedArrayView<const number, n_lanes>(
                                            &phi_gas_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                          EvaluationFlags::values | EvaluationFlags::gradients);

                // do domain integral in liquid phase
                process_cell.template operator()<false>(phi_point_liquid);

                phi_point_liquid.integrate(StridedArrayView<number, n_lanes>(
                                          &phi_liquid_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                        EvaluationFlags::values | EvaluationFlags::gradients);

                // do domain integral in gas phase
                process_cell.template operator()<true>(phi_point_gas);

                phi_point_gas.integrate(StridedArrayView<number, n_lanes>(
                                          &phi_gas_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                        EvaluationFlags::values | EvaluationFlags::gradients);

                // do surface integral TODO
                switch (flow_scratch_data.flow_data.interface_numerical_method){
                  case Flow::InterfaceNumericalMethod::penalty: {
                    for (const unsigned int q : phi_point_surface_liquid.quadrature_point_indices())
                      {
                        auto w_liquid      = phi_point_surface_liquid.get_value(q);
                        auto w_gas      = phi_point_surface_gas.get_value(q);
                        auto grad_w_liquid = phi_point_surface_liquid.get_gradient(q);
                        auto grad_w_gas = phi_point_surface_gas.get_gradient(q);

                        auto flux_data = calculate_interface_flux_penalty<dim,number,ConservedVariablesType,ConservedVariablesGradType>(w_liquid,w_gas,grad_w_liquid,grad_w_gas, flow_scratch_data.flow_data, viscous_terms);

                        auto flux_m = std::get<0>(flux_data);
                        auto flux_p = std::get<1>(flux_data);
                        //normal_velocity_interface = w_m[1][0] / w_m[0][0]; // TODO

                        phi_point_surface_liquid.submit_value(-flux_m, q);
                        phi_point_surface_gas.submit_value(-flux_p, q);
                      }
                    break;
                  }
                  case Flow::InterfaceNumericalMethod::HLLC0_and_Nitsche: {
                    for (const unsigned int q : phi_point_surface_gas.quadrature_point_indices())
                      {
                        auto w_liquid      = phi_point_surface_liquid.get_value(q);
                        auto w_gas      = phi_point_surface_gas.get_value(q);
                        auto grad_w_liquid = phi_point_surface_liquid.get_gradient(q);
                        auto grad_w_gas = phi_point_surface_gas.get_gradient(q);
                        auto normal   = phi_point_surface_liquid.normal_vector(q);

                        auto riemann_flux_data =
                          calculate_convective_interface_flux_HLLC<dim,number,ConservedVariablesType,ConservedVariablesGradType>(
                            w_liquid,
                            w_gas,
                            normal,
                            flow_scratch_data.flow_data.material_data_gas_phase.gamma,
                            flow_scratch_data.flow_data.material_data_liquid_phase.gamma,
                            flow_scratch_data.flow_data.m_dot_evap,
                            convective_terms,
                            flow_scratch_data.flow_data);

                        auto flux_m = std::get<0>(riemann_flux_data);
                        auto flux_p = -std::get<1>(riemann_flux_data); // opposite normal direction for phase 2
                        velocity_interface = std::get<2>(riemann_flux_data)[0];

                        const auto penalty_parameter =
                        (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
                          phi_liquid_intersected.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                          0.;

                        if (flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0)
                          {
                            auto viscous_interface_flux_data = calculate_viscous_interface_flux<dim,number,ConservedVariablesType,ConservedVariablesGradType>(
                              w_liquid,
                              w_gas,
                              grad_w_liquid,
                              grad_w_gas,
                              phi_point_surface_liquid.normal_vector(q),
                              penalty_parameter,
                              flow_scratch_data.flow_data.material_data_gas_phase.gamma,
                              flow_scratch_data.flow_data.material_data_liquid_phase.gamma,
                              flow_scratch_data.flow_data.material_data_gas_phase.specific_gas_constant,
                              flow_scratch_data.flow_data.material_data_liquid_phase.specific_gas_constant,
                              alpha_1,
                              alpha_2,
                              flow_scratch_data.flow_data.m_dot_evap,
                              flow_scratch_data.flow_data.symm_int_penalty_parameter_interface,
                              0. /*p_inf phase 1*/,
                              0. /*p_inf phase 2*/,
                              viscous_terms,
                              flow_scratch_data.flow_data);

                              flux_m -= viscous_interface_flux_data.first;
                              flux_p += viscous_interface_flux_data.second; // opposite normal direction for phase 2
                            }

                        phi_point_surface_liquid.submit_value(-flux_m, q);
                        phi_point_surface_gas.submit_value(-flux_p, q);

                        if (flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0)
                          {
                            auto numerical_flux_gradient =
                              calculate_viscous_interface_flux_gradient<dim,number,ConservedVariablesType,ConservedVariablesGradType>(
                                w_liquid,
                                w_gas,
                                normal,
                                flow_scratch_data.flow_data.material_data_gas_phase.gamma,
                                flow_scratch_data.flow_data.material_data_liquid_phase.gamma,
                                flow_scratch_data.flow_data.material_data_gas_phase.specific_gas_constant,
                                flow_scratch_data.flow_data.material_data_liquid_phase.specific_gas_constant,
                                alpha_1,
                          	    alpha_2,
                          	    flow_scratch_data.flow_data.m_dot_evap,
                          	    0. /*p_inf phase 1*/,
                                0. /*p_inf phase 2*/,
                                viscous_terms,
                                flow_scratch_data.flow_data);

                            phi_point_surface_liquid.submit_gradient(-numerical_flux_gradient.first, q);
                            phi_point_surface_gas.submit_gradient(-numerical_flux_gradient.second, q);
                          }
                      }
                    break;
                  }
                  default:
                    Assert(false, dealii::ExcNotImplemented());
              }

                phi_point_surface_liquid.integrate(StridedArrayView<number, n_lanes>(
                                  &phi_liquid_intersected.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values /*TODO for HLLC0*/, true
                                  /*specify flag 'true' for summing the integrated values into the solution values*/);

                phi_point_surface_gas.integrate(StridedArrayView<number, n_lanes>(
                                  &phi_gas_intersected.begin_dof_values()[0][lane],
                                  n_dofs_per_cell),
                                  EvaluationFlags::values /*TODO for HLLC0*/, true
                                  /*specify flag 'true' for summing the integrated values into the solution values*/);
              }
            phi_liquid_intersected.distribute_local_to_global(dst);
            phi_gas_intersected.distribute_local_to_global(dst);
          }
          break;
        }
        default:
          break;
    }
  }

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::local_apply_face_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_m =
            create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_p =
            create_face_integrator(false, CutUtil::CellCategory::liquid, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_m =
            create_face_integrator(true, CutUtil::CellCategory::gas, dim+2);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_p =
            create_face_integrator(false, CutUtil::CellCategory::gas, dim+2);
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_m_intersected =
            create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_p_intersected =
            create_face_integrator(false, CutUtil::CellCategory::intersected, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_m_intersected =
            create_face_integrator(true, CutUtil::CellCategory::intersected, dim+2);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_p_intersected =
            create_face_integrator(false, CutUtil::CellCategory::intersected, dim+2);

    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    auto process_face = [&]<bool is_gas_phase>(auto &phi_m, auto &phi_p) {
        for (unsigned int face = face_range.first; face < face_range.second; ++face) {
            phi_m.reinit(face);
            phi_p.reinit(face);

            auto eval_flags = EvaluationFlags::values |
                              (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0 ?
                               EvaluationFlags::gradients : EvaluationFlags::nothing);

            phi_m.gather_evaluate(src, eval_flags);
            phi_p.gather_evaluate(src, eval_flags);

            const auto penalty_parameter = (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
                0.5 * std::max(phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                               phi_p.read_cell_data(flow_scratch_data.interior_penalty_parameter)) : 0.;

            for (const unsigned int q : phi_m.quadrature_point_indices()) {
                auto [flux_m, flux_p, grad_flux_m, grad_flux_p] = MeltPoolDG::Flow::
                    rhs_face_integral_kernel<dim, number, FEFaceIntegrator<dim, dim + 2, number>, is_gas_phase>(
                        phi_m, phi_p, q, penalty_parameter,
                        convective_terms, viscous_terms,
                        flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity);

                phi_m.submit_value(flux_m, q);
                phi_p.submit_value(flux_p, q);
                if (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) {
                    phi_m.submit_gradient(grad_flux_m, q);
                    phi_p.submit_gradient(grad_flux_p, q);
                }
            }
            phi_m.integrate_scatter(eval_flags, dst);
            phi_p.integrate_scatter(eval_flags, dst);
        }
    };

    auto process_intersected_face = [&]<bool is_gas_phase>(auto &phi_m_int, auto &phi_p_int, const unsigned int mapping_idx) {
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> phi_point_m(
          *mapping_info_faces[mapping_idx], fe_point_temp);
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> phi_point_p(
        *mapping_info_faces[mapping_idx], fe_point_temp);

      for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_m_int.reinit(face);
            phi_m_int.read_dof_values(src);
            phi_p_int.reinit(face);
            phi_p_int.read_dof_values(src);

            phi_m_int.project_to_face(EvaluationFlags::values |
                                  ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing));
            phi_p_int.project_to_face(EvaluationFlags::values |
                                  ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing));

            const auto face_info =
              flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                   face);
                 ++lane)
              {
                phi_point_m.reinit(face_info.cells_interior[lane],
                                           static_cast<int>(face_info.interior_face_no));
                phi_point_p.reinit(face_info.cells_exterior[lane],
                                          static_cast<int>(face_info.exterior_face_no));

                phi_point_m.evaluate_in_face(
                  &phi_m_int.get_scratch_data().begin()[0][lane],
                  EvaluationFlags::values | ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                               EvaluationFlags::gradients :
                                               EvaluationFlags::nothing));

                phi_point_p.evaluate_in_face(
                  &phi_p_int.get_scratch_data().begin()[0][lane],
                  EvaluationFlags::values | ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                               EvaluationFlags::gradients :
                                               EvaluationFlags::nothing));

                // factor 0.5 for interior face
                const dealii::VectorizedArray<number> penalty_parameter =
                  (flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                    0.5 *
                      std::max(phi_m_int.read_cell_data(flow_scratch_data.interior_penalty_parameter),
                               phi_p_int.read_cell_data(flow_scratch_data.interior_penalty_parameter)) :
                    0.;

                for (const unsigned int q : phi_point_m.quadrature_point_indices())
                  {
                    auto [flux_m, flux_p, grad_flux_m, grad_flux_p] = MeltPoolDG::Flow::rhs_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>,
                      is_gas_phase>(
                      phi_point_m,
                      phi_point_p,
                      q,
                      penalty_parameter,
                      convective_terms,
                      viscous_terms,
                      flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity);

                    phi_point_m.submit_value(flux_m, q);
                    phi_point_p.submit_value(flux_p, q);
                    if (flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0)
                      {
                        phi_point_m.submit_gradient(grad_flux_m, q);
                        phi_point_p.submit_gradient(grad_flux_p, q);
                      }
                  }

                phi_point_m.integrate_in_face(
                  &phi_m_int.get_scratch_data().begin()[0][lane],
                  EvaluationFlags::values | ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                               EvaluationFlags::gradients :
                                               EvaluationFlags::nothing));

                phi_point_p.integrate_in_face(
                  &phi_p_int.get_scratch_data().begin()[0][lane],
                  EvaluationFlags::values | ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                               EvaluationFlags::gradients :
                                               EvaluationFlags::nothing));
              }

            phi_m_int.collect_from_face(EvaluationFlags::values |
                                      ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                         EvaluationFlags::gradients :
                                         EvaluationFlags::nothing),
                                    phi_m_int.begin_dof_values());

            phi_p_int.collect_from_face(EvaluationFlags::values |
                                      ((flow_scratch_data.flow_data.material_data_gas_phase.dynamic_viscosity > 0) ?
                                         EvaluationFlags::gradients :
                                         EvaluationFlags::nothing),
                                    phi_p_int.begin_dof_values());

            phi_m_int.distribute_local_to_global(dst);
            phi_p_int.distribute_local_to_global(dst);
          }
    };

    switch (face_type) {
        case CutUtil::FaceType::inside_face_liquid:
          process_face.template operator()<false>(phi_liquid_m, phi_liquid_p);
          break;
        case CutUtil::FaceType::mixed_face_liquid:
          process_face.template operator()<false>(phi_liquid_m, phi_liquid_p_intersected);
          break;
        case CutUtil::FaceType::inside_face_gas:
          process_face.template operator()<true>(phi_gas_m, phi_gas_p);
          break;
        case CutUtil::FaceType::mixed_face_gas:
          process_face.template operator()<true>(phi_gas_m, phi_gas_p_intersected);
          break;
        case CutUtil::FaceType::intersected_face:
          process_intersected_face.template operator()<false>(phi_liquid_m_intersected, phi_liquid_p_intersected, 0);
          process_intersected_face.template operator()<true>(phi_gas_m_intersected, phi_gas_p_intersected, 1);
        default:
          break;
    }
  }

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::local_apply_boundary_face_rhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_m =
            create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_m =
            create_face_integrator(true, CutUtil::CellCategory::gas, dim+2);
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_m_intersected =
            create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_m_intersected =
            create_face_integrator(true, CutUtil::CellCategory::intersected, dim+2);

    const auto face_category =
      flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);

    auto process_face = [&]<bool is_gas_phase>(auto &phi_m) {
      for (unsigned int face = face_range.first; face < face_range.second; ++face)
        {
          phi_m.reinit(face);
          phi_m.gather_evaluate(src,
                              dealii::EvaluationFlags::values |
                                dealii::EvaluationFlags::gradients);

          const dealii::VectorizedArray<number> penalty_parameter =
            (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
              phi_m.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
              0.;

          for (const unsigned int q : phi_m.quadrature_point_indices())
            {
              auto [flux_m, grad_flux_m] =
                MeltPoolDG::Flow::rhs_boundary_face_integral_kernel<dim,
                                                  number,
                                                  FEFaceIntegrator<dim, dim + 2, number>,
                                                  is_gas_phase>(
                  phi_m,
                  q,
                  flow_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                  penalty_parameter,
                  convective_terms,
                  viscous_terms,
                  flow_scratch_data);
              phi_m.submit_value(flux_m, q);
              if (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0)
                phi_m.submit_gradient(grad_flux_m, q);
            }

          phi_m.integrate_scatter(EvaluationFlags::values |
                                  ((flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
        }
    };

    auto process_intersected_faces = [&]<bool is_gas_phase>(auto &phi_m_int, const unsigned int mapping_idx) {
      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> phi_point_m_int(
          *mapping_info_faces[mapping_idx], fe_point_temp);

        for (unsigned int face = face_range.first; face < face_range.second; ++face)
          {
            phi_m_int.reinit(face);
            phi_m_int.read_dof_values(src);

            phi_m_int.project_to_face(EvaluationFlags::values | EvaluationFlags::gradients);

            const auto face_info =
              flow_scratch_data.scratch_data.get_matrix_free().get_face_info(face);

            for (unsigned int lane = 0;
                 lane <
                 flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_face_batch(
                   face);
                 ++lane)
              {
                phi_point_m_int.reinit(face_info.cells_interior[lane],
                                           static_cast<int>(face_info.interior_face_no));

                phi_point_m_int.evaluate_in_face(&phi_m_int.get_scratch_data().begin()[0][lane],
                                                     EvaluationFlags::values |
                                                       EvaluationFlags::gradients);

                const dealii::VectorizedArray<number> penalty_parameter =
                  (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0) ?
                    phi_m_int.read_cell_data(flow_scratch_data.interior_penalty_parameter) :
                    0.;

                for (const unsigned int q : phi_point_m_int.quadrature_point_indices())
                  {
                    auto [flux_m, grad_flux_m] = MeltPoolDG::Flow::rhs_boundary_face_integral_kernel<
                      dim,
                      number,
                      FEFacePointEvaluation<dim + 2, dim, dim, VectorizedArray<number>>,
                      is_gas_phase>(
                      phi_point_m_int,
                      q,
                      flow_scratch_data.scratch_data.get_matrix_free().get_boundary_id(face),
                      penalty_parameter,
                      convective_terms,
                      viscous_terms,
                      flow_scratch_data);

                    phi_point_m_int.submit_value(flux_m, q);
                    if (flow_scratch_data.flow_data.material_data_liquid_phase.dynamic_viscosity > 0)
                      phi_point_m_int.submit_gradient(grad_flux_m, q);
                  }

                phi_point_m_int.integrate_in_face(&phi_m_int.get_scratch_data().begin()[0][lane],
                                                      dealii::EvaluationFlags::values |
                                                        dealii::EvaluationFlags::gradients);
              }

            phi_m_int.collect_from_face(dealii::EvaluationFlags::values |
                                    dealii::EvaluationFlags::gradients,
                                  phi_m_int.begin_dof_values());

            phi_m_int.distribute_local_to_global(dst);
          }
    };

    switch (face_category.first) {
        case CutUtil::CellCategory::liquid: {
          process_face.template operator()<false>(phi_liquid_m);
          break;
        }
        case CutUtil::CellCategory::gas: {
          process_face.template operator()<true>(phi_gas_m);
          break;
        }
        case CutUtil::CellCategory::intersected: {
          process_intersected_faces.template operator()<false>(phi_liquid_m_intersected, 0);
          process_intersected_faces.template operator()<true>(phi_gas_m_intersected, 1);
          break;
        }
        default:
          break;
    }
  }

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::local_apply_cell_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    const auto active_fe_index =
      flow_scratch_data.scratch_data.get_matrix_free().get_cell_range_category(cell_range);

    FECellIntegrator<dim, dim + 2, number> phi_liquid =
      create_cell_integrator(CutUtil::CellCategory::liquid, 0);
    FECellIntegrator<dim, dim + 2, number> phi_gas =
      create_cell_integrator(CutUtil::CellCategory::gas, dim + 2);
    FECellIntegrator<dim, dim + 2, number> phi_liquid_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, 0);
    FECellIntegrator<dim, dim + 2, number> phi_gas_intersected =
      create_cell_integrator(CutUtil::CellCategory::intersected, dim + 2);

    // Processing function for non-intersected cells
    auto process_cell_range = [&](auto &phi) {
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src, dealii::EvaluationFlags::values);

          for (const unsigned int q : phi.quadrature_point_indices())
            phi.submit_value(phi.get_value(q), q);

          phi.integrate_scatter(dealii::EvaluationFlags::values, dst);
        }
    };

    // Processing function for intersected cells
    auto process_intersected_cell_range = [&](auto &phi_intersected, auto &phi_point) {
      constexpr unsigned int n_lanes = VectorizedArray<number>::size();

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          phi_intersected.reinit(cell);
          phi_intersected.read_dof_values(src);

          for (unsigned int lane = 0;
               lane <
               flow_scratch_data.scratch_data.get_matrix_free().n_active_entries_per_cell_batch(
                 cell);
               ++lane)
            {
              phi_point.reinit(cell * n_lanes + lane);
              phi_point.evaluate(StridedArrayView<const number, n_lanes>(
                                   &phi_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                 dealii::EvaluationFlags::values);

              for (const unsigned int q : phi_point.quadrature_point_indices())
                phi_point.submit_value(phi_point.get_value(q), q);

              phi_point.integrate(StridedArrayView<number, n_lanes>(
                                    &phi_intersected.begin_dof_values()[0][lane], n_dofs_per_cell),
                                  dealii::EvaluationFlags::values);
            }
          phi_intersected.distribute_local_to_global(dst);
        }
    };

    switch (active_fe_index)
      {
        case CutUtil::CellCategory::liquid:
          process_cell_range(phi_liquid);
          break;

        case CutUtil::CellCategory::gas:
          process_cell_range(phi_gas);
          break;

        case CutUtil::CellCategory::intersected: {
          FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> phi_point_liquid(
            *mapping_info_cells[0], fe_point_temp);
          FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> phi_point_gas(
            *mapping_info_cells[1], fe_point_temp);

          process_intersected_cell_range(phi_liquid_intersected, phi_point_liquid);
          process_intersected_cell_range(phi_gas_intersected, phi_point_gas);
        }
        break;

        default:
          break;
      }
  }

  template <unsigned int dim, typename number>
  void CompressibleMultiphaseOperator<dim, number>::local_apply_face_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType &dst,
    const VectorType &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
{
    const auto face_category =
        flow_scratch_data.scratch_data.get_matrix_free().get_face_range_category(face_range);
    const CutUtil::FaceType face_type = CutUtil::get_face_type(face_category);

    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_m =
           create_face_integrator(true, CutUtil::CellCategory::liquid, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_p_intersected =
      create_face_integrator(false, CutUtil::CellCategory::intersected, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_m =
            create_face_integrator(true, CutUtil::CellCategory::gas, dim + 2);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_p_intersected =
      create_face_integrator(false, CutUtil::CellCategory::intersected, dim + 2);
    FEFaceIntegrator<dim, dim + 2, number> phi_liquid_m_intersected =
            create_face_integrator(true, CutUtil::CellCategory::intersected, 0);
    FEFaceIntegrator<dim, dim + 2, number> phi_gas_m_intersected =
            create_face_integrator(true, CutUtil::CellCategory::intersected, dim + 2);

    const number cell_side_length = flow_scratch_data.scratch_data.get_min_cell_size();

    auto apply_ghost_penalty = [&](auto &phi_m, auto &phi_p) {
      EvaluationFlags::EvaluationFlags evaluation_flags =
        dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients |
        ((flow_scratch_data.flow_data.fe.degree == 2) ? dealii::EvaluationFlags::hessians : dealii::EvaluationFlags::nothing);

        for (unsigned int face = face_range.first; face < face_range.second; ++face) {
            phi_m.reinit(face);
            phi_m.gather_evaluate(src, evaluation_flags);

            phi_p.reinit(face);
            phi_p.gather_evaluate(src, evaluation_flags);

            for (const unsigned int q : phi_m.quadrature_point_indices()) {
                const auto u_minus = phi_m.get_value(q);
                const auto u_plus = phi_p.get_value(q);

                const auto u_normal_grad_minus = phi_m.get_normal_derivative(q);
                const auto u_normal_grad_plus = phi_p.get_normal_derivative(q);

                const auto ghost_penalty_term_0 =
                    (u_minus - u_plus) *
                    flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_0 *
                    cell_side_length;

                const auto ghost_penalty_term_1 =
                    (u_normal_grad_minus - u_normal_grad_plus) *
                    flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_1 *
                    Utilities::fixed_power<3>(cell_side_length);

                if (flow_scratch_data.flow_data.fe.degree == 2) {
                    const auto u_normal_hessian_minus = phi_m.get_normal_hessian(q);
                    const auto u_normal_hessian_plus = phi_p.get_normal_hessian(q);

                    const auto ghost_penalty_term_2 =
                        (u_normal_hessian_minus - u_normal_hessian_plus) *
                        flow_scratch_data.flow_data.cut.stabilization.ghost_penalty.gamma_M_degree_2 *
                        Utilities::fixed_power<5>(cell_side_length);

                    phi_m.submit_normal_hessian(ghost_penalty_term_2, q);
                    phi_p.submit_normal_hessian(-ghost_penalty_term_2, q);
                }
                phi_m.submit_normal_derivative(ghost_penalty_term_1, q);
                phi_p.submit_normal_derivative(-ghost_penalty_term_1, q);

                phi_m.submit_value(ghost_penalty_term_0, q);
                phi_p.submit_value(-ghost_penalty_term_0, q);
            }
            phi_m.integrate_scatter(evaluation_flags, dst);
            phi_p.integrate_scatter(evaluation_flags, dst);
        }
    };

    switch (face_type)
      {
        case CutUtil::FaceType::mixed_face_liquid: {
          apply_ghost_penalty(phi_liquid_m, phi_liquid_p_intersected);
        }
        break;

        case CutUtil::FaceType::mixed_face_gas: {
          apply_ghost_penalty(phi_gas_m, phi_gas_p_intersected);
        }
        break;

        case CutUtil::FaceType::intersected_face: {
          apply_ghost_penalty(phi_liquid_m_intersected, phi_liquid_p_intersected);
          apply_ghost_penalty(phi_gas_m_intersected, phi_gas_p_intersected);
        }
        break;

        default:
          break;
      }
}

  template <unsigned int dim, typename number>
  void
  CompressibleMultiphaseOperator<dim, number>::local_apply_boundary_face_lhs(
    const dealii::MatrixFree<dim, number> &,
    VectorType &,
    const VectorType &,
    const std::pair<unsigned, unsigned> &) const
{
    // nothing to do here
  }

  template class CompressibleMultiphaseOperator<1, double>;
  template class CompressibleMultiphaseOperator<2, double>;
  template class CompressibleMultiphaseOperator<3, double>;
} // namespace MeltPoolDG::Multiphase
