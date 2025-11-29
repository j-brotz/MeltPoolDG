
#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include "meltpooldg/flow/adaflo_wrapper_parameters.hpp"
#include "meltpooldg/flow/flow_utils.hpp"
#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/flow/adaflo_wrapper_wrapper.hpp>
#include <meltpooldg/flow/compressible_flow_material.hpp>
#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/dg_compressible_flow_operation.hpp>
#include <meltpooldg/fluid_structure_interaction/brinkman_penalization.hpp>
#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#include <meltpooldg/fluid_structure_interaction/stokes_law.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <memory>
#include <tuple>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  std::tuple<std::shared_ptr<Flow::AdditionalCellAndQuadOperation<dim, number>>,
             std::shared_ptr<Flow::AdditionalCellAndQuadOperationJacobian<dim, number>>,
             std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>>>
  setup_compressible_fluid_structure_interaction(
    const FluidStructureInteractionData<number>              &fsi_data,
    ObstacleField<dim, number, ObstacleType>                 &obstacle_field,
    const Flow::CompressibleFluidMaterialPhaseData<number>   &flow_material,
    const dealii::LinearAlgebra::distributed::Vector<number> &flow_solution,
    const MatrixFreeContext<dim, number>                      flow_mf_context)
  {
    std::shared_ptr<Flow::AdditionalCellAndQuadOperation<dim, number>> fsi_fluid_force_residual;
    std::shared_ptr<Flow::AdditionalCellAndQuadOperationJacobian<dim, number>>
                                                             fsi_fluid_force_jacobian;
    std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>> fsi_obstacle_load;

    switch (fsi_data.fsi_coupling_method)
      {
          case FSICouplingMethod::brinkman_penalization: {
            fsi_fluid_force_residual =
              std::make_shared<BrinkmanPenalizationResidualContribution<dim, number, ObstacleType>>(
                obstacle_field, fsi_data.brinkman_penalization_data);
            fsi_fluid_force_jacobian =
              std::make_shared<BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>>(
                obstacle_field, fsi_data.brinkman_penalization_data);
            fsi_obstacle_load = std::make_unique<ObstacleLoad<dim, number, ObstacleType>>(
              BrinkmanObstacleForce<dim, number, ObstacleType>(obstacle_field,
                                                               flow_solution,
                                                               flow_mf_context,
                                                               fsi_data.brinkman_penalization_data,
                                                               FlowSolverType::compressible));
            break;
          }
          case FSICouplingMethod::stokes_law: {
            fsi_fluid_force_residual =
              std::make_shared<StokesLawFluidForce<dim, number, ObstacleType>>(
                flow_solution, obstacle_field, flow_material.dynamic_viscosity);
            // TODO: The Stokes law coupling currently only works for explicit methods
            fsi_fluid_force_jacobian =
              std::make_shared<BrinkmanPenalizationJacobianContribution<dim, number, ObstacleType>>(
                obstacle_field, fsi_data.brinkman_penalization_data);
            fsi_obstacle_load = std::make_unique<ObstacleLoad<dim, number, ObstacleType>>(
              StokesLawSphericalParticleForce<dim, number, ObstacleType>(
                flow_solution, flow_mf_context, flow_material.dynamic_viscosity));
            break;
          }
        default:
          AssertThrow(
            false,
            dealii::ExcMessage(
              "The provided FSI coupling method is not supported within the compressible flow solver."));
      }

    return {fsi_fluid_force_residual, fsi_fluid_force_jacobian, std::move(fsi_obstacle_load)};
  }

  template <int dim, typename number, typename ObstacleType>
  std::tuple<std::shared_ptr<Flow::IncompressibleExternalFluidForce<dim, number, dim>>,
             std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>>>
  setup_incompressible_fluid_structure_interaction(
    const FluidStructureInteractionData<number>              &fsi_data,
    ObstacleField<dim, number, ObstacleType>                 &obstacle_field,
    const Flow::AdafloWrapperParameters<number>              &flow_params,
    const dealii::LinearAlgebra::distributed::Vector<number> &velocity_solution,
    const MatrixFreeContext<dim, number>                      flow_mf_context)
  {
    std::shared_ptr<Flow::IncompressibleExternalFluidForce<dim, number, dim>> fsi_fluid_force;
    std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>>                  fsi_obstacle_load;

    switch (fsi_data.fsi_coupling_method)
      {
          case FSICouplingMethod::brinkman_penalization: {
            fsi_fluid_force = std::make_shared<
              IncompressibleBrinkmanPenalizationFluidForce<dim, number, dim, ObstacleType>>(
              obstacle_field, fsi_data.brinkman_penalization_data, flow_params.params.density);
            fsi_obstacle_load = std::make_unique<ObstacleLoad<dim, number, ObstacleType>>(
              BrinkmanObstacleForce<dim, number, ObstacleType>(obstacle_field,
                                                               velocity_solution,
                                                               flow_mf_context,
                                                               fsi_data.brinkman_penalization_data,
                                                               FlowSolverType::incompressible,
                                                               flow_params.params.density));
            break;
          }
        default:
          AssertThrow(
            false,
            dealii::ExcMessage(
              "The provided FSI coupling method is not supported within the incompressible flow solver."));
      }

    return {fsi_fluid_force, std::move(fsi_obstacle_load)};
  }

  template <int dim, typename number, typename ObstacleType>
  struct AddFSIForceVisitor
  {
    FluidStructureInteractionData<number>    &fsi_data;
    ObstacleField<dim, number, ObstacleType> &obstacle_field;

    std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>>
    operator()(Flow::DGCompressibleFlowOperation<dim, number> &operation)
    {
      auto [fsi_residual, fsi_jacobian, fsi_obstacle_load] =
        setup_compressible_fluid_structure_interaction(fsi_data,
                                                       obstacle_field,
                                                       operation.get_phase_material_data(),
                                                       operation.get_solution(),
                                                       operation.get_matrix_free_context());

      operation.add_external_force(fsi_residual, fsi_jacobian);
      return fsi_obstacle_load;
    }

    std::unique_ptr<ObstacleLoad<dim, number, ObstacleType>>
    operator()(Flow::IncompressibleFlowSolverWrapper<dim, number> &operation)
    {
      auto [fsi_fluid_force, fsi_obstacle_load] =
        setup_incompressible_fluid_structure_interaction(fsi_data,
                                                         obstacle_field,
                                                         operation.get_flow_params(),
                                                         operation.get_velocity_solution(),
                                                         operation.get_matrix_free_context_u());

      operation.add_external_force(fsi_fluid_force);
      return fsi_obstacle_load;
    }
  };
} // namespace MeltPoolDG