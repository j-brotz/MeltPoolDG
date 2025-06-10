#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_boundary_conditions.hpp>
#include <meltpooldg/flow/compressible_flow_cut_data.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/flow/compressible_flow_material.hpp>
#include <meltpooldg/flow/compressible_flow_utils.hpp>
#include <meltpooldg/flow/compressible_multiphase/compressible_flow_phase_coupling_data.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * A struct providing the relevant data required by all compressible single-phase flow solvers.
   */
  template <int dim, typename number>
  struct CompressibleFlowScratchData
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor
     *
     * @param flow_data_in Reference to the flow data object.
     * @param material_data_in Reference to the material data object.
     * @param scratch_data_in Reference to the scratch data object.
     * @param dof_idx_in Relevant dof index of the flow solver in the scratch data object.
     * @param quad_idx_in Relevant quadrature index of the flow solver in the scratch data object.
     * @param cut_data_in Pointer to the cut data object.
     *
     * @note The cut data object @param cut_data_in is optional. For non-cut applications, it does not
     * need to be passed to the constructor.
     */
    CompressibleFlowScratchData(const CompressibleFlowData<number>               &flow_data_in,
                                const CompressibleFluidMaterialPhaseData<number> &material_data_in,
                                const ScratchData<dim, dim, number>              &scratch_data_in,
                                const unsigned int                                dof_idx_in,
                                const unsigned int                                quad_idx_in,
                                const CompressibleFlowCutData<number> *cut_data_in = nullptr)
      : flow_data(flow_data_in)
      , scratch_data(scratch_data_in)
      , material(material_data_in)
      , cut(cut_data_in)
      , dof_idx(dof_idx_in)
      , quad_idx(quad_idx_in)
    {
      is_viscous = material.data.dynamic_viscosity > 0.;

      if (flow_data.domain_representation_type == "cut")
        AssertThrow(
          cut_data_in != nullptr,
          dealii::ExcMessage(
            "A CompressibleFlowCutData object has to be provided to the constructor of "
            "CompressibleFlowScratchData in the case of domain_representation_type = 'cut'."));
    }

    /**
     * Set up the internal data structures, i.e. allocate memory for the solution history object and
     * precompute the penalty parameter for the symmetric interior penalty method.
     *
     * @param solution_history_size Size of the solution history object, i.e. the number of vectors
     * at different concrete times n for which the solution history is responsible.
     */
    void
    reinit(const unsigned solution_history_size)
    {
      solution_history.resize(solution_history_size);
      solution_history.apply(
        [&scratch_data = scratch_data, comp_flow_dof_idx = dof_idx](VectorType &v) {
          scratch_data.initialize_dof_vector(v, comp_flow_dof_idx);
        });

      if (is_viscous)
        calculate_penalty_parameter(interior_penalty_parameter,
                                    scratch_data.get_matrix_free(),
                                    flow_data.domain_representation_type,
                                    dof_idx);
    }

    const CompressibleFlowData<number> flow_data;

    const ScratchData<dim, dim, number> &scratch_data;

    const CompressibleFlowMaterial<dim, number> material;

    // only relevant for cut applications
    const CompressibleFlowCutData<number> *cut = nullptr;

    const unsigned int dof_idx  = 0;
    const unsigned int quad_idx = 0;

    bool is_viscous = false;

    CompressibleFlowBoundaryConditions<dim, number> boundary_conditions;

    dealii::AlignedVector<dealii::VectorizedArray<number>> interior_penalty_parameter;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::unique_ptr<dealii::Function<dim>> body_force;
  };

  /**
   * A struct providing the relevant data required by all compressible multiphase flow solvers.
   */
  template <int dim, typename number>
  struct CompressibleMultiphaseScratchData
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor
     *
     * @param flow_data_in Reference to the flow data object.
     * @param material_data_gas_in Reference to the material data object for the gas phase.
     * @param material_data_liquid_in Reference to the material data object for the liquid phase.
     * @param cut_data_in Pointer to the cut data object.
     * @param phase_coupling_data_in Reference to the data object for phase coupling parameters.
     * @param scratch_data_in Reference to the scratch data object.
     * @param dof_idx_in Relevant dof index of the flow solver in the scratch data object.
     * @param quad_idx_in Relevant quadrature index of the flow solver in the scratch data object.
     */
    CompressibleMultiphaseScratchData(
      const CompressibleFlowData<number>                          &flow_data_in,
      const CompressibleFluidMaterialPhaseData<number>            &material_data_gas_in,
      const CompressibleFluidMaterialPhaseData<number>            &material_data_liquid_in,
      const CompressibleFlowCutData<number>                       &cut_data_in,
      const Multiphase::CompressibleFlowPhaseCouplingData<number> &phase_coupling_data_in,
      const ScratchData<dim, dim, number>                         &scratch_data_in,
      const unsigned int                                           dof_idx_in,
      const unsigned int                                           quad_idx_in)
      : flow_data(flow_data_in)
      , scratch_data(scratch_data_in)
      , material_gas(material_data_gas_in)
      , material_liquid(material_data_liquid_in)
      , cut(cut_data_in)
      , phase_coupling(phase_coupling_data_in)
      , dof_idx(dof_idx_in)
      , quad_idx(quad_idx_in)
    {
      is_viscous =
        material_gas.data.dynamic_viscosity > 0. or material_liquid.data.dynamic_viscosity > 0.;
    }

    /**
     * Set up the internal data structures, i.e. allocate memory for the solution history object and
     * precompute the penalty parameter for the symmetric interior penalty method.
     *
     * @param solution_history_size Size of the solution history object, i.e. the number of vectors
     * at different concrete times n for which the solution history is responsible.
     */
    void
    reinit(const unsigned solution_history_size)
    {
      solution_history.resize(solution_history_size);
      solution_history.apply(
        [&scratch_data = scratch_data, comp_flow_dof_idx = dof_idx](VectorType &v) {
          scratch_data.initialize_dof_vector(v, comp_flow_dof_idx);
        });

      if (is_viscous)
        calculate_penalty_parameter(interior_penalty_parameter,
                                    scratch_data.get_matrix_free(),
                                    flow_data.domain_representation_type,
                                    dof_idx);
    }

    const CompressibleFlowData<number> flow_data;

    const ScratchData<dim, dim, number> &scratch_data;

    const CompressibleFlowMaterial<dim, number> material_gas;
    const CompressibleFlowMaterial<dim, number> material_liquid;

    const CompressibleFlowCutData<number> cut;

    const Multiphase::CompressibleFlowPhaseCouplingData<number> phase_coupling;

    const unsigned int dof_idx  = 0;
    const unsigned int quad_idx = 0;

    bool is_viscous = false;

    CompressibleFlowBoundaryConditions<dim, number> boundary_conditions;

    dealii::AlignedVector<dealii::VectorizedArray<number>> interior_penalty_parameter;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::unique_ptr<dealii::Function<dim>> body_force;
  };
} // namespace MeltPoolDG::Flow
