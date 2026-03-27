#pragma once
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/phase_coupling_data.hpp>
#include <meltpooldg/compressible_flow/solver_data.hpp>
#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/darcy_damping_data.hpp>
#include <meltpooldg/phase_change/phase_change_data.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <string>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Struct that manages all relevant parameters for compressible multiphase flow simulations.
   */
  template <typename number>
  struct CompressibleMultiphaseCaseParameters final : public ParametersBase
  {
  protected:
    /**
     * @brief Add all relevant parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm) override
    {
      base.add_parameters(prm);
      flow.add_parameters(prm);
      material_gas.add_parameters(prm, true /*is_gas*/);
      material_liquid.add_parameters(prm, false /*is_gas*/);
      phase_change.add_parameters(prm);
      cut.add_parameters(prm);
      phase_coupling.add_parameters(prm);
      darcy_damping.add_parameters(prm);
      time_stepping.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    /**
     * @brief Post-process parameters.
     *
     * @param parameter_filename Name of the parameter file.
     */
    void
    post(const std::string &parameter_filename) override final
    {
      output.post(time_stepping.time_step_size, parameter_filename);
      flow.post(base.fe, base.verbosity_level);
      material_gas.post(true /*is_gas*/);
      material_liquid.post(false /*is_gas*/);
      phase_coupling.post();

      // check input parameters for validity
      profiling.check_input_parameters(time_stepping.time_step_size);

      // The face-based ghost-penalty stabilization is only implemented for polynomial degrees 1
      // and 2. The third derivatives on the element faces are currently not accessible in
      // deal.II.
      AssertThrow(base.fe.degree <= 2,
                  dealii::ExcMessage(
                    "Currently, only polynomial degrees 1 and 2 are implemented for cutDG."));
    }

  public:
    /// Simulation basic data
    BaseData base;

    /// Data specific for compressible flow simulations
    CompressibleFlow::CompressibleFlowData<number> flow;

    /// Material parameters for the gas phase
    CompressibleFlow::CompressibleFluidMaterialPhaseData<number> material_gas;

    /// Material parameters for the liquid phase
    CompressibleFlow::CompressibleFluidMaterialPhaseData<number> material_liquid;

    /// Parameters related to liquid-gas and solid-liquid phase transitions
    PhaseChangeData<number> phase_change;

    /// Cut-related data
    CompressibleFlow::CompressibleFlowCutData<number> cut;

    /// Data for phase-coupling including interface jump conditions and numerical parameters
    CompressibleFlowPhaseCouplingData<number> phase_coupling;

    /// Data for the darcy damping term in the solid domain and the mushy zone
    Flow::DarcyDampingData<number> darcy_damping;

    /// Data for time stepping
    TimeIntegration::TimeSteppingData<number> time_stepping;

    /// Data for output
    OutputData<number> output;

    /// Data for profiling
    Profiling::ProfilingData<number> profiling;
  };

  /**
   * @brief Case base class for compressible multiphase flow cases.
   */
  template <int dim, typename number>
  class CompressibleMultiphaseCase : public SimulationCaseBase<dim, number>
  {
  public:
    /// Case-specific parameters
    CompressibleMultiphaseCaseParameters<number> parameters;

    /**
     * @brief Constructor.
     *
     * @param parameter_file_in Parameter file that contains simulation input settings.
     * @param mpi_communicator_in The MPI communicator used to run the simulation in parallel.
     */
    explicit CompressibleMultiphaseCase(const std::string &parameter_file_in,
                                        MPI_Comm           mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }

  protected:
    /**
     * @brief Compute and print the relative error norm.
     *
     * This function calculates and prints the l2-norm/l2-error of the solution given in
     * @p generic_data_out compared to a reference state given by the function
     * @p reference_function, i.e. it computes ||solution-reference||_2 for the primary variables
     * density, momentum and energy density. The string @p norm_name is printed to the console to
     * indicate what the calculated values represent. If the reference solution represents an exact
     * analytical solution, the norm_name @p "error" should be chosen. Otherwise, the @p norm_name
     * "norm" is appropriate for the l2-norm of the deviation to a certain defined reference state,
     * e.g. initial conditions. It is necessary to provide a reference function for both cases.
     *
     * @param generic_data_out The postprocessing data.
     * @param reference_function Analytical reference function for error/norm computation.
     * @param norm_name Choose between "norm" and "error".
     */
    void
    print_relative_norm(const GenericDataOut<dim, number> &generic_data_out,
                        dealii::Function<dim>             &reference_function,
                        const std::string                 &norm_name = "norm") const
    {
      const dealii::ConditionalOStream pcout(std::cout,
                                             dealii::Utilities::MPI::this_mpi_process(
                                               this->mpi_communicator) == 0 and
                                               parameters.base.verbosity_level >= 1);

      const auto &dof_vector  = generic_data_out.get_vector("density_liquid");
      const auto &dof_handler = generic_data_out.get_dof_handler("density_liquid");

      const unsigned int n_dofs_per_element_per_phase =
        0.5 * dof_handler.get_fe_collection().max_dofs_per_cell();
      reference_function.set_time(generic_data_out.get_time());
      number errors_squared[3] = {};

      // reading DoF-vector and DoFHandler of level-set
      // (an assert is thrown in the getter functions, if 'level_set' can not be found in
      // the entries of 'generic_data_out')
      const auto &level_set             = generic_data_out.get_vector("level_set");
      const auto &level_set_dof_handler = generic_data_out.get_dof_handler("level_set");

      // generate FESystem
      const unsigned int    fe_degree = dof_handler.get_fe_collection().max_degree();
      dealii::FE_DGQ<dim>   fe_q(fe_degree);
      dealii::FESystem<dim> fe_point_temp(fe_q, dim + 2);

      // classify cells
      dealii::NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler, level_set);
      mesh_classifier.reclassify();

      // compute quadrature
      dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
        mapping_info_cell_liquid(generic_data_out.get_mapping(),
                                 dealii::update_values | dealii::update_JxW_values);
      dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
        mapping_info_cell_gas(generic_data_out.get_mapping(),
                              dealii::update_values | dealii::update_JxW_values);

      dealii::hp::QCollection<1> q_collection((dealii::QGauss<1>(fe_degree + 1)));
      dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
        q_collection, level_set_dof_handler, level_set);

      auto physical_liquid = [&](const typename dealii::DoFHandler<dim>::active_cell_iterator &i) {
        return (mesh_classifier.location_to_level_set(i) ==
                dealii::NonMatching::LocationToLevelSet::intersected) or
               (mesh_classifier.location_to_level_set(i) ==
                dealii::NonMatching::LocationToLevelSet::outside);
      };
      auto physical_gas = [&](const typename dealii::DoFHandler<dim>::active_cell_iterator &i) {
        return (mesh_classifier.location_to_level_set(i) ==
                dealii::NonMatching::LocationToLevelSet::intersected) or
               (mesh_classifier.location_to_level_set(i) ==
                dealii::NonMatching::LocationToLevelSet::inside);
      };

      const auto physical_active_cell_iterators_liquid =
        dof_handler.active_cell_iterators() | physical_liquid;
      const auto physical_active_cell_iterators_gas =
        dof_handler.active_cell_iterators() | physical_gas;

      std::vector<dealii::Quadrature<dim>> quad_vec_cell_liquid;
      std::vector<dealii::Quadrature<dim>> quad_vec_cell_gas;

      const unsigned int n_active_cells = std::distance(dof_handler.active_cell_iterators().begin(),
                                                        dof_handler.active_cell_iterators().end());

      for (const auto &cell : physical_active_cell_iterators_liquid)
        {
          quadrature_generator.generate(cell);
          // liquid phase has positive level-set
          quad_vec_cell_liquid.push_back(quadrature_generator.get_outside_quadrature());
        }

      for (const auto &cell : physical_active_cell_iterators_gas)
        {
          quadrature_generator.generate(cell);
          // gas phase has negative level-set
          quad_vec_cell_gas.push_back(quadrature_generator.get_inside_quadrature());
        }

      mapping_info_cell_liquid.reinit_cells(physical_active_cell_iterators_liquid,
                                            quad_vec_cell_liquid,
                                            n_active_cells);
      mapping_info_cell_gas.reinit_cells(physical_active_cell_iterators_gas,
                                         quad_vec_cell_gas,
                                         n_active_cells);

      dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
        fe_point_eval_liquid(mapping_info_cell_liquid, fe_point_temp);
      dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
        fe_point_eval_gas(mapping_info_cell_gas, fe_point_temp);

      auto compute_error_norm = [&](auto      &fe_point_eval,
                                    auto      &physical_domain,
                                    const bool is_gas_phase) {
        const unsigned int offset = is_gas_phase ? 0 : dim + 2;

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_artificial() or cell->is_ghost() or (not physical_domain(cell)))
              continue;

            const unsigned int  n_dofs_per_cell = cell->get_fe().dofs_per_cell;
            std::vector<number> solution_values_in(n_dofs_per_element_per_phase);

            if (n_dofs_per_cell != n_dofs_per_element_per_phase)
              {
                // For an intersected element, we have to extract the Dofs, which are relevant for
                // the current phase
                std::vector<number> solution_values_in_tmp(n_dofs_per_cell);
                cell->get_dof_values(dof_vector,
                                     solution_values_in_tmp.begin(),
                                     solution_values_in_tmp.end());

                if (is_gas_phase)
                  solution_values_in.assign(solution_values_in_tmp.begin() +
                                              n_dofs_per_element_per_phase,
                                            solution_values_in_tmp.end());
                else
                  solution_values_in.assign(solution_values_in_tmp.begin(),
                                            solution_values_in_tmp.begin() +
                                              n_dofs_per_element_per_phase);
              }
            else
              cell->get_dof_values(dof_vector,
                                   solution_values_in.begin(),
                                   solution_values_in.end());

            fe_point_eval.reinit(cell->active_cell_index());
            fe_point_eval.evaluate(solution_values_in, dealii::EvaluationFlags::values);

            dealii::VectorizedArray<number> local_errors_squared[3] = {};

            for (const unsigned int q : fe_point_eval.quadrature_point_indices())
              {
                const auto                      solution       = fe_point_eval.get_value(q);
                dealii::VectorizedArray<number> error[dim + 2] = {};
                for (unsigned int i = 0; i < dim + 2; ++i)
                  {
                    error[i] =
                      VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                        reference_function, fe_point_eval.quadrature_point(q), i + offset) -
                      solution[i];
                  }

                const auto JxW          = fe_point_eval.JxW(q);
                local_errors_squared[0] = error[0] * error[0] * JxW;

                for (unsigned int d = 0; d < dim; ++d)
                  local_errors_squared[1] = (error[d + 1] * error[d + 1]) * JxW;

                local_errors_squared[2] = (error[dim + 1] * error[dim + 1]) * JxW;

                for (unsigned int v = 0; v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                     ++v)
                  for (unsigned int d = 0; d < 3; ++d)
                    errors_squared[d] += local_errors_squared[d][v];
              }
          }
      };

      compute_error_norm(fe_point_eval_liquid, physical_liquid, false);
      compute_error_norm(fe_point_eval_gas, physical_gas, true);

      dealii::Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);
      std::array<number, 3> errors{};
      for (unsigned int d = 0; d < 3; ++d)
        errors[d] = std::sqrt(errors_squared[d]);

      std::ostringstream output;
      output << std::scientific << std::setprecision(4) << norm_name << " rho: " << errors[0]
             << ", rho * u: " << errors[1] << ", rho * energy: " << errors[2];
      Journal::print_line(pcout, output.str(), "compressible_multiphase");
    }
  };
} // namespace MeltPoolDG::Multiphase
