#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/mapping_info.h>

#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/solver_data.hpp>
#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <string>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Struct that manages all relevant parameters for compressible flow simulations.
   */
  template <typename number>
  struct CompressibleFlowCaseParameters final : public ParametersBase
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
      material.add_parameters(prm, true /*is_gas*/);
      cut.add_parameters(prm);
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
      material.post(true /*is_gas*/);

      // check input parameters for validity
      profiling.check_input_parameters(time_stepping.time_step_size);

      // Advanced EOS are currently only allowed for explicit time integration.
      if (material.eos_type != EquationOfState::ideal_gas)
        AssertThrow(not MeltPoolDG::TimeIntegration::time_integrator_scheme_is_explicit(
                      flow.time_integrator.integrator_type),
                    dealii::ExcMessage(
                      "Only the ideal gas EOS is allowed for implicit time integration."));

      if (flow.domain_representation_type == "cut")
        {
          // The face-based ghost-penalty stabilization is only implemented for polynomial degrees 1
          // and 2. The third derivatives on the element faces are currently not accessible in
          // deal.II.
          AssertThrow(base.fe.degree <= 2,
                      dealii::ExcMessage(
                        "Currently, only polynomial degrees 1 and 2 are implemented for cutDG."));
        }
    }

  public:
    /// Simulation basic data
    BaseData base;

    /// Data specific for compressible flow simulations
    CompressibleFlowData<number> flow;

    /// Material parameters for a compressible (or nearly incompressible) fluid
    CompressibleFluidMaterialPhaseData<number> material;

    /// Cut-related data (only relevant for cutDG)
    CompressibleFlowCutData<number> cut;

    /// Data for time stepping
    TimeIntegration::TimeSteppingData<number> time_stepping;

    /// Data for output
    OutputData<number> output;

    /// Data for profiling
    Profiling::ProfilingData<number> profiling;
  };

  /**
   * @brief Case base class for compressible flow cases.
   */
  template <int dim, typename number>
  class CompressibleFlowCase : public SimulationCaseBase<dim, number>
  {
  public:
    /// Case-specific parameters
    CompressibleFlowCaseParameters<number> parameters;

    /**
     * @brief Constructor.
     *
     * @param parameter_file_in Parameter file that contains simulation input settings.
     * @param mpi_communicator_in The MPI communicator used to run the simulation in parallel.
     */
    explicit CompressibleFlowCase(const std::string &parameter_file_in,
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

      const auto &dof_vector  = generic_data_out.get_vector("density");
      const auto &dof_handler = generic_data_out.get_dof_handler("density");

      reference_function.set_time(generic_data_out.get_time());

      std::map<std::string, number> component_errors;

      std::vector<std::string> labels;

      if (parameters.flow.domain_representation_type == "fitted")
        {
          // In the fitted mesh case, we can use dealii vector tools to compute the error norms
          // based on the DoFHandler and DoFVector of the solution. We loop over the components of
          // the solution and compute the error norms separately for each component. We also take
          // into account the case that some components might be part of a vector (e.g. the velocity
          // components) and compute the error norm for the entire vector instead of the individual
          // components in that case.
          const auto &data_postprocessor = generic_data_out.get_data_postprocessor("density");

          const auto &names          = data_postprocessor.get_names();
          const auto &interpretation = data_postprocessor.get_data_component_interpretation();

          const unsigned int n_components = names.size();

          // Initialize errors
          for (const auto &name : names)
            component_errors[name] = 0.0;

          unsigned int i = 0;
          while (i < n_components)
            {
              const std::string &component_name = names[i];
              labels.push_back(component_name);

              // Determine component range and create mask
              unsigned int begin = i;
              unsigned int end   = i + 1;

              if (interpretation[i] ==
                  dealii::DataComponentInterpretation::component_is_part_of_vector)
                {
                  while (end < n_components and names[end] == component_name and
                         interpretation[end] ==
                           dealii::DataComponentInterpretation::component_is_part_of_vector)
                    {
                      ++end;
                    }
                }
              dealii::ComponentSelectFunction<dim> component_mask({begin, end}, n_components);

              // Compute global error
              component_errors[component_name] =
                VectorTools::compute_global_error_norm(dof_vector,
                                                       dof_handler.get_triangulation(),
                                                       generic_data_out.get_mapping(),
                                                       dof_handler,
                                                       dealii::QGauss<dim>(
                                                         dof_handler.get_fe().degree),
                                                       dealii::VectorTools::NormType::L2_norm,
                                                       reference_function,
                                                       &component_mask);

              // Move to next component/group
              i = end;
            }
        }
      else
        {
          // if cutDG is used, we evaluate the error at the quadrature points

          // reading DoF-vector and DoFHandler of level-set
          // (an assert is thrown in the getter functions, if 'level_set' can not be found in
          // the entries of 'generic_data_out')
          const auto &level_set             = generic_data_out.get_vector("level_set");
          const auto &level_set_dof_handler = generic_data_out.get_dof_handler("level_set");

          number errors_squared[3] = {};

          // generate FESystem
          const unsigned int    fe_degree = dof_handler.get_fe_collection().max_degree();
          dealii::FE_DGQ<dim>   fe_q(fe_degree);
          dealii::FESystem<dim> fe_point_temp(fe_q, dim + 2);

          // classify cells
          dealii::NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler,
                                                                   level_set);
          mesh_classifier.reclassify();

          // compute quadrature
          dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>>
            mapping_info_cell(generic_data_out.get_mapping(),
                              dealii::update_values | dealii::update_JxW_values);

          dealii::hp::QCollection<1> q_collection((dealii::QGauss<1>(fe_degree + 1)));
          dealii::NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(
            q_collection, level_set_dof_handler, level_set);

          auto physical = [&](const typename dealii::DoFHandler<dim>::active_cell_iterator &i) {
            return (mesh_classifier.location_to_level_set(i) ==
                    dealii::NonMatching::LocationToLevelSet::intersected) or
                   (mesh_classifier.location_to_level_set(i) ==
                    dealii::NonMatching::LocationToLevelSet::
                      outside /*active phase has positive level-set*/);
          };

          const auto physical_active_cell_iterators =
            dof_handler.active_cell_iterators() | physical;

          std::vector<dealii::Quadrature<dim>> quad_vec_cell;

          unsigned int n_active_cells = std::distance(dof_handler.active_cell_iterators().begin(),
                                                      dof_handler.active_cell_iterators().end());

          for (const auto &cell : physical_active_cell_iterators)
            {
              quadrature_generator.generate(cell);
              // active phase has positive level-set
              quad_vec_cell.push_back(quadrature_generator.get_outside_quadrature());
            }

          mapping_info_cell.reinit_cells(physical_active_cell_iterators,
                                         quad_vec_cell,
                                         n_active_cells);

          // cell-loop for error-norm computation
          dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>>
            fe_point_eval(mapping_info_cell, fe_point_temp);

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_artificial() or cell->is_ghost() or (not physical(cell)))
                continue;

              std::vector<number> solution_values_in(cell->get_fe().dofs_per_cell);

              cell->get_dof_values(dof_vector,
                                   solution_values_in.begin(),
                                   solution_values_in.end());

              fe_point_eval.reinit(cell->active_cell_index());
              fe_point_eval.evaluate(solution_values_in, dealii::EvaluationFlags::values);

              std::array<dealii::VectorizedArray<number>, 3> local_errors_squared = {};

              for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                {
                  const auto error =
                    VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
                      reference_function, fe_point_eval.quadrature_point(q)) -
                    fe_point_eval.get_value(q);

                  const auto JxW          = fe_point_eval.JxW(q);
                  local_errors_squared[0] = error[0] * error[0] * JxW;

                  local_errors_squared[1] = 0.;
                  for (unsigned int d = 0; d < dim; ++d)
                    local_errors_squared[1] += (error[d + 1] * error[d + 1]) * JxW;
                  local_errors_squared[2] = (error[dim + 1] * error[dim + 1]) * JxW;

                  for (unsigned int v = 0;
                       v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                       ++v)
                    for (unsigned int d = 0; d < 3; ++d)
                      errors_squared[d] += local_errors_squared[d][v];
                }
            }

          dealii::Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);
          std::array<number, 3> errors;
          for (unsigned int d = 0; d < 3; ++d)
            errors[d] = std::sqrt(errors_squared[d]);
          labels                           = {"density", "momentum", "total energy"};
          component_errors["density"]      = errors[0];
          component_errors["momentum"]     = errors[1];
          component_errors["total energy"] = errors[2];
        }

      Journal::print_line(pcout, " Solution (L2 " + norm_name + ")", "compressible_flow");
      for (const auto &label : labels)
        {
          std::ostringstream oss;
          oss << "   " << std::left << std::setw(14) << label << ": " << std::scientific
              << std::setprecision(4) << component_errors[label];
          Journal::print_line(pcout, oss.str());
        }
    }
  };
} // namespace MeltPoolDG::Flow
