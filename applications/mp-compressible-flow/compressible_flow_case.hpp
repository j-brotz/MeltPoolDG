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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/post_processing/output_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>
#include <meltpooldg/utilities/functions.hpp>
#include <meltpooldg/utilities/profiling_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

#include <string>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * @brief Struct that manages all relevant parameters for compressible flow simulations.
   */
  template <typename number>
  struct CaseParameters final : public ParametersBase
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
    OperationData<number> flow;

    /// Material parameters for a compressible (or nearly incompressible) fluid
    MaterialPhaseData<number> material;

    /// Cut-related data (only relevant for cutDG)
    CutSolverData<number> cut;

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
  class Case : public SimulationCaseBase<dim, number>
  {
  public:
    /// Case-specific parameters
    CaseParameters<number> parameters;

    /**
     * @brief Constructor.
     *
     * @param parameter_file_in Parameter file that contains simulation input settings.
     * @param mpi_communicator_in The MPI communicator used to run the simulation in parallel.
     */
    explicit Case(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      dealii::ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }

  protected:
    struct DataPostprocessorData
    {
      const std::string            name;
      const dealii::Function<dim> &reference_function;
    };

    /**
     * @brief Pretty-prints component-wise error norms to the console.
     *
     * This function formats and outputs the values of a given norm for multiple labeled components
     * together with their labels.
     *
     * @param norm_name Name of the norm being printed which is included in the output header.
     * @param labels List of component labels defining the order in which values are printed.
     * @param component_errors Mapping from component labels to their corresponding error values.
     * @param label_width Width used for left-aligning labels in the output.
     *
     * @throws If a given label does not have a corresponding entry in the component_errors map.
     */
    void
    pretty_print_norm(const std::string                   &norm_name,
                      const std::vector<std::string>      &labels,
                      const std::map<std::string, number> &component_errors,
                      const unsigned                       label_width = 14) const
    {
      const dealii::ConditionalOStream pcout(std::cout,
                                             dealii::Utilities::MPI::this_mpi_process(
                                               this->mpi_communicator) == 0 and
                                               parameters.base.verbosity_level >= 1);

      Journal::print_line(pcout, " Solution (" + norm_name + ")", "compressible_flow");
      for (const auto &label : labels)
        {
          std::ostringstream oss;
          oss << "   " << std::left << std::setw(label_width) << label << ": " << std::scientific
              << std::setprecision(4) << component_errors.at(label);
          Journal::print_line(pcout, oss.str());
        }
    }

    /**
     * For cases which do not use cut methods, we can directly use the DoFHandler and DoFVector of
     * the solution to compute error norms. This function computes and prints the relative error
     * norm for fitted methods based on the register data post processor in the generic_data_out
     * object. Only those relative errors given in the data vector are printed to the console.
     *
     * @param generic_data_out The postprocessing data.
     * @param data Vector of names and reference functions for the components for which the
     * error/norm should be computed and printed.
     * @param norm_name Name of the norm. Typically either "norm" or "error" depending on whether
     * the reference function represents an exact solution or a certain reference state, e.g.
     * initial conditions.
     *
     * @throws If a name given in the data vector does not have a corresponding registered data.
     */
    void
    print_relative_norm_fitted(const GenericDataOut<dim, number> &generic_data_out,
                               std::vector<DataPostprocessorData> data,
                               const std::string                 &norm_name = "norm") const
    {
      std::map<std::string, number> component_errors;

      std::vector<std::string> labels;

      // In the fitted mesh case, we can use dealii vector tools to compute the error norms
      // based on the DoFHandler and DoFVector of the solution. We loop over the components of
      // the solution and compute the error norms separately for each component. We also take
      // into account the case that some components might be part of a vector (e.g. the velocity
      // components) and compute the error norm for the entire vector instead of the individual
      // components in that case.
      for (const auto &postprocessor_data : data)
        {
          const dealii::DataPostprocessor<dim> &post_processor =
            generic_data_out.get_data_postprocessor(postprocessor_data.name);

          const dealii::DoFHandler<dim> &dof_handler =
            generic_data_out.get_dof_handler(postprocessor_data.name);

          const auto &solution = generic_data_out.get_vector(postprocessor_data.name);

          // Identify the relevant ranges for the specific name
          const auto &names          = post_processor.get_names();
          const auto &interpretation = post_processor.get_data_component_interpretation();

          std::pair<unsigned int, unsigned int> component_range = {0, 0};
          bool                                  name_found      = false;
          unsigned                              i               = 0;
          while (i < names.size() and not name_found)
            {
              if (names[i] == postprocessor_data.name)
                {
                  name_found             = true;
                  component_range.first  = i;
                  component_range.second = i + 1;
                  if (interpretation[i] ==
                      dealii::DataComponentInterpretation::component_is_part_of_vector)
                    {
                      while (component_range.second < interpretation.size() and
                             names[component_range.second] == postprocessor_data.name and
                             interpretation[component_range.second] ==
                               dealii::DataComponentInterpretation::component_is_part_of_vector)
                        {
                          ++component_range.second;
                        }
                    }
                }
              ++i;
            }

          AssertThrow(
            name_found,
            dealii::ExcMessage(
              "The name given in the data vector does not have a corresponding registered data "
              "postprocessor."));

          // Create embedded reference function to match fe element of used dof handler
          MeltPoolDG::Functions::EmbeddedComponentsFunction<dim, number> reference_function(
            postprocessor_data.reference_function,
            dof_handler.get_fe().n_components(),
            component_range.first);

          dealii::ComponentSelectFunction<dim> component_mask({component_range.first,
                                                               component_range.second},
                                                              dof_handler.get_fe().n_components());

          // Compute global error
          component_errors[postprocessor_data.name] =
            VectorTools::compute_global_error_norm(solution,
                                                   dof_handler.get_triangulation(),
                                                   generic_data_out.get_mapping(),
                                                   dof_handler,
                                                   dealii::QGauss<dim>(dof_handler.get_fe().degree),
                                                   dealii::VectorTools::NormType::L2_norm,
                                                   reference_function,
                                                   &component_mask);

          labels.push_back(postprocessor_data.name);
        }

      pretty_print_norm("L2 " + norm_name, labels, component_errors, 20);
    }
    /**
     * @brief Compute and print the relative error norm for cut methods.
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
    print_relative_norm_cut(const GenericDataOut<dim, number> &generic_data_out,
                            dealii::Function<dim>             &reference_function,
                            const std::string                 &norm_name = "norm") const
    {
      const auto &dof_vector  = generic_data_out.get_vector("density");
      const auto &dof_handler = generic_data_out.get_dof_handler("density");

      reference_function.set_time(generic_data_out.get_time());

      std::map<std::string, number> component_errors;

      std::vector<std::string> labels;

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
      dealii::NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler, level_set);
      mesh_classifier.reclassify();

      // compute quadrature
      dealii::NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>> mapping_info_cell(
        generic_data_out.get_mapping(), dealii::update_values | dealii::update_JxW_values);

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

      const auto physical_active_cell_iterators = dof_handler.active_cell_iterators() | physical;

      std::vector<dealii::Quadrature<dim>> quad_vec_cell;

      unsigned int n_active_cells = std::distance(dof_handler.active_cell_iterators().begin(),
                                                  dof_handler.active_cell_iterators().end());

      for (const auto &cell : physical_active_cell_iterators)
        {
          quadrature_generator.generate(cell);
          // active phase has positive level-set
          quad_vec_cell.push_back(quadrature_generator.get_outside_quadrature());
        }

      mapping_info_cell.reinit_cells(physical_active_cell_iterators, quad_vec_cell, n_active_cells);

      // cell-loop for error-norm computation
      dealii::FEPointEvaluation<dim + 2, dim, dim, dealii::VectorizedArray<number>> fe_point_eval(
        mapping_info_cell, fe_point_temp);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_artificial() or cell->is_ghost() or (not physical(cell)))
            continue;

          std::vector<number> solution_values_in(cell->get_fe().dofs_per_cell);

          cell->get_dof_values(dof_vector, solution_values_in.begin(), solution_values_in.end());

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

              for (unsigned int v = 0; v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                   ++v)
                for (unsigned int d = 0; d < 3; ++d)
                  errors_squared[d] += local_errors_squared[d][v];
            }
        }

      dealii::Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);
      std::array<number, 3> errors;
      for (unsigned int d = 0; d < 3; ++d)
        errors[d] = std::sqrt(errors_squared[d]);
      labels                           = {"density", "momentum", "total_energy"};
      component_errors["density"]      = errors[0];
      component_errors["momentum"]     = errors[1];
      component_errors["total_energy"] = errors[2];

      pretty_print_norm("L2 " + norm_name, labels, component_errors);
    }
  };
} // namespace MeltPoolDG::CompressibleFlow
