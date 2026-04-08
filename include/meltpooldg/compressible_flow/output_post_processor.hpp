#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/operation_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <string>
#include <vector>

namespace MeltPoolDG::CompressibleFlow
{
  /**
   * Post-processor for computing conserved variables. The computation of these variables is defined
   * by the provided view and might involve computations based on the conserved variables in the
   * solution vector. The post-processor can directly interact with the deal.II data post-processing
   * framework and can be attached to a GenericDataOut object for output generation.
   */
  template <int dim, typename number, IsConservedView View>
    requires requires { typename View::state_type; }
  class ConservedVariablesPostProcessor : public dealii::DataPostprocessor<dim>
  {
  public:
    /**
     * Constructor for the conserved variables post-processor.
     *
     * @param view_creator A function that defines how to construct the view for computing the conserved
     * variables based on the solution values at a specified node. The function should take a
     * reference to the state type defined by the view and return an instance of the view.
     */
    ConservedVariablesPostProcessor(
      const std::function<View(typename View::state_type &)> &view_creator)
      : view_creator(view_creator)
    {}

    std::vector<std::string>
    get_names() const override
    {
      std::vector<std::string> names;
      names.reserve(n_conserved_variables<dim>);
      names.emplace_back("density");
      names.insert(names.end(), dim, "momentum");
      names.emplace_back("total energy");
      return names;
    }

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> interpretations;
      interpretations.reserve(n_conserved_variables<dim>);
      interpretations.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretations.insert(interpretations.end(),
                             dim,
                             dealii::DataComponentInterpretation::component_is_part_of_vector);
      interpretations.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      return interpretations;
    }

    dealii::UpdateFlags
    get_needed_update_flags() const override
    {
      return dealii::update_values;
    }

    /**
     * Based on the provided solution values, compute or better read the conserved variables and
     * store them in the computed_quantities vector. The mapping from solution values to conserved
     * variables is defined by the provided view.
     *
     * @param inputs The input data containing solution values at the evaluation points.
     * @param computed_quantities The vector where the computed conserved variables will be stored.
     * Each entry corresponds to an evaluation point and should be filled with the conserved
     * variable values in the order defined by get_names().
     */
    void
    evaluate_vector_field(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
                          std::vector<dealii::Vector<number>> &computed_quantities) const override
    {
      const unsigned int n_evaluation_points = inputs.solution_values.size();

      for (unsigned int p = 0; p < n_evaluation_points; ++p)
        {
          typename View::state_type w(dealii::make_array_view(inputs.solution_values[p]));
          const View                accessor = view_creator(w);

          computed_quantities[p](ConservedVariableIndex<dim>::density) = accessor.density();
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p](ConservedVariableIndex<dim>::momentum + d) =
              accessor.momentum(d);
          computed_quantities[p](ConservedVariableIndex<dim>::energy) = accessor.total_energy();
        }
    }

    /// Function defining how to construct the view for computing the conserved variables based on
    /// the solution values at a specified node.
    const std::function<View(typename View::state_type &)> view_creator;
  };

  /**
   * Post-processor for computing primitive variables. The computation of these variables is defined
   * by the provided view and might involve computations based on the conserved variables in the
   * solution vector. The post-processor can directly interact with the deal.II data post-processing
   * framework and can be attached to a GenericDataOut object for output generation.
   */
  template <int dim, typename number, IsPrimitiveView View>
    requires requires { typename View::state_type; }
  class PrimitiveVariablesPostProcessor : public dealii::DataPostprocessor<dim>
  {
    constexpr static unsigned int n_primitive_variables = dim + 2;

    enum class PrimitiveVariablesIndex : unsigned int
    {
      velocity = 0,
      pressure = dim,
      temperature
    };

  public:
    /**
     * Constructor for the primitive variables post-processor.
     *
     * @param view_creator A function that defines how to construct the view for computing the primitive
     * variables based on the solution values at a specified node. The function should take a
     * reference to the state type defined by the view and return an instance of the view.
     */
    PrimitiveVariablesPostProcessor(
      const std::function<View(typename View::state_type &)> &view_creator)
      : view_creator(view_creator)
    {}

    std::vector<std::string>
    get_names() const override
    {
      std::vector<std::string> names;
      names.reserve(n_primitive_variables);
      names.insert(names.end(), dim, "velocity");
      names.emplace_back("pressure");
      names.emplace_back("temperature");
      return names;
    }

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> interpretations;
      interpretations.reserve(n_primitive_variables);
      interpretations.insert(interpretations.end(),
                             dim,
                             dealii::DataComponentInterpretation::component_is_part_of_vector);
      interpretations.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      interpretations.push_back(dealii::DataComponentInterpretation::component_is_scalar);
      return interpretations;
    }

    dealii::UpdateFlags
    get_needed_update_flags() const override
    {
      return dealii::update_values;
    }

    /**
     * Based on the provided solution values, compute the primitive variables and store them in
     * the computed_quantities vector. The computation of the primitive variables is defined by the
     * provided view.
     *
     * @param inputs The input data containing solution values at the evaluation points.
     * @param computed_quantities The vector where the computed primitive variables will be stored.
     * Each entry corresponds to an evaluation point and should be filled with the primitive
     * variable values in the order defined by get_names().
     */
    void
    evaluate_vector_field(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
                          std::vector<dealii::Vector<number>> &computed_quantities) const override
    {
      const unsigned int n_evaluation_points = inputs.solution_values.size();

      for (unsigned int p = 0; p < n_evaluation_points; ++p)
        {
          typename View::state_type w(dealii::make_array_view(inputs.solution_values[p]));
          const View                accessor = view_creator(w);

          computed_quantities[p](static_cast<unsigned int>(PrimitiveVariablesIndex::pressure)) =
            accessor.pressure();
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p](static_cast<unsigned int>(PrimitiveVariablesIndex::velocity) +
                                   d) = accessor.velocity(d);
          computed_quantities[p](static_cast<unsigned int>(PrimitiveVariablesIndex::temperature)) =
            accessor.temperature();
        }
    }

    /// Function defining how to construct the view for computing the primitive variables based on
    /// the solution values at a specified node.
    const std::function<View(typename View::state_type &)> view_creator;
  };

  /**
   * Post-processor for computing material variables such as dynamic viscosity, specific gas
   * constant, heat capacity ratio, specific isobaric heat, and thermal conductivity. The
   * computation of these variables is defined by the provided view and might involve computations
   * based on the conserved variables in the solution vector. The post-processor can directly
   * interact with the deal.II data post-processing framework and can be attached to a
   * GenericDataOut object for output generation.
   */
  template <int dim, typename number, IsMaterialView View>
    requires requires { typename View::state_type; }
  class MaterialVariablesPostProcessor : public dealii::DataPostprocessor<dim>
  {
    constexpr static unsigned int n_material_variables = 5;

    enum class MaterialVariablesIndex : unsigned int
    {
      dynamic_viscosity = 0,
      specific_gas_constant,
      heat_capacity_ratio,
      heat_capacity_at_constant_pressure,
      thermal_conductivity
    };

  public:
    /**
     * Constructor for the material variables post-processor.
     *
     * @param view_creator A function that defines how to construct the view for computing the material
     * variables based on the solution values at a specified node. The function should take a
     * reference to the state type defined by the view and return an instance of the view.
     */
    MaterialVariablesPostProcessor(
      const std::function<View(typename View::state_type &)> &view_creator)
      : view_creator(view_creator)
    {}

    std::vector<std::string>
    get_names() const override
    {
      std::vector<std::string> names;
      names.reserve(n_material_variables);
      names.emplace_back("dynamic_viscosity");
      names.emplace_back("specific_gas_constant");
      names.emplace_back("heat_capacity_ratio");
      names.emplace_back("specific_isobaric_heat");
      names.emplace_back("thermal_conductivity");
      return names;
    }

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> interpretations(
        n_material_variables, dealii::DataComponentInterpretation::component_is_scalar);
      return interpretations;
    }

    dealii::UpdateFlags
    get_needed_update_flags() const override
    {
      return dealii::update_values;
    }

    /**
     * Based on the provided solution values, compute the material variables and store them in
     * the computed_quantities vector. The computation of the material variables is defined by the
     * provided view.
     *
     * @param inputs The input data containing solution values at the evaluation points.
     * @param computed_quantities The vector where the computed material variables will be stored.
     * Each entry corresponds to an evaluation point and should be filled with the material
     * variable values in the order defined by get_names().
     */
    void
    evaluate_vector_field(const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
                          std::vector<dealii::Vector<number>> &computed_quantities) const override
    {
      const unsigned int n_evaluation_points = inputs.solution_values.size();

      for (unsigned int p = 0; p < n_evaluation_points; ++p)
        {
          typename View::state_type w(dealii::make_array_view(inputs.solution_values[p]));
          const View                accessor = view_creator(w);

          computed_quantities[p](static_cast<unsigned int>(
            MaterialVariablesIndex::dynamic_viscosity))     = accessor.dynamic_viscosity();
          computed_quantities[p](static_cast<unsigned int>(
            MaterialVariablesIndex::specific_gas_constant)) = accessor.specific_gas_constant();
          computed_quantities[p](static_cast<unsigned int>(
            MaterialVariablesIndex::heat_capacity_ratio))   = accessor.heat_capacity_ratio();
          computed_quantities[p](
            static_cast<unsigned int>(MaterialVariablesIndex::heat_capacity_at_constant_pressure)) =
            accessor.specific_isobaric_heat();
          computed_quantities[p](static_cast<unsigned int>(
            MaterialVariablesIndex::thermal_conductivity)) = accessor.thermal_conductivity();
        }
    }

    /// Function defining how to construct the view for computing the material variables based on
    /// the solution values at a specified node.
    const std::function<View(typename View::state_type &)> view_creator;
  };

  /**
   * A convenient output manager that can be used to attach post-processors for conserved variables,
   * primitive variables and material variables to a GenericDataOut object. Its main purpose is to
   * provide a single point of interaction for post-processors and decide which post-processor to
   * attach based on the provided output options.
   */
  template <int dim, typename number>
  struct OutputManager
  {
    /**
     * Attach the relevant post-processors to the provided GenericDataOut object based on the
     * provided output options. The post-processors are attached with the provided DoFHandler and
     * solution vector.
     *
     * @param data_out The GenericDataOut object to which the post-processors will be attached.
     * @param dof_handler The DoFHandler associated with the solution vector.
     * @param solution The solution vector containing the conserved variables.
     * @param output_types The list of output options that defines which post-processors to attach
     */
    void
    attach_to_data_out(GenericDataOut<dim, number>                              &data_out,
                       const dealii::DoFHandler<dim>                            &dof_handler,
                       const dealii::LinearAlgebra::distributed::Vector<number> &solution,
                       const std::vector<OutputType>                            &output_types) const
    {
      if (Utils::contains(output_types, OutputType::conserved_variables))
        {
          for (const auto &post_processor : conserved_post_processors)
            {
              data_out.add_data_vector(&dof_handler, &solution, post_processor.get());
            }
        }
      if (Utils::contains(output_types, OutputType::primitive_variables))
        {
          for (const auto &post_processor : primitive_post_processors)
            {
              data_out.add_data_vector(&dof_handler, &solution, post_processor.get());
            }
        }
      if (Utils::contains(output_types, OutputType::material_quantities))
        {
          for (const auto &post_processor : material_post_processors)
            {
              data_out.add_data_vector(&dof_handler, &solution, post_processor.get());
            }
        }
    }

    /**
     * Add a post-processor for conserved variables to the output manager. The post-processor will
     * be attached to the GenericDataOut object in attach_to_data_out() if the corresponding output
     * option is provided.
     */
    void
    add_conserved_variables_post_processor(
      std::unique_ptr<dealii::DataPostprocessor<dim>> &&post_processor)
    {
      conserved_post_processors.emplace_back(std::move(post_processor));
    }

    /**
     * Add a post-processor for primitive variables to the output manager. The post-processor will
     * be attached to the GenericDataOut object in attach_to_data_out() if the corresponding output
     * option is provided.
     */
    void
    add_primitive_variables_post_processor(
      std::unique_ptr<dealii::DataPostprocessor<dim>> &&post_processor)
    {
      primitive_post_processors.emplace_back(std::move(post_processor));
    }

    /**
     * Add a post-processor for material variables to the output manager. The post-processor will
     * be attached to the GenericDataOut object in attach_to_data_out() if the corresponding output
     * option is provided.
     */
    void
    add_material_quantities_post_processor(
      std::unique_ptr<dealii::DataPostprocessor<dim>> &&post_processor)
    {
      material_post_processors.emplace_back(std::move(post_processor));
    }

  private:
    std::vector<std::unique_ptr<dealii::DataPostprocessor<dim>>> conserved_post_processors;
    std::vector<std::unique_ptr<dealii::DataPostprocessor<dim>>> primitive_post_processors;
    std::vector<std::unique_ptr<dealii::DataPostprocessor<dim>>> material_post_processors;
  };
} // namespace MeltPoolDG::CompressibleFlow