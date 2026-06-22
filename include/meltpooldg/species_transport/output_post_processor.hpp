#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <string>
#include <vector>

namespace MeltPoolDG::SpeciesTransport
{
  /**
   * Post-processor for computing species mass fractions. The computation of these variables is
   * defined by the provided view and might involve computations based on the species mass fractions
   * in the solution vector. The post-processor can directly interact with the deal.II
   * data post-processing framework and can be attached to a GenericDataOut object for output
   * generation.
   */
  template <int dim, int n_species, typename number, typename View>
    requires requires { typename View::state_type; }
  class MassFractionPostProcessor : public dealii::DataPostprocessor<dim>
  {
  public:
    /**
     * Constructor for the species mass fractions post-processor.
     *
     * @param view_creator A function that defines how to construct the view for computing the species
     * mass fractions based on the solution values at a specified node. The function should take a
     * reference to the state type defined by the view and return an instance of the view.
     * @param species_names A vector of strings containing the names of the species for output
     * labeling. The number of names provided should match the number of species in the simulation.
     * The names will be postfixed with " mass fraction" to indicate the nature of the output
     * variable.
     */
    MassFractionPostProcessor(const std::function<View(typename View::state_type &)> &view_creator,
                              const std::vector<std::string>                         &species_names)
      : view_creator(view_creator)
      , species_names(species_names)
    {
      Assert(
        species_names.size() == n_species,
        dealii::ExcMessage(
          "The number of provided species names must match the number of species in the simulation."));
      for (std::string &name : this->species_names)
        {
          name += "_mass_fraction";
        }
    }

    std::vector<std::string>
    get_names() const override
    {
      return species_names;
    }

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      return std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>(
        n_species, dealii::DataComponentInterpretation::component_is_scalar);
    }

    dealii::UpdateFlags
    get_needed_update_flags() const override
    {
      return dealii::update_values;
    }

    /**
     * Based on the provided solution values, get the mass fractions and store them in the
     * computed_quantities vector. The mapping from solution values to solution variables is
     * defined by the provided view.
     *
     * @param inputs The input data containing solution values at the evaluation points.
     * @param computed_quantities The vector where the computed mass fractions will be stored.
     * Each entry corresponds to an evaluation point and should be filled with the mass fractions
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

          for (int i = 0; i < n_species; ++i)
            {
              computed_quantities[p](i) = accessor.mass_fraction(i);
            }
        }
    }

    /// Function defining how to construct the view for computing the mass fractions based on
    /// the solution values at a specified node.
    const std::function<View(typename View::state_type &)> view_creator;

    /// Names of the species for output labeling.
    std::vector<std::string> species_names;
  };

  /**
   * Post-processor for computing species partial densities. The computation of these variables is
   * defined by the provided view and might involve computations based on the species partial
   * densities in the solution vector. The post-processor can directly interact with the deal.II
   * data post-processing framework and can be attached to a GenericDataOut object for output
   * generation.
   */
  template <int dim, int n_species, typename number, typename View>
    requires requires { typename View::state_type; }
  class PartialDensityPostProcessor : public dealii::DataPostprocessor<dim>
  {
  public:
    /**
     * Constructor for the species partial densities post-processor.
     *
     * @param view_creator A function that defines how to construct the view for computing the species
     * partial densities based on the solution values at a specified node. The function should take
     * a reference to the state type defined by the view and return an instance of the view.
     * @param species_names A vector of strings containing the names of the species for output
     * labeling. The number of names provided should match the number of species in the simulation.
     * The names will be postfixed with " partial density" to indicate the nature of the output
     * variable.
     */
    PartialDensityPostProcessor(
      const std::function<View(typename View::state_type &)> &view_creator,
      const std::vector<std::string>                         &species_names)
      : view_creator(view_creator)
      , species_names(species_names)
    {
      Assert(
        species_names.size() == n_species,
        dealii::ExcMessage(
          "The number of provided species names must match the number of species in the simulation."));
      for (std::string &name : this->species_names)
        {
          name += "_partial_density";
        }
    }

    std::vector<std::string>
    get_names() const override
    {
      return species_names;
    }

    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override
    {
      return std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>(
        n_species, dealii::DataComponentInterpretation::component_is_scalar);
    }

    dealii::UpdateFlags
    get_needed_update_flags() const override
    {
      return dealii::update_values;
    }

    /**
     * Based on the provided solution values, get the partial densities and store them in the
     * computed_quantities vector. The mapping from solution values to solution variables is
     * defined by the provided view.
     *
     * @param inputs The input data containing solution values at the evaluation points.
     * @param computed_quantities The vector where the computed partial densities will be stored.
     * Each entry corresponds to an evaluation point and should be filled with the partial densities
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

          for (int i = 0; i < n_species; ++i)
            {
              computed_quantities[p](i) = accessor.partial_density(i);
            }
        }
    }

    /// Function defining how to construct the view for computing the conserved variables based on
    /// the solution values at a specified node.
    const std::function<View(typename View::state_type &)> view_creator;

    /// Names of the species for output labeling.
    std::vector<std::string> species_names;
  };
} // namespace MeltPoolDG::SpeciesTransport
