#include <deal.II/base/function_parser.h>

#include <meltpooldg/compressible_flow/boundary_condition_functions.hpp>
#include <meltpooldg/compressible_flow/boundary_conditions.hpp>

namespace MeltPoolDG::CompressibleFlow
{
  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number                                         initial_time,
    const std::shared_ptr<dealii::Function<dim, number>> density,
    const std::shared_ptr<dealii::Function<dim, number>> velocity,
    const std::shared_ptr<dealii::Function<dim, number>> inner_energy,
    const std::shared_ptr<dealii::Function<dim, number>> mass_fractions)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + mass_fractions->n_components,
                                    initial_time)
    , density(density)
    , velocity(velocity)
    , inner_energy(inner_energy)
    , mass_fractions(mass_fractions)
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number       initial_time,
    const std::string &density_function_string,
    const std::string &velocity_function_string,
    const std::string &inner_energy_function_string,
    const std::string &mass_fraction_functions_string,
    const unsigned     n_mass_fractions)
    : dealii::Function<dim, number>(n_conserved_variables<dim> + n_mass_fractions, initial_time)
    , density(std::make_shared<dealii::FunctionParser<dim>>(density_function_string))
    , velocity(std::make_shared<dealii::FunctionParser<dim>>(velocity_function_string))
    , inner_energy(std::make_shared<dealii::FunctionParser<dim>>(inner_energy_function_string))
    , mass_fractions(n_mass_fractions == 0 ? nullptr :
                                             std::make_shared<dealii::FunctionParser<dim>>(
                                               mass_fraction_functions_string))
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number                                         initial_time,
    const std::shared_ptr<dealii::Function<dim, number>> density,
    const std::shared_ptr<dealii::Function<dim, number>> velocity,
    const std::shared_ptr<dealii::Function<dim, number>> inner_energy)
    : dealii::Function<dim, number>(n_conserved_variables<dim>, initial_time)
    , density(density)
    , velocity(velocity)
    , inner_energy(inner_energy)
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  ConservativeVariablesFunction<dim, number>::ConservativeVariablesFunction(
    const number       initial_time,
    const std::string &density_function_string,
    const std::string &velocity_function_string,
    const std::string &inner_energy_function_string)
    : dealii::Function<dim, number>(n_conserved_variables<dim>, initial_time)
    , density(std::make_shared<dealii::FunctionParser<dim>>(density_function_string))
    , velocity(std::make_shared<dealii::FunctionParser<dim>>(velocity_function_string))
    , inner_energy(std::make_shared<dealii::FunctionParser<dim>>(inner_energy_function_string))
  {
    set_time(initial_time);
  }

  template <int dim, typename number>
  void
  ConservativeVariablesFunction<dim, number>::set_time(const number new_time)
  {
    dealii::Function<dim>::set_time(new_time);
    density->set_time(new_time);
    velocity->set_time(new_time);
    inner_energy->set_time(new_time);

    // Only set time for species function if it exists, since this function can also be used for
    // single-species flow cases where no species function is provided.
    if (mass_fractions)
      mass_fractions->set_time(new_time);
  }

  template <int dim, typename number>
  number
  ConservativeVariablesFunction<dim, number>::value(const dealii::Point<dim, number> &loc,
                                                    const unsigned int component) const
  {
    const auto ramp_up_scaling = [&](const RampUpParameters param) -> number {
      if (this->get_time() > param.duration or param.duration == 0. or
          param.type == RampUpType::none)
        return 1.;

      switch (param.type)
        {
          case RampUpType::linear:
            return this->get_time() / param.duration;
            break;
          case RampUpType::exponential:
            return (std::exp(this->get_time() / param.duration) - 1) / (std::numbers::e - 1);
            break;
          case RampUpType::cosine:
            return 0.5 * (1 - std::cos(std::numbers::pi * this->get_time() / param.duration));
            break;
          default:
            AssertThrow(false, dealii::ExcInternalError());
        }
    };

    const auto compute_conserved_energy = [&](const dealii::Point<dim, number> &loc) -> number {
      dealii::Vector<number> velocity_vec(dim);
      velocity->vector_value(loc, velocity_vec);
      velocity_vec *= ramp_up_scaling(velocity_ramp_up);
      return density->value(loc, 0) * (inner_energy->value(loc, 0) + 0.5 * velocity_vec.norm_sqr());
    };

    if (component == 0)
      return density->value(loc, 0);
    else if (component > 0 and component <= dim)
      return density->value(loc, 0) * velocity->value(loc, component - 1) *
             ramp_up_scaling(velocity_ramp_up);
    else if (component == dim + 1)
      return compute_conserved_energy(loc);
    else if (component < this->n_components)
      return density->value(loc, 0) * mass_fractions->value(loc, component - dim - 2);
    else
      AssertThrow(false,
                  dealii::ExcMessage("Invalid component for conservative variable function: " +
                                     std::to_string(component) +
                                     ". Only components from within the range [0, " +
                                     std::to_string(this->n_components) + ") are valid."));
  }
} // namespace MeltPoolDG::CompressibleFlow

template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<1, double>;
template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<2, double>;
template class MeltPoolDG::CompressibleFlow::ConservativeVariablesFunction<3, double>;