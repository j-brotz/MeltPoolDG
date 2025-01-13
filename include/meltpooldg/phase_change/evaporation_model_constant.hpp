/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, June 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>

#include <meltpooldg/phase_change/evaporation_model_base.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>

namespace MeltPoolDG::Evaporation
{
  class EvaporationModelConstant : public EvaporationModelBase
  {
  private:
    // function to compute the evaporative mass flux at a given time
    mutable dealii::FunctionParser<1> m_dot;

  public:
    EvaporationModelConstant(const std::string &expr_evaporative_mass_flux)
      : m_dot(expr_evaporative_mass_flux)
    {}

    double
    local_compute_evaporative_mass_flux(const double time /*here used as time*/) const final
    {
      AssertThrow(!dealii::numbers::is_invalid(time),
                  dealii::ExcMessage("Time must be set to compute the evaporative mass flux."));
      m_dot.set_time(time);
      return m_dot.value(dealii::Point<1>() /* dummy value*/);
    }
  };
} // namespace MeltPoolDG::Evaporation
