/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, June 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/evaporation/evaporation_model_base.hpp>
#include <meltpooldg/utilities/numbers.hpp>

namespace MeltPoolDG::Evaporation
{
  class EvaporationModelConstant : public EvaporationModelBase
  {
  private:
    // function to compute the evaporative mass flux at a given time
    mutable FunctionParser<1> m_dot;

  public:
    EvaporationModelConstant(const std::string &expr_evaporative_mass_flux)
      : m_dot(expr_evaporative_mass_flux)
    {}

    double
    local_compute_evaporative_mass_flux(const double time /*here used as time*/) const final
    {
      AssertThrow(!numbers::is_invalid(time),
                  ExcMessage("Time must be set to compute the evaporative mass flux."));
      m_dot.set_time(time);
      return m_dot.value(Point<1>() /* dummy value*/);
    }
  };
} // namespace MeltPoolDG::Evaporation
