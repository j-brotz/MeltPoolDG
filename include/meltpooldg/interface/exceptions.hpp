#pragma once
#include <deal.II/base/exceptions.h>

#define AssertThrowZeroTimeIncrement(dt) AssertThrow(dt > 0, MeltPoolDG::ExcZeroTimeIncrement())

namespace MeltPoolDG
{
  DeclException1(ExcBCAlreadyAssigned,
                 std::string,
                 << "You try to attach a " << arg1 << " boundary condition "
                 << "for a boundary_id for which a boundary condition is already "
                 << "specified. Check your input related to boundary conditions!");

  DeclException2(ExcFieldNotAttached,
                 std::string,
                 std::string,
                 << "It seems that you have not called SimulationBase::" << arg1
                 << "() for the operator \"" << arg2 << "\". You can do that, e.g., "
                 << "in your simulation by overriding SimulationBase::set_field_conditions().");
  DeclExceptionMsg(ExcZeroTimeIncrement,
                   "It seems that the time increment is zero. Make sure that "
                   "the time increment is larger than zero.");
  DeclExceptionMsg(ExcNewtonDidNotConverge, "The Newton-Raphson solver did not converge.");
  DeclExceptionMsg(ExcHeatTransferNoConvergence, "The heat transfer solver did not converge.");
} // namespace MeltPoolDG
