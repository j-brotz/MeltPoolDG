#include <meltpooldg/utilities/amr_indicators.templates.hpp>

#define INSTANTIATE(dim, type)                                                                  \
  template class MeltPoolDG::AMR::BinaryOpIndicatorComposite<dim, type>;                        \
  template class MeltPoolDG::AMR::JumpIndicator<dim, type, dim + 2, dim>; /*compressible flow*/ \
  template class MeltPoolDG::AMR::SSEDIndicator<dim, type, dim + 2, dim>; /*compressible flow*/

INSTANTIATE(1, double)
INSTANTIATE(2, double)
INSTANTIATE(3, double)