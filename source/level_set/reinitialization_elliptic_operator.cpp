#include <meltpooldg/level_set/reinitialization_elliptic_operator.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number, bool is_dg>
  ReinitializationEllipticOperator<dim, number, is_dg>::ReinitializationEllipticOperator(
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
    const ReinitializationData<number>              &reinit_data_in,
    const unsigned int                               reinit_dof_idx_in,
    const unsigned int                               reinit_quad_idx_in,
    const BlockVectorType                           &normal_vector_in)
  {}

  template class ReinitializationEllipticOperator<1, double, true>;
  template class ReinitializationEllipticOperator<2, double, true>;
  template class ReinitializationEllipticOperator<3, double, true>;
  template class ReinitializationEllipticOperator<1, double, false>;
  template class ReinitializationEllipticOperator<2, double, false>;
  template class ReinitializationEllipticOperator<3, double, false>;
} // namespace MeltPoolDG::LevelSet
