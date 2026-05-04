#include <meltpooldg/level_set/reinitialization_elliptic_operator.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationEllipticOperator<dim, number>::ReinitializationEllipticOperator(
    const MeltPoolDG::ScratchData<dim, dim, number> & /*scratch_data_in*/,
    const ReinitializationData<number> & /*reinit_data_in*/,
    const unsigned int /*reinit_dof_idx_in*/,
    const unsigned int /*reinit_quad_idx_in*/,
    const BlockVectorType & /*normal_vector_in*/,
    const bool is_dg_in)
    : is_dg(is_dg_in)
  {}

  template class ReinitializationEllipticOperator<1, double>;
  template class ReinitializationEllipticOperator<2, double>;
  template class ReinitializationEllipticOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
