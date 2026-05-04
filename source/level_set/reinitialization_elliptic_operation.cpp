#include <meltpooldg/level_set/reinitialization_elliptic_operation.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  ReinitializationEllipticOperation<dim, number>::ReinitializationEllipticOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const ReinitializationData<number>  &reinit_data,
    const unsigned int                   reinit_dof_idx_in,
    const unsigned int                   reinit_quad_idx_in,
    const unsigned int                   ls_dof_idx_in,
    const NormalVectorData<number>      &normal_vec_data)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
  {
    // TODO: handle DG vs. CG
    normal_vector_operation = std::make_shared<NormalVectorDGOperation<dim, number>>(
      scratch_data_in, reinit_dof_idx, reinit_quad_idx, solution_level_set, normal_vec_data);

    reinit_operator = std::make_unique<ReinitializationEllipticOperator<dim, number>>(
      scratch_data_in,
      reinit_data,
      reinit_dof_idx_in,
      reinit_quad_idx_in,
      normal_vector_operation->get_solution_normal_vector(),
      reinit_data.fe.type == FiniteElementType::FE_DGQ);
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::solve()
  {
    // TODO
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::reinit()
  {
    // TODO
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::update_dof_idx(
    const unsigned int &reinit_dof_idx_in)
  {
    reinit_dof_idx = reinit_dof_idx_in;
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::set_initial_condition(
    const VectorType & /*solution_level_set_in*/)
  {
    // TODO
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::set_initial_condition(
    const Function<dim> & /*initial_field_function*/)
  {
    // TODO
  }

  template <int dim, typename number>
  const typename ReinitializationEllipticOperation<dim, number>::VectorType &
  ReinitializationEllipticOperation<dim, number>::get_level_set() const
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  typename ReinitializationEllipticOperation<dim, number>::VectorType &
  ReinitializationEllipticOperation<dim, number>::get_level_set()
  {
    return solution_level_set;
  }

  template <int dim, typename number>
  number
  ReinitializationEllipticOperation<dim, number>::get_max_change_level_set() const
  {
    return max_change_level_set;
  }

  template <int dim, typename number>
  const typename ReinitializationEllipticOperation<dim, number>::BlockVectorType &
  ReinitializationEllipticOperation<dim, number>::get_normal_vector() const
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim, typename number>
  typename ReinitializationEllipticOperation<dim, number>::BlockVectorType &
  ReinitializationEllipticOperation<dim, number>::get_normal_vector()
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<number> *> &)
  {
    //  no need to transfer vectors during AMR
  }

  template <int dim, typename number>
  void
  ReinitializationEllipticOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                             solution_level_set,
                             "level_set");

    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));
  }

  template class ReinitializationEllipticOperation<1, double>;
  template class ReinitializationEllipticOperation<2, double>;
  template class ReinitializationEllipticOperation<3, double>;
} // namespace MeltPoolDG::LevelSet
