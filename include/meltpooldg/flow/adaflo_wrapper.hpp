/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#  include <meltpooldg/flow/flow_base.hpp>
#  include <meltpooldg/interface/scratch_data.hpp>
#  include <meltpooldg/interface/simulation_base.hpp>
#  include <meltpooldg/utilities/generic_data_out.hpp>
#  include <meltpooldg/utilities/time_iterator.hpp>
#  include <meltpooldg/utilities/utility_functions.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/navier_stokes.h>
#  include <adaflo/parameters.h>

namespace MeltPoolDG::Flow
{
  template <int dim>
  class AdafloWrapper : public FlowBase<dim>
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

  public:
    /**
     * Constructor.
     */
    AdafloWrapper(ScratchData<dim, dim, double, VectorizedArray<double>> &scratch_data,
                  std::shared_ptr<SimulationBase<dim>>                    base_in,
                  const TimeIterator<double> &                            time_iterator,
                  const bool                                              do_evaporative_mass_flux);

    void
    set_initial_condition(const Function<dim> &initial_field_function_velocity);

    void
    reinit_1();

    void
    reinit_2();

    void
    reinit_3();

    void
    init_time_advance() override;

    /**
     * Solver time step
     */
    void
    solve() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_velocity() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity_old() const;

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity_old_old() const;

    const DoFHandler<dim> &
    get_dof_handler_velocity() const override;

    const unsigned int &
    get_dof_handler_idx_velocity() const override;

    const unsigned int &
    get_dof_handler_idx_hanging_nodes_velocity() const override;

    const unsigned int &
    get_quad_idx_velocity() const override;

    const unsigned int &
    get_quad_idx_pressure() const override;

    const AffineConstraints<double> &
    get_constraints_velocity() const override;

    AffineConstraints<double> &
    get_constraints_velocity() override;

    const AffineConstraints<double> &
    get_hanging_node_constraints_velocity() const override;

    const LinearAlgebra::distributed::Vector<double> &
    get_pressure() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_pressure() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_pressure_old() const;

    const LinearAlgebra::distributed::Vector<double> &
    get_pressure_old_old() const;

    const DoFHandler<dim> &
    get_dof_handler_pressure() const override;

    const unsigned int &
    get_dof_handler_idx_pressure() const override;

    const AffineConstraints<double> &
    get_constraints_pressure() const override;

    AffineConstraints<double> &
    get_constraints_pressure() override;

    const AffineConstraints<double> &
    get_hanging_node_constraints_pressure() const override;

    void
    set_force_rhs(const LinearAlgebra::distributed::Vector<double> &vec) override;

    void
    set_mass_balance_rhs(const LinearAlgebra::distributed::Vector<double> &vec) override;

    void
    set_user_defined_material(std::function<Tensor<2, dim, VectorizedArray<double>>(
                                const Tensor<2, dim, VectorizedArray<double>> &,
                                const unsigned int,
                                const unsigned int,
                                const bool)> my_user_defined_material) override;

    VectorizedArray<double> &
    get_density(const unsigned int cell, const unsigned int q) override;

    const VectorizedArray<double> &
    get_density(const unsigned int cell, const unsigned int q) const override;

    VectorizedArray<double> &
    get_viscosity(const unsigned int cell, const unsigned int q) override;

    const VectorizedArray<double> &
    get_viscosity(const unsigned int cell, const unsigned int q) const override;

    VectorizedArray<double> &
    get_damping(const unsigned int cell, const unsigned int q) override;

    const VectorizedArray<double> &
    get_damping(const unsigned int cell, const unsigned int q) const override;

    void
    attach_vectors_u(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    void
    attach_vectors_p(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    void
    distribute_constraints() override;

    void
    attach_output_vectors(GenericDataOut<dim> &data_out);

  private:
    void
    create_parameters(Parameters<double> &parameters, const std::string parameter_file);

    bool
    time_stepping_synchronized();

    ScratchData<dim, dim, double, VectorizedArray<double>> &scratch_data;
    /**
     * Timer
     */
    TimerOutput timer;

    /**
     * Reference to the actual Navier-Stokes solver from adaflo
     */
    std::unique_ptr<NavierStokes<dim>> navier_stokes;

    const FlowParameters &adaflo_params;

    const bool do_evaporative_mass_flux;

    const TimeIterator<double> &time_iterator;

    unsigned int dof_index_u;
    unsigned int dof_index_p;
    unsigned int dof_index_hanging_nodes_u;
    unsigned int dof_index_parameters;

    unsigned int quad_index_u;
    unsigned int quad_index_p;

    DoFHandler<dim>           dof_handler_parameters;
    AffineConstraints<double> constraints_parameters;

    // temporal vectors for output
    VectorType force_rhs_velocity_projected;
    VectorType mass_balance_source_term_projected;
    VectorType density;
    VectorType viscosity;
  };
} // namespace MeltPoolDG::Flow
#endif
