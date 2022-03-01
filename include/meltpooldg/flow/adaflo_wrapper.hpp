/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/base/function.h>
#  include <deal.II/base/quadrature.h>
#  include <deal.II/base/tensor.h>
#  include <deal.II/base/timer.h>
#  include <deal.II/base/vectorization.h>

#  include <deal.II/dofs/dof_handler.h>

#  include <deal.II/grid/tria.h>

#  include <deal.II/lac/affine_constraints.h>
#  include <deal.II/lac/generic_linear_algebra.h>
#  include <deal.II/lac/la_parallel_vector.h>

#  include <meltpooldg/flow/flow_base.hpp>
#  include <meltpooldg/interface/parameters.hpp>
#  include <meltpooldg/interface/scratch_data.hpp>
#  include <meltpooldg/interface/simulation_base.hpp>
#  include <meltpooldg/material/material.hpp>
#  include <meltpooldg/post_processing/generic_data_out.hpp>
#  include <meltpooldg/utilities/time_iterator.hpp>

#  include <adaflo/navier_stokes.h>
#  include <adaflo/parameters.h>

#  include <functional>
#  include <memory>
#  include <string>
#  include <vector>

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
                  const TimeIterator<double>                             &time_iterator,
                  const bool                                              do_evaporative_mass_flux);

    void
    set_initial_condition(const Function<dim> &initial_field_function_velocity) override;

    void
    reinit_1();

    void
    reinit_2();

    void
    reinit_3();

    void
    init_time_advance() override;

    /**
     * solve time step
     */
    void
    solve() override;

    /**
     * Set phase-dependent densities on faces for face integrals in augmented
     * Taylor-Hood elements with a pressure ansatz space containing elementwise
     * constant functions (element type FE_Q_DG0).
     *
     * This function calculates and assigns phase-dependent densities on faces for
     * use in face integrals of augmented Taylor-Hood elements.
     * The densities may depend on both the level-set field and the temperature
     * field.
     *
     * @param material [in] Material object holding the phase-specific parameters
     *                 necessary for density calculation.
     * @param ls_as_heaviside [in] DoF vector representing the indicator function.
     * @param ls_dof_idx [in] DoF index of the indicator function within the
     *                   matrix-free object.
     * @param temperature [in, opt] DoF vector representing the temperature field.
     * @param temp_dof_idx [in, opt] DoF index of the temperature function within
     *                     the matrix-free object.
     */
    void
    set_face_average_density_augmented_taylor_hood(const Material<double> &material,
                                                   const VectorType       &ls_as_heaviside,
                                                   const unsigned int      ls_dof_idx,
                                                   const VectorType       *temperature  = nullptr,
                                                   unsigned int            temp_dof_idx = -1);

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
    attach_output_vectors(GenericDataOut<dim> &data_out) const override;

    void
    attach_output_vectors_failed_step(GenericDataOut<dim> &data_out) const override;

    void
    set_face_average_density(const typename Triangulation<dim>::cell_iterator &cell,
                             const unsigned int                                face,
                             const double                                      density);

    const Quadrature<dim> &
    get_face_center_quad();

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
    std::unique_ptr<Quadrature<dim>>   face_center_quad;

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
    mutable VectorType force_rhs_velocity_projected;
    mutable VectorType mass_balance_source_term_projected;
    mutable VectorType density;
    mutable VectorType viscosity;

    // determine whether solution vectors are prepared for time advance
    bool ready_for_time_advance = false;
  };
} // namespace MeltPoolDG::Flow
#endif
