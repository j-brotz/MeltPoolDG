#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/core/scratch_data.hpp>
#  include <meltpooldg/core/simulation_base.hpp>
#  include <meltpooldg/level_set/normal_vector_operation_adaflo_wrapper.hpp>
#  include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#  include <meltpooldg/post_processing/generic_data_out.hpp>
#  include <meltpooldg/utilities/conditional_ostream.hpp>
#  include <meltpooldg/utilities/time_iterator.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_compute_normal.h>
#  include <adaflo/level_set_okz_preconditioner.h>
#  include <adaflo/level_set_okz_reinitialization.h>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class ReinitializationOperationAdaflo : public ReinitializationOperationBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    /**
     * Constructor.
     */
    ReinitializationOperationAdaflo(const ScratchData<dim, dim, number> &scratch_data,
                                    const TimeIterator<number>          &time_iterator,
                                    const int                            reinit_dof_idx,
                                    const int                            reinit_quad_idx,
                                    const int                            normal_dof_idx,
                                    const TimeSteppingData<number>      &time_stepping,
                                    const NormalVectorData<number>      &normal_vec_data,
                                    const number       interface_thickness_parameter_value,
                                    const unsigned int n_subdivisions);

    void
    update_dof_idx(const unsigned int &reinit_dof_idx) override;

    void
    reinit() override;

    /**
     * Solver time step
     */
    void
    solve() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() override;

    number
    get_max_change_level_set() const final;

    const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() const override;

    dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() override;

    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    void
    set_initial_condition(const VectorType &level_set_in) override;

    /**
     * Interpolates the initial conditions from a function to the level set field
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function) override;

    number
    compute_CFL_based_timestep() const override
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "CFL based time stepping is not implemented for continous elements!"));
    }

  private:
    void
    set_adaflo_parameters(const TimeSteppingData<number> &time_stepping,
                          const int                       reinit_dof_idx,
                          const int                       reinit_quad_idx,
                          const int                       normal_dof_idx);

    void
    initialize_vectors();

  private:
    void
    create_operator();

    void
    create_normal_vector_operator();

    const ScratchData<dim, dim, number> &scratch_data;

    const TimeIterator<number> &time_iterator;
    /**
     *  advected field
     */
    VectorType level_set;
    /**
     *  vectors for the solution of the linear system
     */
    VectorType increment;
    VectorType rhs;

    /**
     * Adaflo parameters for the level set problem
     */
    adaflo::LevelSetOKZSolverReinitializationParameter reinit_params_adaflo;

    /**
     * Reference to the actual advection diffusion solver from adaflo
     */
    std::shared_ptr<NormalVectorOperationAdaflo<dim, number>>       normal_vector_operation_adaflo;
    std::shared_ptr<adaflo::LevelSetOKZSolverReinitialization<dim>> reinit_operation_adaflo;

    /**
     *  Diagonal preconditioner
     */
    adaflo::DiagonalPreconditioner<number> preconditioner;

    std::pair<number, number>                              last_concentration_range = {-1.0, 1.0};
    dealii::AlignedVector<dealii::VectorizedArray<number>> cell_diameters;
    number                                                 cell_diameter_min;
    number                                                 cell_diameter_max;
    number                                                 epsilon_used;
    bool                                                   first_reinit_step    = true;
    bool                                                   force_compute_normal = true;
    std::function<void(bool)>                              compute_normal;
    const ConditionalOStream                               pcout;

    const NormalVectorData<number> &normal_vector_data;
    const number                    eps_cell_factor;

    // maximum change of the level set due to the current reinitialization step
    number max_change_level_set = std::numeric_limits<number>::max();
  };
} // namespace MeltPoolDG::LevelSet

#endif
