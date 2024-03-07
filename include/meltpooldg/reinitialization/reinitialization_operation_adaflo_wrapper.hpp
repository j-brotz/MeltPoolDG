/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/interface/scratch_data.hpp>
#  include <meltpooldg/interface/simulation_base.hpp>
#  include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>
#  include <meltpooldg/post_processing/generic_data_out.hpp>
#  include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#  include <meltpooldg/utilities/conditional_ostream.hpp>
#  include <meltpooldg/utilities/time_iterator.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_compute_normal.h>
#  include <adaflo/level_set_okz_preconditioner.h>
#  include <adaflo/level_set_okz_reinitialization.h>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    template <int dim>
    class ReinitializationOperationAdaflo : public ReinitializationOperationBase<dim>
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      /**
       * Constructor.
       */
      ReinitializationOperationAdaflo(const ScratchData<dim>     &scratch_data,
                                      const TimeIterator<double> &time_iterator,
                                      const int                   reinit_dof_idx,
                                      const int                   reinit_quad_idx,
                                      const int                   normal_dof_idx,
                                      const Parameters<double>   &parameters);

      void
      update_dof_idx(const unsigned int &reinit_dof_idx) override;

      void
      reinit() override;

      /**
       * Solver time step
       */
      void
      solve() override;

      const LinearAlgebra::distributed::Vector<double> &
      get_level_set() const override;

      LinearAlgebra::distributed::Vector<double> &
      get_level_set() override;

      double
      get_max_change_level_set() const final;

      const LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() const override;

      LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() override;

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

      void
      attach_output_vectors(GenericDataOut<dim> &data_out) const override;

      void
      set_initial_condition(const VectorType &level_set_in) override;

    private:
      void
      set_adaflo_parameters(const Parameters<double> &parameters,
                            const int                 reinit_dof_idx,
                            const int                 reinit_quad_idx,
                            const int                 normal_dof_idx);

      void
      initialize_vectors();

    private:
      void
      create_operator();

      void
      create_normal_vector_operator();

      const ScratchData<dim> &scratch_data;

      const TimeIterator<double> &time_iterator;
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
      LevelSetOKZSolverReinitializationParameter reinit_params_adaflo;

      /**
       * Reference to the actual advection diffusion solver from adaflo
       */
      std::shared_ptr<NormalVectorOperationAdaflo<dim>>       normal_vector_operation_adaflo;
      std::shared_ptr<LevelSetOKZSolverReinitialization<dim>> reinit_operation_adaflo;

      /**
       *  Diagonal preconditioner
       */
      DiagonalPreconditioner<double> preconditioner;

      std::pair<double, double>              last_concentration_range = {-1.0, 1.0};
      AlignedVector<VectorizedArray<double>> cell_diameters;
      double                                 cell_diameter_min;
      double                                 cell_diameter_max;
      double                                 epsilon_used;
      bool                                   first_reinit_step    = true;
      bool                                   force_compute_normal = true;
      std::function<void(bool)>              compute_normal;
      const ConditionalOStream               pcout;

      const NormalVectorData<double> &normal_vector_data;
      const double                    eps_cell_factor;

      // maximum change of the level set due to the current reinitialization step
      double max_change_level_set = std::numeric_limits<double>::max();
    };
  } // namespace LevelSet
} // namespace MeltPoolDG

#endif
