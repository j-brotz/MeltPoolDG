#pragma once

#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/reinitialization/reinitialization_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

/**
 * Class for the diffusive stabilization of the levelset reinizilization
 */
namespace MeltPoolDG::LevelSet
{
  template <int dim, typename Number = double>
  class ReinitializationDGDiffusionOperator
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

    ReinitializationDGDiffusionOperator(const MeltPoolDG::ScratchData<dim> &scratch_datain,
                          const ReinitializationData<Number> &reinit_data_in,
                          const unsigned int                  reinit_dof_idx_in,
                          const unsigned int                  reinit_quad_idx_in);

    /**
     * Computes the necessary amount of diffusion
     */
    void
    compute_viscosity_value();

    /**
     * Computes the necessary penalty parameter
     */
    void
    compute_penalty_parameter();

    /**
     * If an analytical function for a field is provided and an analytical update is
     * enabled, this function sets the field according to the anylytical function. This function
     * implementation is needed for a time integrator. This function is currently unused
     * @param time
     */
    void
    set_field_functions([[maybe_unused]] const Number time) const {};

    /** 
     * Applies the DG diffusion operator to the src vector and stores the result in the dst vector.
     * The dst vector is zeroed out before the operation.
     * @param time
     * @param dst destination vecor where the diffusive part is stored
     * @param src source vector for the operator
     */
    void
    apply_operator([[maybe_unused]] const Number time,
                   VectorType                   &dst,
                   const VectorType             &src) const;

    /**
     *  Applies the dirichlet contribution of the DG diffusion operator to the src vector and stores
     * the result in the dst vector. The dst vector is NOT zeroed out before the operation.
     * Is not implemented since for a levelset reinilization, dirichlet boundaries don't make much
     * sense.
     * */
    void
    apply_dirichlet_boundary_operator([[maybe_unused]] const Number      time,
                                      [[maybe_unused]] VectorType       &dst,
                                      [[maybe_unused]] const VectorType &src) const {};

    double
    get_viscosity() const;

    /**
     *Variable is needed for time integration
     */
    bool update_field_functions = false;

  private:
    const MeltPoolDG::ScratchData<dim> &scratch_data;
    const ReinitializationData<Number> &reinit_data;

    const unsigned int reinit_dof_idx;
    const unsigned int reinit_quad_idx;

    mutable Number                                 viscosity = 1.;
    mutable AlignedVector<VectorizedArray<Number>> array_penalty_parameter;


    /**
     * Applies the domain integral
     * * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_domain(const MatrixFree<dim, Number>               &data,
                                    VectorType                                  &dst,
                                    const VectorType                            &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the inner face integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_inner_face(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the boundary face integral
     * * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number>               &data,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;
  };
} // namespace MeltPoolDG::LevelSet