#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <functional>
#include <iostream>

namespace MeltPoolDG::CutUtil
{
  template <int dim, typename Number>
  class SolutionTransferOperator
  {
  private:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  public:
    SolutionTransferOperator(const Number gamma_degree_0,
                             const Number gamma_degree_1,
                             const Number gamma_degree_2,
                             const bool   is_two_phase,
                             const int    verbosity = 0);

    /**
     * Reinit functions for the solution transfer according to the new immersed
     * interface position. New DoFs are ghost-penalty extrapolated.
     */
    void
    reinit(dealii::DoFHandler<dim>                        &dof_handler,
           dealii::Triangulation<dim>                     &tria,
           const VectorType                               &solution,
           const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old,
           const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
           const std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> &reinit_vector,
           const std::function<void(const dealii::DoFHandler<dim> &)> &reinit_matrix_free = {});

    /**
     * Getter functions for the transferred solution.
     */
    const VectorType &
    get_updated_solution() const
    {
      return new_solution;
    };

    VectorType &
    get_updated_solution()
    {
      return new_solution;
    };

  private:
    /**
     * Update FE-index and distribute DoFs according to new state with the moved interface.
     */
    void
    transfer_solution_constant_dofs(
      dealii::DoFHandler<dim>                                                  &dof_handler,
      dealii::Triangulation<dim>                                               &tria,
      const VectorType                                                         &solution,
      const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier_old,
      const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier,
      const std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> &reinit_vector,
      const std::function<void(const dealii::DoFHandler<dim> &)>               &reinit_matrix_free);

    /**
     * Mark the unknown DoFs, which need to be ghost-penalty extrapolated.
     */
    VectorType
    mark_dofs_for_gp_extrapolation(
      const dealii::DoFHandler<dim>                  &dof_handler,
      const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
      const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old) const;

    /**
     * Create constraints for the known DoFs, which not need to be ghost-penalty extrapolated.
     */
    dealii::AffineConstraints<Number>
    create_constraints_gp_extrapolation(const dealii::DoFHandler<dim> &dof_handler,
                                        const VectorType &flags_dofs_gp_extrapolation) const;

    /**
     * Assemble and solve system for ghost-penalty extrapolation, apply constraints.
     */
    void
    extrapolate_solution_new_dofs(
      const dealii::DoFHandler<dim>                                            &dof_handler,
      const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier,
      const dealii::NonMatching::MeshClassifier<dim>                           &mesh_classifier_old,
      const std::function<void(VectorType &, const dealii::DoFHandler<dim> &)> &reinit_vector);

    VectorType new_solution;

    unsigned int fe_degree;
    // components of the solution in one phase
    // (for scalar-valued solutions: 1, for vector-valued solutions: >1)
    unsigned int n_components_per_phase;
    bool         is_dg;
    const Number gamma_degree_0;
    const Number gamma_degree_1;
    const Number gamma_degree_2;
    const bool   is_two_phase;

    // verbosity level for output (0: nothing(=default), 1: something, 2: some more details)
    const unsigned int verbosity;

    dealii::ConditionalOStream pcout;
  };
} // namespace MeltPoolDG::CutUtil