#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>
#include <vector>


namespace MeltPoolDG::CutUtil
{
  /**
   * @brief Operator for the solution transfer in moving boundary/interface simulations.
   *
   * The operator transfers the solution vector between different function spaces at two consecutive
   * time steps. It adapts the DoF layout to a moved interface position and extrapolates unknown
   * new DoF values using ghost-penalty extrapolation.
   *
   * The solution transfer works for both single- and two-phase cases, and for scalar as well as
   * vector-valued solution fields.
   *
   * @note Currently, the solution transfer is limited to interface movements of less than one
   * element length between two subsequent time steps.
   */
  template <int dim, typename Number>
  class SolutionTransferOperator
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

    /**
     * @brief Constructor.
     *
     * @param ghost_penalty_in Set of ghost-penalty parameters for stabilization of cut elements.
     * @param is_two_phase Boolean indicator whether a two-phase case is considered.
     * @param verbosity Verbosity level with default value 0.
     */
    SolutionTransferOperator(const GhostPenaltyData<Number> &ghost_penalty_in,
                             const bool                      is_two_phase,
                             const int                       verbosity = 0);

    /**
     * @brief Reinit function for the solution transfer according to the new immersed interface position.
     *
     * 1) Update FE-index and distribute DoFs according to new state with the moved interface.
     * 2) Set up and solve system for ghost-penalty extrapolation to determine the values of the
     * remaining undetermined DoFs
     *
     * @param cut_dof_handler DoF-Handler of the considered field with non-fitted (cut) domain representation.
     * @param tria Triangulation object.
     * @param cut_solution Solution vector which corresponds to the DoF-layout of the @p mesh_classifier_old.
     * @param mesh_classifier_old Mesh classifier object which corresponds to the old interface position.
     * @param mesh_classifier Mesh classifier object which corresponds to the new (moved) interface position.
     * @param reinit_cut_vector A Lambda function for the vector reinitialization.
     * @param setup_dof_system Set up the dof system, this includes:
     *                        - distribute DoFs on the new mesh
     *                        - create partitioning for the new mesh
     *                        - set up constraints on the new mesh
     *                        - reinit the MatrixFree object for the new DoFs (ScratchData::build())
     *                        - initialize all DoF vectors for the new DoF layout
     * @param attach_vectors Lambda function of type AttachDoFHandlerAndVectorsType, that attaches
     *                       all DoFHandlers and their respective DoFVectors that ought to be
     *                       reconstructed.
     *
     * @throws dealii::ExcMessage if the FECollection contains unsupported finite element types or
     * if assumptions about the problem phase structure are violated.
     */
    void
    reinit(dealii::DoFHandler<dim>                               &cut_dof_handler,
           dealii::Triangulation<dim>                            &tria,
           const VectorType                                      &cut_solution,
           const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier_old,
           const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier,
           const std::function<void(VectorType &)>               &reinit_cut_vector,
           const std::function<void()>                           &setup_dof_system,
           const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors = {});

    /**
     * @brief Same as above but with multiple cut solution vectors given by @p cut_solutions.
     *
     * 1) Update FE-index and distribute DoFs according to new state with the moved interface.
     * 2) Set up and solve system for ghost-penalty extrapolation to determine the values of the
     * remaining undetermined DoFs.
     *
     * @param cut_dof_handler DoF-Handler of the considered field with non-fitted (cut) domain representation.
     * @param tria Triangulation object.
     * @param cut_solutions Collection of solution vectors which corresponds to the DoF-layout of the
     * @p mesh_classifier_old.
     * @param mesh_classifier_old Mesh classifier object which corresponds to the old interface position.
     * @param mesh_classifier Mesh classifier object which corresponds to the new (moved) interface position.
     * @param reinit_cut_vector A Lambda function for the vector reinitialization.
     * @param setup_dof_system Set up the dof system, this includes:
     *                       - distribute DoFs on the new mesh
     *                       - create partitioning for the new mesh
     *                       - set up constraints on the new mesh
     *                       - reinit the MatrixFree object for the new DoFs (ScratchData::build())
     *                       - initialize all DoF vectors for the new DoF layout
     * @param attach_vectors Lambda function of type AttachDoFHandlerAndVectorsType, that attaches
     *                       all DoFHandlers and their respective DoFVectors that ought to be
     *                       reconstructed.
     */
    void
    reinit(dealii::DoFHandler<dim>                               &cut_dof_handler,
           dealii::Triangulation<dim>                            &tria,
           const std::vector<const VectorType *>                 &cut_solutions,
           const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier_old,
           const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier,
           const std::function<void(VectorType &)>               &reinit_cut_vector,
           const std::function<void()>                           &setup_dof_system,
           const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors = {});

    /**
     * @brief Getter function for multiple transferred solution vectors (const. version).
     *
     * @return A constant reference to the std::vector containing updated solution vectors.
     */
    const std::vector<VectorType> &
    get_updated_solutions() const
    {
      return new_solutions;
    }

    /**
     * @brief Getter function for multiple transferred solution vectors (non-const. version).
     *
     * @return A reference to the std::vector containing updated solution vectors.
     */
    std::vector<VectorType> &
    get_updated_solutions()
    {
      return new_solutions;
    }

    /**
     * @brief Getter function for the transferred solution vector (const. version).
     *
     * It is intended to be used in contexts where only a single solution vector is present.
     *
     * @return A constant reference to the updated solution vector.
     *
     * @throws dealii::ExcMessage if there is not exactly one solution vector in the `new_solutions`
     * container.
     */
    const VectorType &
    get_updated_solution() const
    {
      Assert(
        new_solutions.size() == 1,
        dealii::ExcMessage(
          "This function assumes that only one cut solution vector was attached, which is not the case."));
      return new_solutions[0];
    }

    /**
     * @brief Getter function for the transferred solution vector (non-const. version).
     *
     * It is intended to be used in contexts where only a single solution vector is present.
     *
     * @return A reference to the updated solution vector.
     *
     * @throws dealii::ExcMessage if there is not exactly one solution vector in the `new_solutions`
     * container.
     */
    VectorType &
    get_updated_solution()
    {
      Assert(
        new_solutions.size() == 1,
        dealii::ExcMessage(
          "This function assumes that only one cut solution vector was attached, which is not the case."));
      return new_solutions[0];
    }

  private:
    /// Collection of extrapolated solution vectors
    std::vector<VectorType> new_solutions;

    /// Finite element degree
    unsigned int fe_degree;

    /// Components of the solution in one phase
    /// (for scalar-valued solutions: 1, for vector-valued solutions: >1)
    unsigned int n_components_per_phase;

    /// Boolean indicator whether DG is used
    bool is_dg;

    /// Ghost-penalty parameters for stabilization of cut elements
    const GhostPenaltyData<Number> ghost_penalty;

    /// Boolean indicator whether a two-phase case is considered
    const bool is_two_phase;

    /// Verbosity level for output (0: nothing(=default), 1: something, 2: some more details)
    const unsigned int verbosity;

    /// Stream object for conditional output
    dealii::ConditionalOStream pcout;

    /**
     * @brief Update FE-index and distribute DoFs according to the new state with the moved interface.
     *
     * @param cut_dof_handler DoF-Handler of the considered field with non-fitted (cut) domain representation.
     * @param tria Triangulation object.
     * @param cut_solutions Collection of solution vectors which corresponds to the DoF-layout of the
     * @p mesh_classifier_old.
     * @param mesh_classifier_old Mesh classifier object which corresponds to the old interface position.
     * @param mesh_classifier Mesh classifier object which corresponds to the new (moved) interface position.
     * @param reinit_cut_vector A Lambda function for the vector reinitialization.
     * @param setup_dof_system Set up the dof system, this includes:
     *                       - distribute DoFs on the new mesh
     *                       - create partitioning for the new mesh
     *                       - set up constraints on the new mesh
     *                       - reinit the MatrixFree object for the new DoFs (ScratchData::build())
     *                       - initialize all DoF vectors for the new DoF layout
     * @param attach_vectors Lambda function of type AttachDoFHandlerAndVectorsType, that attaches
     *                       all DoFHandlers and their respective DoFVectors that ought to be
     *                       reconstructed.
     */
    void
    transfer_solution_constant_dofs(
      dealii::DoFHandler<dim>                               &cut_dof_handler,
      dealii::Triangulation<dim>                            &tria,
      const std::vector<const VectorType *>                 &cut_solutions,
      const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier_old,
      const dealii::NonMatching::MeshClassifier<dim>        &mesh_classifier,
      const std::function<void(VectorType &)>               &reinit_cut_vector,
      const std::function<void()>                           &setup_dof_system,
      const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors = {});

    /**
     * @brief Mark the unknown DoFs, which need to be ghost-penalty extrapolated.
     *
     * @param cut_dof_handler DoF-Handler of the considered field with non-fitted (cut) domain representation.
     * @param mesh_classifier_old Mesh classifier object which corresponds to the old interface position.
     * @param mesh_classifier Mesh classifier object which corresponds to the new (moved) interface position.
     *
     * @return A DoF vector in which the DoF values set to 1 indicate the degrees of freedom to be extrapolated.
     * All other DoF values are set to 0.
     */
    VectorType
    mark_dofs_for_gp_extrapolation(
      const dealii::DoFHandler<dim>                  &cut_dof_handler,
      const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
      const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old) const;

    /**
     * @brief Create constraints for the known DoFs that do not need to be ghost-penalty extrapolated.
     *
     * @param cut_dof_handler DoF-Handler of the considered field with non-fitted (cut) domain representation.
     * @param flags_dofs_gp_extrapolation A DoF vector in which the DoF values set to 1 indicate the degrees of freedom
     * to be extrapolated. All other DoF values are set to 0.
     *
     * @returns A dealii::AffineConstraints object that contains the constraints for the DoFs that do not need to be
     * extrapolated.
     */
    dealii::AffineConstraints<Number>
    create_constraints_gp_extrapolation(const dealii::DoFHandler<dim> &cut_dof_handler,
                                        const VectorType &flags_dofs_gp_extrapolation) const;

    /**
     * @brief Assemble and solve system for ghost-penalty extrapolation, apply constraints.
     *
     * @param cut_dof_handler DoF-Handler of the considered field with non-fitted (cut) domain representation.
     * @param mesh_classifier_old Mesh classifier object which corresponds to the old interface position.
     * @param mesh_classifier Mesh classifier object which corresponds to the new (moved) interface position.
     * @param reinit_vector A Lambda function for the vector reinitialization.
     */
    void
    extrapolate_solution_new_dofs(
      const dealii::DoFHandler<dim>                  &cut_dof_handler,
      const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier,
      const dealii::NonMatching::MeshClassifier<dim> &mesh_classifier_old,
      const std::function<void(VectorType &)>        &reinit_vector);
  };
} // namespace MeltPoolDG::CutUtil
