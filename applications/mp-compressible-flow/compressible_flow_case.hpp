#pragma once
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/mapping_info.h>

#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <meltpooldg/core/parameters_base.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <string>

namespace MeltPoolDG::Flow
{
  template <typename number>
  struct CompressibleFlowCaseParameters final : public ParametersBase
  {
  protected:
    void
    add_parameters(dealii::ParameterHandler &prm) override
    {
      base.add_parameters(prm);
      flow.add_parameters(prm);
      time_stepping.add_parameters(prm);
      output.add_parameters(prm);
      profiling.add_parameters(prm);
    }

    void
    post(const std::string &parameter_filename) override
    {
      output.post(time_stepping.time_step_size, parameter_filename);
      flow.post(base.fe, base.verbosity_level);

      // check input parameters for validity
      profiling.check_input_parameters(time_stepping.time_step_size);

      if (flow.domain_representation_type == "cut")
        {
          // The face-based ghost-penalty stabilization is only implemented for polynomial degrees 1
          // and 2. The third derivatives on the element faces are currently not accessible in
          // deal.II.
          AssertThrow(base.fe.degree <= 2,
                      ExcMessage(
                        "Currently, only polynomial degrees 1 and 2 are implemented for cutDG."));
        }
    }

  public:
    BaseData                         base;
    CompressibleFlowData<number>     flow;
    TimeSteppingData<number>         time_stepping;
    OutputData<number>               output;
    Profiling::ProfilingData<number> profiling;
  };

  template <int dim, typename number>
  class CompressibleFlowCase : public SimulationCaseBase<dim, number>
  {
  public:
    CompressibleFlowCaseParameters<number> parameters;

    CompressibleFlowCase(const std::string &parameter_file_in, MPI_Comm mpi_communicator_in)
      : SimulationCaseBase<dim, number>(parameter_file_in, mpi_communicator_in)
    {
      ParameterHandler prm;
      parameters.process_parameters_file(prm, parameter_file_in);
    }

  protected:
    /**
     * This function calculates and prints the l2-norm/l2-error of the solution given in
     * @p generic_data_out compared to a reference state given by the function
     * @p reference_function, i.e. it computes ||solution-reference||_2 for the primary variables
     * density, momentum and energy density. The string @p norm_name is printed to the console to
     * indicate what the calculated values represent. If the reference solution represents an exact
     * analytical solution, the norm_name @p "error" should be chosen. Otherwise, the @p norm_name
     * "norm" is appropriate for the l2-norm of the deviation to a certain defined reference state,
     * e.g. initial conditions. It is necessary to provide a reference function for both cases.
     *
     * @param generic_data_out The postprocessing data.
     * @param reference_function Analytical reference function for error/norm computation.
     * @param norm_name Choose between "norm" and "error".
     */
    void
    print_relative_norm(const GenericDataOut<dim, number> &generic_data_out,
                        dealii::Function<dim>             &reference_function,
                        const std::string                 &norm_name = "norm") const
    {
      const dealii::ConditionalOStream pcout(std::cout,
                                             Utilities::MPI::this_mpi_process(
                                               this->mpi_communicator) == 0 and
                                               parameters.base.verbosity_level >= 1);

      const auto &dof_vector  = generic_data_out.get_vector("density");
      const auto &dof_handler = generic_data_out.get_dof_handler("density");

      reference_function.set_time(generic_data_out.get_time());

      number errors_squared[3] = {};

      bool is_cut = false;
      if (dof_handler.get_fe_collection().size() > 1)
        is_cut = true;

      if (not is_cut)
        {
          // in the fitted mesh case, we can use the DoF values for error computation

          std::map<types::global_dof_index, Point<dim, number>> support_points =
            DoFTools::map_dofs_to_support_points(generic_data_out.get_mapping(), dof_handler);
          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_artificial() or cell->is_ghost())
                continue;
              const unsigned int nodes_per_cell = cell->get_fe().dofs_per_cell / (dim + 2);
              std::vector<types::global_dof_index> local_dof_indices(cell->get_fe().dofs_per_cell);
              cell->get_dof_indices(local_dof_indices);

              for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
                {
                  switch (unsigned int component = i / nodes_per_cell)
                    {
                        case 0: {
                          const number error =
                            dof_vector[local_dof_indices[i]] -
                            reference_function.value(support_points[local_dof_indices[i]], 0);
                          errors_squared[0] += error * error;
                          break;
                        }
                        case dim + 1: {
                          const number error =
                            dof_vector[local_dof_indices[i]] -
                            reference_function.value(support_points[local_dof_indices[i]], dim + 1);
                          errors_squared[2] += error * error;
                          break;
                        }
                        default: {
                          Assert(component > 0 and component < dim + 1, dealii::ExcInternalError());
                          const number error =
                            dof_vector[local_dof_indices[i]] -
                            reference_function.value(support_points[local_dof_indices[i]],
                                                     component);
                          errors_squared[1] += error * error;
                        }
                    }
                }
            }
        }
      else
        {
          // if cutDG is used, we evaluate the error at the quadrature points

          // reading DoF-vector and DoFHandler of level-set
          // (an assert is thrown in the getter functions, if 'level_set' can not be found in
          // the entries of 'generic_data_out')
          const auto &level_set             = generic_data_out.get_vector("level_set");
          const auto &level_set_dof_handler = generic_data_out.get_dof_handler("level_set");

          // generate FESystem
          const unsigned int fe_degree = dof_handler.get_fe_collection().max_degree();
          FE_DGQ<dim>        fe_q(fe_degree);
          FESystem<dim>      fe_point_temp(fe_q, dim + 2);

          // classify cells
          NonMatching::MeshClassifier<dim> mesh_classifier(level_set_dof_handler, level_set);
          mesh_classifier.reclassify();

          // compute quadrature
          NonMatching::MappingInfo<dim, dim, dealii::VectorizedArray<number>> mapping_info_cell(
            generic_data_out.get_mapping(), update_values | update_JxW_values);

          hp::QCollection<1> q_collection((dealii::QGauss<1>(fe_degree + 1)));
          NonMatching::DiscreteQuadratureGenerator<dim> quadrature_generator(q_collection,
                                                                             level_set_dof_handler,
                                                                             level_set);

          auto physical = [&](const typename DoFHandler<dim>::active_cell_iterator &i) {
            return (mesh_classifier.location_to_level_set(i) ==
                    NonMatching::LocationToLevelSet::intersected) or
                   (mesh_classifier.location_to_level_set(i) ==
                    NonMatching::LocationToLevelSet::
                      outside /*active phase has positive level-set*/);
          };

          const auto physical_active_cell_iterators =
            dof_handler.active_cell_iterators() | physical;

          std::vector<Quadrature<dim>> quad_vec_cell;

          unsigned int n_active_cells = std::distance(dof_handler.active_cell_iterators().begin(),
                                                      dof_handler.active_cell_iterators().end());

          for (const auto &cell : physical_active_cell_iterators)
            {
              quadrature_generator.generate(cell);
              // active phase has positive level-set
              quad_vec_cell.push_back(quadrature_generator.get_outside_quadrature());
            }

          mapping_info_cell.reinit_cells(physical_active_cell_iterators,
                                         quad_vec_cell,
                                         n_active_cells);

          // cell-loop for error-norm computation
          FEPointEvaluation<dim + 2, dim, dim, VectorizedArray<number>> fe_point_eval(
            mapping_info_cell, fe_point_temp);

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_artificial() or cell->is_ghost() or !physical(cell))
                continue;

              std::vector<number> solution_values_in(cell->get_fe().dofs_per_cell);

              cell->get_dof_values(dof_vector,
                                   solution_values_in.begin(),
                                   solution_values_in.end());

              fe_point_eval.reinit(cell->active_cell_index());
              fe_point_eval.evaluate(solution_values_in, EvaluationFlags::values);

              VectorizedArray<number> local_errors_squared[3] = {};

              for (const unsigned int q : fe_point_eval.quadrature_point_indices())
                {
                  const auto error =
                    VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
                      reference_function, fe_point_eval.quadrature_point(q)) -
                    fe_point_eval.get_value(q);

                  const auto JxW          = fe_point_eval.JxW(q);
                  local_errors_squared[0] = error[0] * error[0] * JxW;

                  for (unsigned int d = 0; d < dim; ++d)
                    local_errors_squared[1] = (error[d + 1] * error[d + 1]) * JxW;
                  local_errors_squared[2] = (error[dim + 1] * error[dim + 1]) * JxW;

                  for (unsigned int v = 0;
                       v < fe_point_eval.n_active_entries_per_quadrature_batch(q);
                       ++v)
                    for (unsigned int d = 0; d < 3; ++d)
                      errors_squared[d] += local_errors_squared[d][v];
                }
            }
        }

      Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);
      std::array<number, 3> errors;
      for (unsigned int d = 0; d < 3; ++d)
        errors[d] = std::sqrt(errors_squared[d]);

      std::ostringstream output;
      output << std::scientific << std::setprecision(4) << norm_name << " rho: " << errors[0]
             << ", rho * u: " << errors[1] << ", rho * energy: " << errors[2];
      Journal::print_line(pcout, output.str(), "compressible_flow");
    }
  };
} // namespace MeltPoolDG::Flow
