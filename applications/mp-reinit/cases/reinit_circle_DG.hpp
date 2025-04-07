#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>

#include "../reinitialization_case.hpp"

namespace MeltPoolDG::Simulation::ReinitCircleDG
{
  /*
   * this function specifies the initial field of the level set equation
   */

  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim, number>
  {
  public:
    InitializePhi()
      : dealii::Function<dim, number>()
      , distance_sphere(dim == 1 ? dealii::Point<dim, number>(0.0) :
                                   dealii::Point<dim, number>(0.0, 0.5),
                        0.25)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      number value;
      if (dim != 3)
        {
          value = 2.0 * (std::sqrt(p[0] * p[0] + p[1] * p[1]) - 0.15);
        }
      else
        {
          value = 2.0 * (std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - 0.15);
        }

      return value;
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim, typename number>
  class ExactSolution : public dealii::Function<dim, number>
  {
  public:
    ExactSolution(const number eps)
      : dealii::Function<dim, number>()
      , distance_sphere(dealii::Point<dim, number>(0.0, 0.5), 0.25)
      , eps_interface(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      number value;
      if (dim != 3)
        {
          value = (std::sqrt(p[0] * p[0] + p[1] * p[1]) - 0.15);
        }
      else
        {
          value = (std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - 0.15);
        }

      return value;
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
    const number                                         eps_interface;
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationReinitDG : public LevelSet::ReinitializationCase<dim, number>
  {
  public:
    SimulationReinitDG(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::ReinitializationCase<dim, number>(parameter_file, mpi_communicator)
    {
      if (dim == 1)
        AssertThrow(false, dealii::ExcMessage("Dimension 1 is not supported for this simulation"));
    }

    void
    create_spatial_discretization() override
    {
      using namespace dealii;
      if (dim == 1 || this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<Triangulation<dim>>();
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          GridGenerator::subdivided_hyper_cube_with_simplices(
            *this->triangulation,
            Utilities::pow(2, this->parameters.base.global_refinements),
            left_domain,
            right_domain);
        }
      else
        {
          GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() override
    {
      /**
       * For the DG case no boundary conditions are implemented
       */
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(), "level_set");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      using namespace dealii;
      dealii::ConditionalOStream pcout(std::cout,
                                       Utilities::MPI::this_mpi_process(this->mpi_communicator) ==
                                         0);

      pcout << "---------------------------------------------" << std::endl;
      pcout << "    Starting user defined postprocessing" << std::endl;
      pcout << "---------------------------------------------" << std::endl;
      pcout << "Accessible vectors:" << std::endl;
      for (const auto &entry : generic_data_out.entries)
        for (const auto &name : std::get<2>(entry))
          {
            pcout << " * " << std::setw(20) << name << " Max-norm: " << std::setprecision(5)
                  << std::setw(10) << generic_data_out.get_vector(name).linfty_norm()
                  << " number of dofs: " << generic_data_out.get_dof_handler(name).n_dofs()
                  << std::endl;
            break;
          }

      /*Error Calculation*/
      ExactSolution<dim, number> exact_solution(0.1);
      exact_solution.set_time(generic_data_out.get_time());

      const auto  n_q_points = this->parameters.reinit.fe.degree + 20;
      FE_DGQ<dim> fe(this->parameters.reinit.fe.degree);

      QGauss<dim>   quadrature(n_q_points);
      FEValues<dim> fe_values(fe,
                              quadrature,
                              update_values | update_JxW_values | update_quadrature_points |
                                update_gradients);

      FEValues<dim> fe_values_curvature(
        fe, quadrature, update_values | update_JxW_values | update_quadrature_points);

      std::vector<number>                 phi_at_q(QGauss<dim>(n_q_points).size());
      std::vector<number>                 curvature_at_q(QGauss<dim>(n_q_points).size());
      std::vector<Tensor<1, dim, number>> gradient_at_q(QGauss<dim>(n_q_points).size());



      generic_data_out.get_vector("reinit_DG_psi").update_ghost_values();
      generic_data_out.get_vector("curvature").update_ghost_values();

      number error                = 0.0;
      number norm_exact           = 0.0;
      number mass                 = 0.0;
      number error_curvature      = 0.0;
      number norm_curvature_exact = 0.0;
      number error_gradient       = 0.0;
      number norm_gradient_exact  = 0.0;
      // Initialize mass
      if (init_mass)
        {
          for (const auto &cell :
               generic_data_out.get_dof_handler("reinit_DG_psi").active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);

                fe_values.get_function_values(generic_data_out.get_vector("reinit_DG_psi"),
                                              phi_at_q); // compute values of old solution

                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                  {
                    if (phi_at_q[q_index] < 0.0)
                      {
                        initial_mass += fe_values.JxW(q_index);
                      }
                  }
              }
          initial_mass = dealii::Utilities::MPI::sum(initial_mass, this->mpi_communicator);

          init_mass = false;
        }
      for (const auto &cell :
           generic_data_out.get_dof_handler("reinit_DG_psi").active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values_curvature.reinit(cell);

            fe_values.get_function_values(generic_data_out.get_vector("reinit_DG_psi"),
                                          phi_at_q); // compute values of old solution

            fe_values_curvature.get_function_values(generic_data_out.get_vector("curvature"),
                                                    curvature_at_q); // compute curvature values

            fe_values.get_function_gradients(generic_data_out.get_vector("reinit_DG_psi"),
                                             gradient_at_q); // compute values of old solution

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
              {
                const auto radius = std::sqrt(
                  fe_values.quadrature_point(q_index)[0] * fe_values.quadrature_point(q_index)[0] +
                  fe_values.quadrature_point(q_index)[1] * fe_values.quadrature_point(q_index)[1]);

                const auto exact_curvature = -1. / radius;

                auto sol_difference = std::abs(
                  exact_solution.value(fe_values.quadrature_point(q_index), 0) - phi_at_q[q_index]);

                auto curv_difference = std::abs(exact_curvature - curvature_at_q[q_index]);

                number norm_gradient = 0.0;

                for (unsigned int d = 0; d < dim; d++)
                  {
                    norm_gradient += gradient_at_q[q_index][d] * gradient_at_q[q_index][d];
                  }
                norm_gradient = std::sqrt(norm_gradient);

                if (std::abs(exact_solution.value(fe_values.quadrature_point(q_index), 0)) < 0.02)
                  {
                    {
                      error += sol_difference * sol_difference * fe_values.JxW(q_index);
                      error_curvature += curv_difference * curv_difference * fe_values.JxW(q_index);

                      norm_exact += exact_solution.value(fe_values.quadrature_point(q_index), 0) *
                                    exact_solution.value(fe_values.quadrature_point(q_index), 0) *
                                    fe_values.JxW(q_index);
                      norm_curvature_exact +=
                        exact_curvature * exact_curvature * fe_values.JxW(q_index);

                      error_gradient +=
                        (norm_gradient - 1.0) * (norm_gradient - 1.0) * fe_values.JxW(q_index);

                      norm_gradient_exact += fe_values.JxW(q_index);
                    }
                  }
                if (phi_at_q[q_index] < 0.0)
                  {
                    mass += fe_values.JxW(q_index);
                  }
              }
          }

      error           = dealii::Utilities::MPI::sum(error, this->mpi_communicator);
      norm_exact      = dealii::Utilities::MPI::sum(norm_exact, this->mpi_communicator);
      mass            = dealii::Utilities::MPI::sum(mass, this->mpi_communicator);
      error_curvature = dealii::Utilities::MPI::sum(error_curvature, this->mpi_communicator);
      norm_curvature_exact =
        dealii::Utilities::MPI::sum(norm_curvature_exact, this->mpi_communicator);

      error_gradient = dealii::Utilities::MPI::sum(error_gradient, this->mpi_communicator);
      norm_gradient_exact =
        dealii::Utilities::MPI::sum(norm_gradient_exact, this->mpi_communicator);

      pcout << "Relative error to exact solution " << std::sqrt(error) / std::sqrt(norm_exact)
            << std::endl;

      pcout << "Mass of bubble " << mass << std::endl;
      pcout << "Mass loss " << ((mass - initial_mass) / initial_mass) * 100.0 << " % " << std::endl;

      pcout << "Relative error to exact curvature "
            << std::sqrt(error_curvature) / std::sqrt(norm_curvature_exact) << std::endl;

      pcout << "Relative error in gradient "
            << std::sqrt(error_gradient) / std::sqrt(norm_gradient_exact) << std::endl;

      pcout << "---------------------------------------------" << std::endl;
      pcout << "    End of user defined postprocessing" << std::endl;
      pcout << "---------------------------------------------" << std::endl;
    }

  private:
    number         left_domain  = -0.5;
    number         right_domain = 0.5;
    mutable bool   init_mass    = true;
    mutable number initial_mass = 0.0;
  };

} // namespace MeltPoolDG::Simulation::ReinitCircleDG
