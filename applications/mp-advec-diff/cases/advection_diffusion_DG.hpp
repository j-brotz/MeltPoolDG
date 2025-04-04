#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "../advection_diffusion_case.hpp"

namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
{
  static bool inflow_outflow_bc = false;


  template <int dim, typename number>
  class ExactSolution : public Function<dim, number>
  {
  public:
    ExactSolution()
      : Function<dim, number>()

    {}

    number
    value(const Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const number t = this->get_time();
      return std::sin(4.0 * numbers::PI * (p[0] - 1.1 * t));
    }

  private:
  };

  /*
   * this function specifies the initial field of the level set equation
   */
  template <int dim, typename number>
  class InitializePhi : public Function<dim, number>
  {
  public:
    InitializePhi()
      : Function<dim, number>()
      , distance_sphere(dim == 1   ? Point<dim, number>(0.0) :
                        (dim == 2) ? Point<dim, number>(0.0, 0.5) :
                                     Point<dim, number>(0, 0, 0.5),
                        0.25)
    {}

    number
    value(const Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return std::sin(4.0 * numbers::PI * p[0]);
    }

  private:
    const Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim, typename number>
  class AdvectionField : public Function<dim, number>
  {
  public:
    AdvectionField()
      : Function<dim, number>(dim)
    {}

    number
    value([[maybe_unused]] const Point<dim, number> &p, const unsigned int component) const override
    {
      if (component == 0)
        return 1.1;
      else
        return 0.0;
    }
  };

  template <int dim, typename number>
  class DirichletConditions : public Function<dim, number>
  {
  public:
    DirichletConditions()
      : Function<dim, number>(dim)
    {}

    number
    value([[maybe_unused]] const Point<dim, number> &p,
          [[maybe_unused]] const unsigned int        component) const override
    {
      const number t = this->get_time();
      return std::sin(4.0 * numbers::PI * (p[0] - 1.1 * t));
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationAdvecDG : public LevelSet::AdvectionDiffusionCase<dim, number>
  {
  public:
    SimulationAdvecDG(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::AdvectionDiffusionCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
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
          GridGenerator::subdivided_hyper_cube(*this->triangulation, 2, left_domain, right_domain);
          this->triangulation->refine_global(this->parameters.base.global_refinements - 1);
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */
      constexpr types::boundary_id inflow_bc  = 42;
      constexpr types::boundary_id do_nothing = 0;

      auto dirichlet = std::make_shared<DirichletConditions<dim, number>>();

      if (inflow_outflow_bc)
        {
          this->attach_boundary_condition({inflow_bc, dirichlet},
                                          "inflow_outflow",
                                          "advection_diffusion");
          this->attach_boundary_condition({do_nothing, dirichlet},
                                          "inflow_outflow",
                                          "advection_diffusion");
        }
      else
        this->attach_boundary_condition({inflow_bc, dirichlet}, "dirichlet", "advection_diffusion");


      if constexpr (dim >= 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[0] < -0.49)
                    {
                      face->set_boundary_id(inflow_bc);
                    }
                  else
                    {
                      face->set_boundary_id(do_nothing);
                    }
                }
        }
      else
        {
          (void)do_nothing; // suppress unused variable for 1D
        }
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(),
                                     "advection_diffusion");
      this->attach_field_function(std::make_shared<AdvectionField<dim, number>>(),
                                  "prescribed_velocity",
                                  "advection_diffusion");
    }

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("inflow outflow bc",
                          inflow_outflow_bc,
                          "Set if the inflow/outflow boundary condition should be enabled.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
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
      ExactSolution<dim, number> exact_solution;
      exact_solution.set_time(generic_data_out.get_time());

      const auto n_q_points = 50; // Number is high to get accurate error even on a coarse mesh
      FE_Q<dim>  fe(this->parameters.advec_diff.fe.degree);

      QGauss<dim>   quadrature(n_q_points);
      FEValues<dim> fe_values(fe,
                              quadrature,
                              update_values | update_JxW_values | update_quadrature_points);

      std::vector<number> phi_at_q(QGauss<dim>(n_q_points).size());



      generic_data_out.get_vector("advected_field").update_ghost_values();

      number error      = 0.0;
      number norm_exact = 0.0;

      for (const auto &cell :
           generic_data_out.get_dof_handler("advected_field").active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values.get_function_values(generic_data_out.get_vector("advected_field"),
                                          phi_at_q); // compute values of old solution

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
              {
                auto sol_difference = std::abs(
                  exact_solution.value(fe_values.quadrature_point(q_index), 0) - phi_at_q[q_index]);
                error += sol_difference * sol_difference * fe_values.JxW(q_index);
                norm_exact += exact_solution.value(fe_values.quadrature_point(q_index), 0) *
                              exact_solution.value(fe_values.quadrature_point(q_index), 0) *
                              fe_values.JxW(q_index);
              }
          }
      error      = dealii::Utilities::MPI::sum(error, this->mpi_communicator);
      norm_exact = dealii::Utilities::MPI::sum(norm_exact, this->mpi_communicator);

      pcout << "Relative error to exact solution " << std::sqrt(error) / std::sqrt(norm_exact)
            << std::endl;
      pcout << "---------------------------------------------" << std::endl;
      pcout << "    End of user defined postprocessing" << std::endl;
      pcout << "---------------------------------------------" << std::endl;
    }

  private:
    const number left_domain  = -0.5;
    const number right_domain = 0.5;
  };

} // namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
