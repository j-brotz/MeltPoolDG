#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>

#include "advection_diffusion.hpp"

/*
 * This is a simple simulation where the access to the output vectors in the Simulation
 * class is shown.
 */

namespace MeltPoolDG::Simulation::AdvectionDiffusion
{
  using namespace dealii;

  template <int dim>
  class SimulationAdvectionDiffusionUserOutput : public SimulationBase<dim>
  {
  public:
    SimulationAdvectionDiffusionUserOutput(std::string    parameter_file,
                                           const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (dim == 1 || this->parameters.base.do_simplex)
        {
          AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<Triangulation<dim>>();
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      if (this->parameters.base.do_simplex)
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

      auto dirichlet = std::make_shared<DirichletCondition<dim>>();

      this->attach_dirichlet_boundary_condition(inflow_bc, dirichlet, "advection_diffusion");

      /*
       *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
       *  due to the hyperbolic nature of the analyzed problem
       *
                  out    in
      (-1,1)  +---------------+ (1,1)
              |       :       |
        in    |       :       | out
              |_______________|
              |       :       |
        out   |       :       | in
              |       :       |
              +---------------+
       * (-1,-1)  in     out   (1,-1)
       */
      if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  const double half_line = (right_domain + left_domain) / 2;

                  if (face->center()[0] == left_domain && face->center()[1] > half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[0] == right_domain && face->center()[1] < half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[1] == right_domain && face->center()[0] > half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[1] == left_domain && face->center()[0] < half_line)
                    face->set_boundary_id(inflow_bc);
                  else
                    face->set_boundary_id(do_nothing);
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
      this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(), "advection_diffusion");
      this->attach_advection_field(std::make_shared<AdvectionField<dim>>(), "advection_diffusion");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const final
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
      pcout << "---------------------------------------------" << std::endl;
      pcout << "    End of user defined postprocessing" << std::endl;
      pcout << "---------------------------------------------" << std::endl;
    }

  private:
    const double left_domain  = -1.0;
    const double right_domain = 1.0;
  };
} // namespace MeltPoolDG::Simulation::AdvectionDiffusion
