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
// c++
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Simulation::VortexBubbleDG
{
  using namespace dealii;

  // period of the vortex flow
  static double Tf = 4.0;

  /*
   * this function specifies the initial field of the level set equation
   */
  template <int dim>
  class InitializePhi : public Function<dim>
  {
  public:
    InitializePhi()
      : Function<dim>()
      , distance_sphere(dim == 1 ? Point<dim>(0.5) : Point<dim>(0.5, 0.75), 0.15)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return -distance_sphere.value(p);
    }

  private:
    const Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim>
  class PhiExact : public Function<dim>
  {
  public:
    PhiExact(const double eps)
      : eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return distance_sphere.value(p, 0) / (2 * eps);
    }

  private:
    const double             eps;
    const InitializePhi<dim> distance_sphere;
  };

  template <int dim>
  class SignedDistanceExact : public Function<dim>
  {
  public:
    SignedDistanceExact(const double eps)
      : eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      const auto val = distance_sphere.value(p, 0);
      return val > 0 ? std::min(val, 8 * eps) : std::max(val, -8 * eps);
    }

  private:
    const double             eps;
    const InitializePhi<dim> distance_sphere;
  };

  template <int dim>
  class AdvectionField : public Function<dim>
  {
  public:
    AdvectionField()
      : Function<dim>(dim)
    {}

    void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      if constexpr (dim == 2)
        {
          const double time = this->get_time();

          const double x = p[0];
          const double y = p[1];

          const double reverseCoefficient = std::cos(numbers::PI * time / Tf);

          values[0] = reverseCoefficient *
                      (std::sin(2. * numbers::PI * y) * std::pow(std::sin(numbers::PI * x), 2.));
          values[1] = reverseCoefficient *
                      (-std::sin(2. * numbers::PI * x) * std::pow(std::sin(numbers::PI * y), 2.));
        }
      else
        AssertThrow(false, ExcMessage("Advection field for dim!=2 not implemented"));
    }
  };

  /* for constant Dirichlet conditions we could also use the ConstantFunction
   * utility from dealii
   */
  template <int dim>
  class DirichletCondition : public Function<dim>
  {
  public:
    DirichletCondition()
      : Function<dim>()
    {}

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)p;
      (void)component;
      return -1.0;
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationVortexBubbleDG : public SimulationBase<dim>
  {
  public:
    SimulationVortexBubbleDG(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
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
    set_boundary_conditions() override
    {
      /*
       *  In the DG case it its required to set the boundary even if it is a do nothing boundary
       */
      constexpr types::boundary_id do_nothing = 0;


      this->attach_open_boundary_condition(do_nothing, "level_set");



      if constexpr (dim >= 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  face->set_boundary_id(do_nothing);
                }
        }
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(), "signed_distance");
      this->attach_advection_field(std::make_shared<AdvectionField<dim>>(), "level_set");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const final
    {
      if constexpr (dim == 2)
        {
          dealii::ConditionalOStream pcout(
            std::cout, Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
          // compute area
          const auto  n_q_points = this->parameters.base.fe.degree + 3;
          FE_DGQ<dim> fe(this->parameters.base.fe.degree);

          QGauss<dim>   quadrature(n_q_points);
          FEValues<dim> fe_values(fe,
                                  quadrature,
                                  update_values | update_JxW_values | update_quadrature_points);

          std::vector<double> phi_at_q(QGauss<dim>(n_q_points).size());

          std::vector<double> volume_fraction;
          double              area_droplet = 0;
          double              area_bulk    = 0;
          const double        threshhold   = 0.0;

          generic_data_out.get_vector("level_set").update_ghost_values();

          for (const auto &cell :
               generic_data_out.get_dof_handler("level_set").active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                fe_values.get_function_values(generic_data_out.get_vector("level_set"),
                                              phi_at_q); // compute values of old solution

                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                  {
                    if (phi_at_q[q_index] >= threshhold)
                      area_droplet += fe_values.JxW(q_index);
                    else
                      area_bulk += fe_values.JxW(q_index);
                  }
              }
          generic_data_out.get_vector("level_set").zero_out_ghost_values();

          area_droplet = Utilities::MPI::sum(area_droplet, this->mpi_communicator);
          area_bulk    = Utilities::MPI::sum(area_bulk, this->mpi_communicator);

          std::ostringstream str;

          str << "area (phi>0) " << std::setw(10) << std::left << area_droplet << "   ";
          str << "area (phi<0) " << std::setw(10) << std::left << area_bulk;

          Journal::print_line(pcout, str.str(), "postprocessing", 1);
        }
    }

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("T", Tf, "Period of vortex flow.");
      }
      prm.leave_subsection();
    }

  private:
    double               left_domain  = 0.0;
    double               right_domain = 1.0;
    mutable TableHandler table;
  };
} // namespace MeltPoolDG::Simulation::VortexBubbleDG
