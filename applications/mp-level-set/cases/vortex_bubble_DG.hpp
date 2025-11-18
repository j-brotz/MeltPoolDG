#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "../level_set_case.hpp"

namespace MeltPoolDG::Simulation::VortexBubbleDG
{
  // period of the vortex flow
  static double Tf = 4.0;

  /*
   * this function specifies the initial field of the level set equation
   */
  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim, number>
  {
  public:
    InitializePhi()
      : dealii::Function<dim, number>()
      , distance_sphere(dim == 1 ? dealii::Point<dim, number>(0.5) :
                                   dealii::Point<dim, number>(0.5, 0.75),
                        0.15)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return -distance_sphere.value(p);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim, typename number>
  class PhiExact : public dealii::Function<dim, number>
  {
  public:
    PhiExact(const number eps)
      : eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return distance_sphere.value(p, 0) / (2 * eps);
    }

  private:
    const number                     eps;
    const InitializePhi<dim, number> distance_sphere;
  };

  template <int dim, typename number>
  class SignedDistanceExact : public dealii::Function<dim, number>
  {
  public:
    SignedDistanceExact(const number eps)
      : eps(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      const auto val = distance_sphere.value(p, 0);
      return val > 0 ? std::min(val, 8 * eps) : std::max(val, -8 * eps);
    }

  private:
    const number                     eps;
    const InitializePhi<dim, number> distance_sphere;
  };

  template <int dim, typename number>
  class AdvectionField : public dealii::Function<dim, number>
  {
  public:
    AdvectionField()
      : dealii::Function<dim, number>(dim)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int c) const override
    {
      dealii::Vector<number> values(dim);
      vector_value(p, values);
      return values[c];
    }

    void
    vector_value(const dealii::Point<dim, number> &p, dealii::Vector<number> &values) const override
    {
      if constexpr (dim == 2)
        {
          const number time = this->get_time();

          const number x = p[0];
          const number y = p[1];

          const number reverseCoefficient = std::cos(dealii::numbers::PI * time / Tf);

          values[0] = reverseCoefficient * (std::sin(2. * dealii::numbers::PI * y) *
                                            std::pow(std::sin(dealii::numbers::PI * x), 2.));
          values[1] = reverseCoefficient * (-std::sin(2. * dealii::numbers::PI * x) *
                                            std::pow(std::sin(dealii::numbers::PI * y), 2.));
        }
      else
        AssertThrow(false, dealii::ExcMessage("Advection field for dim!=2 not implemented"));
    }
  };

  /* for constant Dirichlet conditions we could also use the ConstantFunction
   * utility from dealii
   */
  template <int dim, typename number>
  class DirichletCondition : public dealii::Function<dim, number>
  {
  public:
    DirichletCondition()
      : dealii::Function<dim, number>()
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int component = 0) const override
    {
      (void)p;
      (void)component;
      return -1.0;
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationVortexBubbleDG : public LevelSet::LevelSetCase<dim, number>
  {
  public:
    SimulationVortexBubbleDG(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::LevelSetCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (dim == 1 || this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          AssertDimension(dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<dealii::Triangulation<dim>>();
        }
      else
        {
          this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
            this->mpi_communicator);
        }

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          dealii::GridGenerator::subdivided_hyper_cube_with_simplices(
            *this->triangulation,
            dealii::Utilities::pow(2, this->parameters.base.global_refinements),
            left_domain,
            right_domain);
        }
      else
        {
          dealii::GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() override
    {
      /*
       *  In the DG case it its required to set the boundary even if it is a do nothing boundary
       */
      constexpr dealii::types::boundary_id do_nothing = 0;

      // this->attach_boundary_condition(do_nothing, "open" "level_set");

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
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(),
                                     "signed_distance");
      this->attach_field_function(std::make_shared<AdvectionField<dim, number>>(),
                                  "prescribed_velocity",
                                  "level_set");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const final
    {
      if constexpr (dim == 2)
        {
          dealii::ConditionalOStream pcout(
            std::cout, dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
          // compute area
          const auto          n_q_points = this->parameters.base.fe.degree + 3;
          dealii::FE_DGQ<dim> fe(this->parameters.base.fe.degree);

          dealii::QGauss<dim>   quadrature(n_q_points);
          dealii::FEValues<dim> fe_values(fe,
                                          quadrature,
                                          dealii::update_values | dealii::update_JxW_values |
                                            dealii::update_quadrature_points);

          std::vector<number> phi_at_q(dealii::QGauss<dim>(n_q_points).size());

          std::vector<number> volume_fraction;
          number              area_droplet = 0;
          number              area_bulk    = 0;
          const number        threshhold   = 0.0;

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

          area_droplet = dealii::Utilities::MPI::sum(area_droplet, this->mpi_communicator);
          area_bulk    = dealii::Utilities::MPI::sum(area_bulk, this->mpi_communicator);

          std::ostringstream str;

          str << "area (phi>0) " << std::setw(10) << std::left << area_droplet << "   ";
          str << "area (phi<0) " << std::setw(10) << std::left << area_bulk;

          Journal::print_line(pcout, str.str(), "postprocessing", 1);
        }
    }

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("T", Tf, "Period of vortex flow.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    number                       left_domain  = 0.0;
    number                       right_domain = 1.0;
    mutable dealii::TableHandler table;
  };
} // namespace MeltPoolDG::Simulation::VortexBubbleDG
