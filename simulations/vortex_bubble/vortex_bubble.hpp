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
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace VortexBubble
    {
      using namespace dealii;

      // period of the vortex flow
      static double Tf = 2.0;

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
          return std::tanh(distance_sphere.value(p, 0) / (2 * eps));
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

              values[0] = reverseCoefficient * (std::sin(2. * numbers::PI * y) *
                                                std::pow(std::sin(numbers::PI * x), 2.));
              values[1] = reverseCoefficient * (-std::sin(2. * numbers::PI * x) *
                                                std::pow(std::sin(numbers::PI * y), 2.));
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
      class SimulationVortexBubble : public SimulationBase<dim>
      {
      public:
        SimulationVortexBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
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
              GridGenerator::subdivided_hyper_cube(*this->triangulation,
                                                   2,
                                                   left_domain,
                                                   right_domain);
              this->triangulation->refine_global(this->parameters.base.global_refinements - 1);
            }
        }

        void
        set_boundary_conditions() override
        {
          // none
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
              const auto n_q_points = this->parameters.base.fe.degree + 3;
              FE_Q<dim>  fe(this->parameters.base.fe.degree);

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

              pcout << "area (phi>0) " << area_droplet << std::endl;
              pcout << "area (phi<0) " << area_bulk << std::endl;

              // compute circularity
              // 1) compute the surface of the droplet
              Triangulation<std::max(1, dim - 1), dim> tria_droplet_surface;

              GridGenerator::create_triangulation_with_marching_cube_algorithm<dim>(
                tria_droplet_surface,
                MappingQ<dim>(3),
                generic_data_out.get_dof_handler("level_set"),
                generic_data_out.get_vector("level_set"),
                0. /*iso level*/,
                1 /*n subdivisions of surface mesh*/);

              double area_droplet_boundary = 0;

              // check if partitioned domains contain surface elements in case of parallel execution
              if (tria_droplet_surface.n_cells() > 0)
                area_droplet_boundary =
                  GridTools::volume<std::max(1, dim - 1), dim>(tria_droplet_surface);

              area_droplet_boundary = Utilities::MPI::sum(area_droplet_boundary, MPI_COMM_WORLD);

              AssertThrow(area_droplet_boundary >= 1e-16, ExcMessage("Area of droplet is zero."));

              // 2) compute circularity
              double circularity =
                2. * std::sqrt(numbers::PI * area_droplet) / (area_droplet_boundary);
              pcout << "circularity " << circularity << std::endl;

              table.add_value("time", generic_data_out.get_time());
              table.add_value("circularity", circularity);
              table.add_value("area_droplet", area_droplet);

              if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
                {
                  namespace fs = std::filesystem;
                  std::ofstream output(
                    fs::path(this->parameters.output.directory) /
                    fs::path(this->parameters.output.paraview.filename + ".txt"));
                  table.write_text(output);
                }

              if (std::abs(generic_data_out.get_time() - this->parameters.time_stepping.end_time) <
                    1e-10 ||
                  (std::abs(generic_data_out.get_time() -
                            this->parameters.time_stepping.start_time) < 1e-10))
                {
                  namespace fs = std::filesystem;
                  std::ofstream output(
                    fs::path(this->parameters.output.directory) /
                    fs::path(this->parameters.output.paraview.filename + "_L2norm.txt"));


                  generic_data_out.get_vector("distance").update_ghost_values();
                  generic_data_out.get_vector("level_set").update_ghost_values();

                  const double eps =
                    this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
                      GridTools::minimal_cell_diameter(*this->triangulation) /
                      this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

                  // compute L2Norm of level_set
                  Vector<float> difference_per_cell(this->triangulation->n_active_cells());
                  dealii::VectorTools::integrate_difference(
                    generic_data_out.get_mapping(),
                    generic_data_out.get_dof_handler("level_set"),
                    generic_data_out.get_vector("level_set"),
                    PhiExact<dim>(eps),
                    difference_per_cell,
                    quadrature,
                    dealii::VectorTools::L2_norm);

                  const double norm =
                    dealii::VectorTools::compute_global_error(*this->triangulation,
                                                              difference_per_cell,
                                                              dealii::VectorTools::L2_norm);
                  difference_per_cell = 0;

                  // compute L2Norm of distance
                  dealii::VectorTools::integrate_difference(generic_data_out.get_mapping(),
                                                            generic_data_out.get_dof_handler(
                                                              "level_set"),
                                                            generic_data_out.get_vector("distance"),
                                                            SignedDistanceExact<dim>(eps),
                                                            difference_per_cell,
                                                            quadrature,
                                                            dealii::VectorTools::L2_norm);
                  const double norm_sd =
                    dealii::VectorTools::compute_global_error(*this->triangulation,
                                                              difference_per_cell,
                                                              dealii::VectorTools::L2_norm);

                  // compute relative L2 norm in a finite interval around the interface
                  SignedDistanceExact<dim> sd_circle(eps);
                  FEValues<dim>            fe(generic_data_out.get_mapping(),
                                   generic_data_out.get_dof_handler("level_set").get_fe(),
                                   quadrature,
                                   update_values | update_JxW_values | update_quadrature_points);

                  const unsigned int  n_q_points = fe.get_quadrature().size();
                  std::vector<double> phi_at_q(n_q_points);

                  double norm_interface_region       = 0;
                  double norm_interface_region_exact = 0;
                  double norm_interface_kreiss       = 0;

                  for (const auto &cell :
                       generic_data_out.get_dof_handler("level_set").active_cell_iterators())
                    {
                      if (cell->is_locally_owned())
                        {
                          fe.reinit(cell);
                          fe.get_function_values(generic_data_out.get_vector("distance"), phi_at_q);

                          for (const auto &q : fe.quadrature_point_indices())
                            {
                              norm_interface_kreiss +=
                                Utilities::fixed_power<2>(
                                  UtilityFunctions::heaviside(phi_at_q[q]) -
                                  UtilityFunctions::heaviside(
                                    sd_circle.value(fe.quadrature_point(q), 0))) *
                                fe.JxW(q);
                              if (std::abs(phi_at_q[q]) < 0.02)
                                {
                                  norm_interface_region +=
                                    Utilities::fixed_power<2>(
                                      sd_circle.value(fe.quadrature_point(q), 0) - phi_at_q[q]) *
                                    fe.JxW(q);
                                  norm_interface_region_exact +=
                                    Utilities::fixed_power<2>(
                                      sd_circle.value(fe.quadrature_point(q), 0)) *
                                    fe.JxW(q);
                                }
                            }
                        }
                    }

                  norm_interface_region_exact =
                    std::sqrt(Utilities::MPI::sum(norm_interface_region_exact, MPI_COMM_WORLD));
                  norm_interface_region =
                    std::sqrt(Utilities::MPI::sum(norm_interface_region, MPI_COMM_WORLD));
                  norm_interface_kreiss =
                    std::sqrt(Utilities::MPI::sum(norm_interface_kreiss, MPI_COMM_WORLD));

                  generic_data_out.get_vector("level_set").zero_out_ghost_values();
                  generic_data_out.get_vector("distance").zero_out_ghost_values();

                  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
                    output
                      << "n_dofs L2_norm(phi-phiExact) L2_norm(signed_distance-exact) relL2_normGamma(signed_distance-exact) L2_normGamma(sd_exact) L2(Kreiss)"
                      << std::endl
                      << generic_data_out.get_dof_handler("level_set").n_dofs() << " " << norm
                      << " " << norm_sd << " "
                      << norm_interface_region / norm_interface_region_exact << " "
                      << norm_interface_region_exact << " " << norm_interface_kreiss;
                }
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

    } // namespace VortexBubble
  }   // namespace Simulation
} // namespace MeltPoolDG
