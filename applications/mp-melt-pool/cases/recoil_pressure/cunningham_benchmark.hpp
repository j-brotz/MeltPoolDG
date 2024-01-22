#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/core/simulation_base.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::CunninghamBenchmark
{

  using namespace dealii;

  static double width = 600e-6;

  // Note: For 1d we consider the coordinates along the y-axis.
  static double height_substrate = 430e-6; // m
  static double height_gas       = 170e-6;

  static double delta_h = 20e-6; // TODO: add to ParameterHandler

  // initial temperature values
  static double T_initial_top    = 298; // K
  static double T_initial_bottom = T_initial_top;

  // boundary conditions
  static double inflow_velocity    = 0.1;
  static double outlet_pressure    = 0.0;
  static double inflow_temperature = T_initial_top;

  template <int dim>
  class InflowVelocity : public Function<dim>
  {
  public:
    InflowVelocity()
      : Function<dim>(dim)
    {}

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      if (std::abs(p[0] + width * 0.5) < 1e-12 && p[dim - 1] > delta_h)
        return (component == 0) ?
                 std::min(inflow_velocity, inflow_velocity * p[dim - 1] / (1.05 * delta_h)) :
                 0.0;
      else
        return 0;
    }
  };

  template <int dim>
  class InitialConditionTemperature : public Function<dim>
  {
  public:
    InitialConditionTemperature(const double T_initial_bottom,
                                const double T_initial_top,
                                const double y_min,
                                const double y_max)
      : Function<dim>()
      , T_initial_bottom(T_initial_bottom)
      , T_initial_top(T_initial_top)
      , y_min(y_min)
      , grad_T((T_initial_top - T_initial_bottom) / (y_max - y_min))
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      if (T_initial_top == T_initial_bottom)
        return T_initial_top;
      else

        return T_initial_bottom + grad_T * (p[dim - 1] - y_min);
    }

    const double T_initial_bottom;
    const double T_initial_top;
    const double y_min;
    const double grad_T;
  };



  template <int dim, typename Number>
  class SimulationCunninghamBenchmark : public MeltPoolCase<dim, Number>
  {
  private:
    std::vector<unsigned int> cell_repetitions;

    unsigned int n_local_refinement = 0;

    Point<dim> local_refinement_1_bottom_left;
    Point<dim> local_refinement_1_top_right;
    Point<dim> local_refinement_2_bottom_left;
    Point<dim> local_refinement_2_top_right;
    Point<dim> bottom_left;
    Point<dim> top_right;

    // Postprocessor
    mutable std::ofstream file_conservation_variables;
    mutable int           n_time_step = 0;

    mutable TableHandler output_table;

  public:
    SimulationCunninghamBenchmark(std::string parameter_file, const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, Number>(parameter_file, mpi_communicator)
      , cell_repetitions(dim, 1)
    {
      const Number half_width = width * 0.5;
      bottom_left             = (dim == 1) ? Point<dim>(-height_substrate) :
                                (dim == 2) ? Point<dim>(-half_width, -height_substrate) :
                                             Point<dim>(-half_width, -half_width, -height_substrate);
      top_right               = (dim == 1) ? Point<dim>(height_gas) :
                                (dim == 2) ? Point<dim>(half_width, height_gas) :
                                             Point<dim>(half_width, half_width, height_gas);
    }

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.enter_subsection("domain");
        {
          prm.add_parameter("width", width, "Width of the box. Only relevant for dim > 1.");
          prm.add_parameter("height substrate", height_substrate, "Height of the metal substrate.");
          prm.add_parameter("height gas", height_gas, "Height of the ambient gas.");
        }
        prm.leave_subsection();

        prm.enter_subsection("mesh");
        {
          prm.add_parameter("cell repetitions",
                            cell_repetitions,
                            "cell repetitions per dim applied before global refinement or amr");
          prm.add_parameter("n local refinement",
                            n_local_refinement,
                            "number of (additional to the global) refinements for local region.");
          prm.add_parameter("local refinement 1 bottom left",
                            local_refinement_1_bottom_left,
                            "Bottom left point of locally refined region.");
          prm.add_parameter("local refinement 1 top right",
                            local_refinement_1_top_right,
                            "Top right point of locally refined region.");
          prm.add_parameter("local refinement 2 bottom left",
                            local_refinement_2_bottom_left,
                            "Bottom left point of locally refined region.");
          prm.add_parameter("local refinement 2 top right",
                            local_refinement_2_top_right,
                            "Top right point of locally refined region.");
        }
        prm.leave_subsection();

        prm.enter_subsection("initial temperature");
        {
          prm.add_parameter("top",
                            T_initial_top,
                            "Set the initial temperature on the top boundary.");
          prm.add_parameter("bottom",
                            T_initial_bottom,
                            "Set the initial temperature on the bottom boundary.");
        }
        prm.leave_subsection();

        prm.enter_subsection("bc");
        {
          prm.add_parameter("inflow velocity",
                            inflow_velocity,
                            "Set the inflow velocity at the vertical domain boundary.");
          prm.add_parameter("inflow temperature",
                            inflow_temperature,
                            "Set the inflow temperature at the vertical domain boundary.");
          prm.add_parameter("outlet pressure",
                            outlet_pressure,
                            "Set the pressure at the vertical outlet boundary.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
        {
#ifdef DEAL_II_WITH_METIS
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (Triangulation<dim>::MeshSmoothing::none),
            true,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
          AssertThrow(
            false,
            ExcMessage(
              "Missing Metis support of the deal.II installation. "
              "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      // create mesh
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          std::vector<unsigned int> subdivisions(
            dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;
          for (int d = 0; d < dim; d++)
            subdivisions[d] *= cell_repetitions[d];

          GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                   subdivisions,
                                                                   bottom_left,
                                                                   top_right);
        }
      else
        {
          GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                    cell_repetitions,
                                                    bottom_left,
                                                    top_right);
        }
    }

    void
    set_boundary_conditions() override
    {
      // refine global before setting IDs as we consider dimension-dependent
      // splitting of the boundaries
      this->triangulation->refine_global(this->parameters.base.global_refinements);

      //  Note: For 1D we consider the constraints along the z-axis, i.e. bottom_bc and top_bc.
      const types::boundary_id substrate_bc  = 1;
      const types::boundary_id gas_top_bc    = 2;
      const types::boundary_id gas_inflow_bc = 3;
      const types::boundary_id gas_outlet_bc = 4;

      const auto double_eq = [](const Number a, const Number b) {
        return std::abs(a - b) <= 1e-10;
      };

      // set boundary ids
      for (const auto &cell : this->triangulation->cell_iterators())
        //| IteratorFilters::LocallyOwnedCell())
        for (const auto &face : cell->face_iterators())
          if ((face->at_boundary()))
            {
              // set boundaries at faces where z=const
              if (double_eq(face->center()[dim - 1], bottom_left[dim - 1]))
                face->set_boundary_id(substrate_bc);
              else if (double_eq(face->center()[dim - 1], top_right[dim - 1]))
                face->set_boundary_id(gas_top_bc);
              // set wall boundaries
              else if (dim > 1)
                {
                  // substrate: walls (all points with z<delta_h)
                  if (face->center()[dim - 1] <= delta_h)
                    face->set_boundary_id(substrate_bc);
                  // gas inflow: all points with x=0 and z>=delta_h
                  else if (double_eq(face->center()[0], bottom_left[0]) &&
                           (face->center()[dim - 1] > delta_h))
                    face->set_boundary_id(gas_inflow_bc);
                  else if (face->center()[dim - 1] > delta_h)
                    face->set_boundary_id(gas_outlet_bc);
                  else
                    AssertThrow(false, ExcNotImplemented());
                }
            }

      // set field-specific BCs

      ////////////// substrate
      this->attach_boundary_condition(substrate_bc, "no_slip", "navier_stokes_u");
      this->attach_boundary_condition(
        {substrate_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_initial_bottom)},
        "dirichlet",
        "heat_transfer");
      // this->attach_dirichlet_boundary_condition(substrate_bc,
      // std::make_shared<Functions::ZeroFunction<dim>>(),
      //"reinitialization");

      ////////////// gas: top
      this->attach_boundary_condition(gas_top_bc, "symmetry", "navier_stokes_u");
      // note: isolated for temperature field

      ////////////// gas: inflow
      this->attach_boundary_condition({gas_inflow_bc, std::make_shared<InflowVelocity<dim>>()},
                                      "dirichlet",

                                      "navier_stokes_u");
      this->attach_boundary_condition(
        {gas_inflow_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_initial_bottom)},
        "dirichlet",
        "heat_transfer");
      this->attach_boundary_condition({gas_inflow_bc,
                                       std::make_shared<Functions::ConstantFunction<dim>>(-1.0)},
                                      "dirichlet",
                                      "level_set");
      // this->attach_dirichlet_boundary_condition(gas_inflow_bc,
      // std::make_shared<Functions::ZeroFunction<dim>>(),
      //"reinitialization");

      ////////////// gas: outlet
      this->attach_boundary_condition(
        {gas_outlet_bc, std::make_shared<Functions::ConstantFunction<dim>>(outlet_pressure)},
        "open",
        "navier_stokes_u");
      // this->attach_dirichlet_boundary_condition(gas_outlet_bc,
      // std::make_shared<Functions::ZeroFunction<dim>>(),
      //"reinitialization");
      // note: isolated for temperature field

      /*
       * locally refined region described by max. 2 bounding boxes
       *
       *
       *  +-------------------------+
       *  |                         |
       *  |         +--------+      |
       *  |         |refine 1|      |
       *  |         +--------+      |
       *  |                         |
       *  |                         |
       *  |         +--------+      |
       *  |         |refine 2|      |
       *  |         +--------+      |
       *  |                         |
       *  |                         |
       *  +-------------------------+
       *
       */
      if (n_local_refinement > 0)
        {
          if constexpr (dim == 2)
            {
              // 1. region
              const auto refinement_region =
                BoundingBox<dim>({local_refinement_1_bottom_left, local_refinement_1_top_right});

              // 2. region
              const auto refinement_region_2 =
                BoundingBox<dim>({local_refinement_2_bottom_left, local_refinement_2_top_right});

              for (unsigned int j = 0; j < n_local_refinement; ++j)
                {
                  for (auto &cell : this->triangulation->active_cell_iterators() |
                                      IteratorFilters::LocallyOwnedCell())
                    {
                      for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                        if (refinement_region.point_inside(cell->vertex(i)) ||
                            refinement_region_2.point_inside(cell->vertex(i)))
                          {
                            cell->set_refine_flag();
                            break;
                          }
                    }
                  this->triangulation->execute_coarsening_and_refinement();
                }
            }
          else
            AssertThrow(false, ExcNotImplemented());
        }
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<Functions::SignedDistance::Plane<dim>>(
                                       Point<dim>(), -Point<dim>::unit_vector(dim - 1)),
                                     "signed_distance");

      this->attach_initial_condition(std::make_shared<InflowVelocity<dim>>(), "navier_stokes_u");

      this->attach_initial_condition(
        std::make_shared<InitialConditionTemperature<dim>>(
          T_initial_bottom, T_initial_top, bottom_left[dim - 1], top_right[dim - 1]),
        "heat_transfer");
    }

    void
    do_postprocessing(
      [[maybe_unused]] const GenericDataOut<dim, Number> &generic_data_out) const final
    {
      if (this->parameters.output.paraview.enable == false)
        return;

      n_time_step += 1;

      if ((n_time_step % this->parameters.output.write_frequency) &&
          generic_data_out.get_time() != this->parameters.time_stepping.end_time)
        return;

      generic_data_out.get_vector("velocity").update_ghost_values();
      generic_data_out.get_vector("density").update_ghost_values();
      generic_data_out.get_vector("rho_cp").update_ghost_values();
      generic_data_out.get_vector("temperature").update_ghost_values();

      QGauss<dim> quad(generic_data_out.get_dof_handler("density").get_fe().tensor_degree() + 2);

      FEValues<dim> dens_values(generic_data_out.get_mapping(),
                                generic_data_out.get_dof_handler("density").get_fe(),
                                quad,
                                update_values | update_JxW_values);

      FEValues<dim> rho_cp_values(generic_data_out.get_mapping(),
                                  generic_data_out.get_dof_handler("rho_cp").get_fe(),
                                  quad,
                                  update_values);

      FEValues<dim> vel_values(generic_data_out.get_mapping(),
                               generic_data_out.get_dof_handler("velocity").get_fe(),
                               quad,
                               update_values);

      FEValues<dim> temp_values(generic_data_out.get_mapping(),
                                generic_data_out.get_dof_handler("temperature").get_fe(),
                                quad,
                                update_values);


      // temporary variables
      std::vector<Number>              density(dens_values.get_quadrature().size());
      std::vector<Number>              rho_cp(dens_values.get_quadrature().size());
      std::vector<Number>              temperature(dens_values.get_quadrature().size());
      std::vector<Tensor<1, dim>>      velocity(density.size(), Tensor<1, dim>());
      const FEValuesExtractors::Vector velocities(0);

      // output variables
      Number              mass = 0;
      std::vector<Number> momentum(dim, 0);
      Number              thermal_energy = 0;
      Number              kinetic_energy = 0;

      for (const auto &cell :
           this->triangulation->active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
        {
          {
            TriaIterator<DoFCellAccessor<dim, dim, false>> dof_cell(
              &generic_data_out.get_dof_handler("density").get_triangulation(),
              cell->level(),
              cell->index(),
              &generic_data_out.get_dof_handler("density"));
            dens_values.reinit(dof_cell);
            dens_values.get_function_values(generic_data_out.get_vector("density"), density);
          }

          {
            TriaIterator<DoFCellAccessor<dim, dim, false>> dof_cell(
              &generic_data_out.get_dof_handler("density").get_triangulation(),
              cell->level(),
              cell->index(),
              &generic_data_out.get_dof_handler("rho_cp"));
            rho_cp_values.reinit(dof_cell);
            rho_cp_values.get_function_values(generic_data_out.get_vector("rho_cp"), rho_cp);
            temp_values.reinit(dof_cell);
            temp_values.get_function_values(generic_data_out.get_vector("temperature"),
                                            temperature);
          }

          {
            TriaIterator<DoFCellAccessor<dim, dim, false>> dof_cell(
              &generic_data_out.get_dof_handler("density").get_triangulation(),
              cell->level(),
              cell->index(),
              &generic_data_out.get_dof_handler("velocity"));
            vel_values.reinit(dof_cell);
            vel_values[velocities].get_function_values(generic_data_out.get_vector("velocity"),
                                                       velocity);
          }

          for (const unsigned int q : dens_values.quadrature_point_indices())
            {
              mass += density[q] * dens_values.JxW(q);
              thermal_energy += rho_cp[q] * temperature[q] * dens_values.JxW(q);
              for (unsigned int d = 0; d < dim; ++d)
                {
                  momentum[d] += density[q] * velocity[q][d] * dens_values.JxW(q);
                  kinetic_energy +=
                    density[q] * velocity[q][d] * velocity[q][d] * dens_values.JxW(q);
                }
            }
        }

      generic_data_out.get_vector("velocity").zero_out_ghost_values();
      generic_data_out.get_vector("density").zero_out_ghost_values();
      generic_data_out.get_vector("rho_cp").zero_out_ghost_values();
      generic_data_out.get_vector("temperature").zero_out_ghost_values();

      mass =
        Utilities::MPI::sum(mass, generic_data_out.get_vector("density").get_mpi_communicator());

      for (unsigned int d = 0; d < dim; ++d)
        momentum[d] =
          Utilities::MPI::sum(momentum[d],
                              generic_data_out.get_vector("density").get_mpi_communicator());
      kinetic_energy =
        Utilities::MPI::sum(kinetic_energy,
                            generic_data_out.get_vector("density").get_mpi_communicator());
      thermal_energy =
        Utilities::MPI::sum(thermal_energy,
                            generic_data_out.get_vector("density").get_mpi_communicator());

      output_table.add_value("time", generic_data_out.get_time());
      output_table.add_value("mass", mass);
      output_table.add_value("kinetic_energy", kinetic_energy);
      output_table.add_value("thermal_energy", thermal_energy);

      for (unsigned int d = 0; d < dim; ++d)
        output_table.add_value("momentum_" + std::to_string(d), momentum[d]);

      // write values to file
      if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        {
          namespace fs = std::filesystem;
          std::ofstream output(
            fs::path(this->parameters.output.directory) /
            fs::path(this->parameters.output.paraview.filename + "_conservation_variables.txt"));

          output_table.write_text(output);
        }
    }
  };

} // namespace MeltPoolDG::Simulation::CunninghamBenchmark
