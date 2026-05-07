#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <memory>
#include <string>

#include "../../../mp-heat-transfer/heat_transfer_case.hpp"
#include "../../melt_pool_case.hpp"
#include "single_track_melt.hpp"

namespace MeltPoolDG::Simulation::SingleTrackMelt
{
  template <int dim>
  GasInletVelocity<dim>::GasInletVelocity(const double inlet_velocity)
    : Function<dim>(dim)
    , inlet_velocity(inlet_velocity)
  {}



  template <int dim>
  double
  GasInletVelocity<dim>::value(const Point<dim> & /*p*/, const unsigned int component) const
  {
    AssertThrow(dim == 3, ExcMessage("GasInletVelocity is defined for dim = 3."));

    return component == 2 ? inlet_velocity : 0.0;
  }



  template <int dim>
  InitialSignedDistance<dim>::InitialSignedDistance(const double y_interface)
    : Function<dim>()
    , y_interface(y_interface)
  {}



  template <int dim>
  double
  InitialSignedDistance<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    return -p[1] + y_interface;
  }



  template <int dim, typename Number, typename CaseClass>
  SimulationSingleTrackMelt<dim, Number, CaseClass>::SimulationSingleTrackMelt(
    std::string    parameter_file,
    const MPI_Comm mpi_communicator)
    : CaseClass(parameter_file, mpi_communicator)
  {}



  template <int dim, typename Number, typename CaseClass>
  bool
  SimulationSingleTrackMelt<dim, Number, CaseClass>::add_simulation_specific_parameters(
    ParameterHandler &prm)
  {
    prm.enter_subsection("case specific");
    {
      prm.enter_subsection("mesh");
      {
        prm.add_parameter("gmsh file name", mesh_file_name, "Path to the gmsh .msh file.");
        prm.add_parameter("additional refinement 1",
                          n_additional_refinement_1,
                          "Additional refinement of box 1.");
        prm.add_parameter("additional refinement 2",
                          n_additional_refinement_2,
                          "Additional refinement of box 1.");
        prm.add_parameter("local refinement 1 bottom left",
                          local_refinement_1_bottom_left,
                          "Bottom left point of locally refined region.");
        prm.add_parameter("local refinement 1 top right",
                          local_refinement_1_top_right,
                          "Bottom left point of locally refined region.");
        prm.add_parameter("local refinement 2 bottom left",
                          local_refinement_2_bottom_left,
                          "Bottom left point of locally refined region.");
        prm.add_parameter("local refinement 2 top right",
                          local_refinement_2_top_right,
                          "Bottom left point of locally refined region.");
      }
      prm.leave_subsection();

      prm.enter_subsection("initial conditions");
      {
        prm.add_parameter("temperature", T_initial, "Initial temperature.");

        prm.add_parameter("interface y", y_interface, "Initial y-position of the interface.");
      }
      prm.leave_subsection();

      prm.enter_subsection("bc");
      {
        prm.add_parameter("inlet velocity", inlet_velocity, "Gas inlet velocity in z-direction.");

        prm.add_parameter("inlet temperature",
                          inlet_temperature,
                          "Temperature at gas inlet and outlet.");

        prm.add_parameter("outlet pressure", outlet_pressure, "Outlet pressure.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();

    return this->parameters.base.do_print_parameters;
  }



  template <int dim, typename Number, typename CaseClass>
  void
  SimulationSingleTrackMelt<dim, Number, CaseClass>::create_spatial_discretization()
  {
    AssertThrow(dim == 3, ExcMessage("SimulationSingleTrackMelt expects a 3D gmsh mesh."));

    this->triangulation =
      std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

    // ---------------------------------------------------------------------------
    // Use gmsh mesh if provided
    // ---------------------------------------------------------------------------
    if (!mesh_file_name.empty())
      {
        std::ifstream input_file(mesh_file_name);

        if (input_file)
          {
            GridIn<dim> grid_in;
            grid_in.attach_triangulation(*this->triangulation);

            grid_in.read_msh(input_file);

            return;
          }
      }

    // ---------------------------------------------------------------------------
    // Otherwise create a simple hyper-rectangle domain
    // Dimensions taken from the reference geometry (in mm)
    // then converted to meters.
    // ---------------------------------------------------------------------------

    constexpr double mm_to_m = 1e-3;

    const Point<dim> p1(-0.5 * mm_to_m,  // x_min
                        -1.4 * mm_to_m,  // y_min
                        -1.5 * mm_to_m); // z_min

    const Point<dim> p2(0.5 * mm_to_m,  // x_max
                        0.6 * mm_to_m,  // y_max
                        1.5 * mm_to_m); // z_max

    GridGenerator::hyper_rectangle(*this->triangulation, p1, p2);
  }



  template <int dim, typename Number, typename CaseClass>
  void
  SimulationSingleTrackMelt<dim, Number, CaseClass>::set_boundary_conditions()
  {
    /*
     * Boundary ids from gmsh physical groups / Lethe prm:
     *
     * 2,3,4,5,6  : solid walls, no-slip
     * 7,8,9      : gas slip boundaries
     * 10         : gas outlet
     * 11         : gas inlet
     */

    const std::vector<types::boundary_id> wall_bcs     = {2, 3, 4, 5, 6};
    const std::vector<types::boundary_id> gas_slip_bcs = {7, 8, 9};

    const types::boundary_id gas_outlet_bc = 10;
    const types::boundary_id gas_inlet_bc  = 11;

    if constexpr (std::is_same_v<CaseClass, MeltPoolCase<dim, Number>>)
      {
        if (inlet_velocity > 0)
          this->attach_boundary_condition(
            {gas_inlet_bc, std::make_shared<Functions::ConstantFunction<dim>>(inlet_temperature)},
            "dirichlet",
            "heat_transfer");

        for (const auto id : wall_bcs)
          this->attach_boundary_condition(id, "no_slip", "navier_stokes_u");

        for (const auto id : gas_slip_bcs)
          this->attach_boundary_condition(id, "symmetry", "navier_stokes_u");

        this->attach_boundary_condition(
          {gas_outlet_bc, std::make_shared<Functions::ConstantFunction<dim>>(outlet_pressure)},
          "open",
          "navier_stokes_u");

        this->attach_boundary_condition({gas_inlet_bc,
                                         std::make_shared<GasInletVelocity<dim>>(inlet_velocity)},
                                        "dirichlet",
                                        "navier_stokes_u");
      }

    this->triangulation->refine_global(this->parameters.base.global_refinements);

    // local refinement
    // 1. region
    const auto refinement_region_1 =
      dealii::BoundingBox<dim>({local_refinement_1_bottom_left, local_refinement_1_top_right});

    for (unsigned int j = 0; j < n_additional_refinement_1; ++j)
      {
        for (auto &cell : this->triangulation->active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                  if (refinement_region_1.point_inside(cell->vertex(i)))
                    {
                      cell->set_refine_flag();
                      break;
                    }
              }
          }
        this->triangulation->execute_coarsening_and_refinement();
      }

    // 2. region
    const auto refinement_region_2 =
      dealii::BoundingBox<dim>({local_refinement_2_bottom_left, local_refinement_2_top_right});

    for (unsigned int j = 0; j < n_additional_refinement_2; ++j)
      {
        for (auto &cell : this->triangulation->active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                  if (refinement_region_2.point_inside(cell->vertex(i)))
                    {
                      cell->set_refine_flag();
                      break;
                    }
              }
          }
        this->triangulation->execute_coarsening_and_refinement();
      }
  }



  template <int dim, typename Number, typename CaseClass>
  void
  SimulationSingleTrackMelt<dim, Number, CaseClass>::set_field_conditions()
  {
    if constexpr (std::is_same_v<CaseClass, MeltPoolCase<dim, Number>>)
      {
        this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                       "navier_stokes_u");

        this->attach_initial_condition(std::make_shared<InitialSignedDistance<dim>>(y_interface),
                                       "signed_distance");
      }
    if constexpr (std::is_same_v<CaseClass, Heat::HeatTransferCase<dim, Number>>)
      this->attach_initial_condition(std::make_shared<InitialSignedDistance<dim>>(y_interface),
                                     "prescribed_signed_distance");

    this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(T_initial),
                                   "heat_transfer");
  }
} // namespace MeltPoolDG::Simulation::SingleTrackMelt
