#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out_resample.h>
#include <deal.II/numerics/vector_tools.h>

// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

/**
 * This example is derived from
 *
 * Hardt, S., and F. Wondra. "Evaporation model for interfacial flows based on a continuum-field
 * representation of the source terms." Journal of Computational Physics 227.11 (2008): 5871-5895.
 *
 * and represents the film boiling example.
 */

namespace dealii::GridGenerator
{
  template <int dim, typename VectorType>
  void
  create_triangulation_with_marching_cube_algorithm(Triangulation<dim - 1, dim> &tria,
                                                    const Mapping<dim> &         mapping,
                                                    const DoFHandler<dim> &background_dof_handler,
                                                    const VectorType &     ls_vector,
                                                    const double           iso_level,
                                                    const unsigned int     n_subdivisions = 1,
                                                    const double           tolerance      = 1e-10)
  {
    std::vector<Point<dim>>        vertices;
    std::vector<CellData<dim - 1>> cells;
    SubCellData                    subcelldata;

    const GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(mapping,
                                                               background_dof_handler.get_fe(),
                                                               n_subdivisions,
                                                               tolerance);

    mc.process(background_dof_handler, ls_vector, iso_level, vertices, cells);

    std::vector<unsigned int> considered_vertices;

    // note: the following operation does not work for simplex meshes yet
    // GridTools::delete_duplicated_vertices (vertices, cells, subcelldata,
    // considered_vertices);

    if (vertices.size() > 0)
      tria.create_triangulation(vertices, cells, subcelldata);
  }
} // namespace dealii::GridGenerator

namespace MeltPoolDG::Simulation::FilmBoiling
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  /**
   *  Initial level set field
   */
  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const double x_min,
                    const double x_max,
                    const double y_min,
                    const double y_interface,
                    const double lambda0)
      : Function<dim>()
      , x_min(x_min)
      , x_max(x_max)
      , y_min(y_min)
      , y_interface(y_interface)
      , lambda0(lambda0)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      double radius;
      if (dim == 2)
        radius = p[0];
      else
        radius = std::sqrt(p[0] * p[0] + p[1] * p[1]);

      double y_interface_disturbed =
        y_interface +
        lambda0 / 160. * std::cos(2 * numbers::PI * radius / lambda0); // according to Hardt

      Point<dim> lower_left  = dim == 1 ? Point<dim>(y_min) :
                               dim == 2 ? Point<dim>(x_min, y_min) :
                                          Point<dim>(x_min, x_min, y_min);
      Point<dim> upper_right = dim == 1 ? Point<dim>(y_interface_disturbed) :
                               dim == 2 ? Point<dim>(x_max, y_interface_disturbed) :
                                          Point<dim>(x_max, x_max, y_interface_disturbed);

      return -UtilityFunctions::CharacteristicFunctions::sgn(
        DistanceFunctions::rectangular_manifold<dim>(p, lower_left, upper_right));
    }
    double x_min, x_max, y_min, y_interface, lambda0;
  };
  /**
   *  Initial temperature field
   */
  template <int dim>
  class InitialValuesTemperature : public Function<dim>
  {
  public:
    InitialValuesTemperature(const double T_max,
                             const double T_min,
                             const double y_interface,
                             const double lambda0)
      : Function<dim>()
      , T_max(T_max)
      , T_min(T_min)
      , y_interface(y_interface)
      , lambda0(lambda0)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      double y_interface_disturbed =
        y_interface + lambda0 / 160. * std::cos(2 * numbers::PI * p[0] / lambda0);

      if (p[dim - 1] > y_interface_disturbed)
        return T_min;
      else
        return T_max - (T_max - T_min) / y_interface_disturbed * p[dim - 1];
    }
    double T_max, T_min, y_interface, lambda0;
  };

  /**
   *      This class collects all relevant input data for the simulation.
   */
  template <int dim>
  class SimulationFilmBoiling : public SimulationBase<dim>
  {
  public:
    SimulationFilmBoiling(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
      , lambda0(
          2. * numbers::PI *
          std::sqrt(3. * this->parameters.surface_tension.surface_tension_coefficient /
                    (this->parameters.base.gravity * (this->parameters.material.second.density -
                                                      this->parameters.material.first.density))))
      , x_max(lambda0 / 2.)
      , y_max(lambda0)
      , x_min(-x_max)
    {}

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.do_simplex || dim == 1)
        {
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      // create mesh
      const Point<dim> bottom_left = dim == 1   ? Point<dim>(y_min) :
                                     (dim == 2) ? Point<dim>(x_min, y_min) :
                                                  Point<dim>(x_min, x_min, y_min);
      const Point<dim> top_right   = dim == 1   ? Point<dim>(y_max) :
                                     (dim == 2) ? Point<dim>(x_max, y_max) :
                                                  Point<dim>(x_max, x_max, y_max);

      if (this->parameters.base.do_simplex)
        {
          // create mesh
          std::vector<unsigned int> subdivisions(
            dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;

          GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                   subdivisions,
                                                                   bottom_left,
                                                                   top_right);
        }
      else
        {
          std::vector<unsigned int> subdivisions(dim, 1);
          subdivisions[dim - 1] *= 3;
          GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                    subdivisions,
                                                    bottom_left,
                                                    top_right);
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const types::boundary_id lower_bc = 1;
      const types::boundary_id upper_bc = 2;
      const types::boundary_id left_bc  = 3;
      const types::boundary_id right_bc = 4;
      const types::boundary_id front_bc = 5;
      const types::boundary_id back_bc  = 6;


      if constexpr (dim == 1)
        {
          for (auto &cell : this->triangulation->cell_iterators())
            for (auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[0] == y_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[0] == y_max)
                    face->set_boundary_id(upper_bc);
                }
        }
      else if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[1] == y_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[1] == y_max)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == x_min)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
                }
        }
      else if constexpr (dim == 3)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[2] == y_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[2] == y_max)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == x_min)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
                  else if (face->center()[1] == x_min)
                    face->set_boundary_id(front_bc);
                  else if (face->center()[1] == x_max)
                    face->set_boundary_id(back_bc);
                }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");

      this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");

      this->attach_periodic_boundary_condition(left_bc, right_bc, 0);
      if (dim == 3)
        this->attach_periodic_boundary_condition(front_bc, back_bc, 1);

      this->attach_dirichlet_boundary_condition(lower_bc,
                                                std::make_shared<Functions::ConstantFunction<dim>>(
                                                  this->parameters.material.boiling_temperature +
                                                  5.),
                                                "heat_transfer");

      // @todo: had to comment out the following line to not get a partitioner error --> maybe its
      //        the corner nodes have both -- dirichlet and PBC?
      this->attach_dirichlet_boundary_condition(
        lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1), "level_set");

      if (!this->parameters.base.do_simplex)
        this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(
        std::make_shared<InitialValuesLS<dim>>(x_min, x_max, y_min, 9. * lambda0 / 128., lambda0),
        "level_set");

      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");

      // boiling temperature at the interface
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(
                                       this->parameters.material.boiling_temperature + 5,
                                       this->parameters.material.boiling_temperature,
                                       9. * lambda0 / 128.,
                                       lambda0),
                                     "heat_transfer");
    }

    void
    do_postprocessing([[maybe_unused]] const GenericDataOut<dim> &generic_data_out) const final
    {
      if (this->parameters.paraview.do_output == false)
        return;

      // create slice
      if constexpr (dim == 3)
        {
          parallel::distributed::Triangulation<2, 3> tria_slice(this->mpi_communicator);

          const Point<2> bottom_left(x_min, y_min);
          const Point<2> top_right(x_max, y_max);

          std::vector<unsigned int> subdivisions{1, 3};

          GridGenerator::subdivided_hyper_rectangle(tria_slice,
                                                    subdivisions,
                                                    bottom_left,
                                                    top_right);

          GridTools::rotate(Point<dim>::unit_vector(0), 0.5 * numbers::PI, tria_slice);

          tria_slice.refine_global(this->parameters.base.global_refinements);

          MappingQ1<2, 3> mapping_slice;

          DataOutResample<3, 2, 3> data_out(tria_slice, mapping_slice);
          data_out.add_data_vector(generic_data_out.get_dof_handler("level_set"),
                                   generic_data_out.get_vector("level_set"),
                                   "level_set");
          data_out.add_data_vector(generic_data_out.get_dof_handler("temperature"),
                                   generic_data_out.get_vector("temperature"),
                                   "temperature");
          data_out.update_mapping(generic_data_out.get_mapping());
          data_out.build_patches();
          data_out.write_vtu_with_pvtu_record(
            "./", "data_out_01" /*TODO*/, 0, this->mpi_communicator, 1 /*TODO*/, 1);
        }

      // create iso-surface
      if constexpr (dim >= 2)
        {
          Triangulation<dim - 1, dim> tria;

          GridGenerator::create_triangulation_with_marching_cube_algorithm(
            tria,
            generic_data_out.get_mapping(),
            generic_data_out.get_dof_handler("level_set"),
            generic_data_out.get_vector("level_set"),
            0 /*iso_level*/,
            1 /*n_subdivisions*/);


          DataOut<dim - 1, dim> data_out;
          data_out.attach_triangulation(tria);
          if (tria.n_cells() > 0)
            data_out.build_patches();
          data_out.write_vtu_with_pvtu_record(
            "./", "data_out_02" /*TODO*/, 0, this->mpi_communicator, 1 /*TODO*/, 1);
        }
    }

  private:
    double       lambda0 = 0.0;
    double       x_max;
    double       y_max;
    const double x_min;
    const double y_min = 0.0;
  };
} // namespace MeltPoolDG::Simulation::FilmBoiling
