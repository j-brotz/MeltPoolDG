/* ---------------------------------------------------------------------
 *
 * Author: Johannes Ressch TUM, June 2024
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim>
  class LevelSetOperationBase
  {
    // triangulation info on surface mesh of zero level set contour
    using SurfaceMeshInfo =
      std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                             std::vector<Point<dim>> /*quad_points*/,
                             std::vector<double> /*weights*/
                             >>;

  public:
    virtual void
    set_initial_condition(const Function<dim> &initial_field_function_level_set,
                          const bool is_signed_distance_initial_field_function = false) = 0;

    virtual void
    set_inflow_outflow_bc(
      [[maybe_unused]] const std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
        inflow_outflow_bc) = 0;

    virtual void
    reinit() = 0;

    virtual void
    distribute_constraints() = 0;


    virtual void
    init_time_advance() = 0;

    virtual void
    solve(const bool do_finish_time_step = true) = 0;

    virtual void
    finish_time_advance() = 0;

    virtual void
    set_level_set_user_rhs(const VectorType &level_set_user_rhs) = 0;

    virtual void
    update_normal_vector() = 0;

    /*
     *  getter functions for solution vectors
     */
    virtual const LinearAlgebra::distributed::Vector<double> &
    get_curvature() const = 0;

    virtual LinearAlgebra::distributed::Vector<double> &
    get_curvature() = 0;

    virtual const LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() const = 0;

    virtual LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() = 0;

    virtual const LinearAlgebra::distributed::Vector<double> &
    get_level_set() const = 0;

    virtual LinearAlgebra::distributed::Vector<double> &
    get_level_set() = 0;

    virtual const LinearAlgebra::distributed::Vector<double> &
    get_level_set_as_heaviside() const = 0;

    virtual LinearAlgebra::distributed::Vector<double> &
    get_level_set_as_heaviside() = 0;

    virtual const LinearAlgebra::distributed::Vector<double> &
    get_distance_to_level_set() const = 0;

    virtual const SurfaceMeshInfo &
    get_surface_mesh_info() const = 0;
    /**
     * register vectors for adaptive mesh refinement
     */
    virtual void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;

    virtual void
    attach_output_vectors(GenericDataOut<dim> &data_out) const = 0;

    virtual void
    update_surface_mesh() = 0;

    virtual void
    transform_level_set_to_smooth_heaviside() = 0;

  protected:
    /**
     * The given distance value is transformed to a smooth heaviside function \f$H_\epsilon\f$,
     * which has the property of \f$\int \nabla H_\epsilon=1\f$. This function has its transition
     * region between -2 and 2.
     */
    virtual inline double
    smooth_heaviside_from_distance_value(const double x /*distance*/)
    {
      if (x > 0)
        return 1. - smooth_heaviside_from_distance_value(-x);
      else if (x < -2.)
        return 0;
      else if (x < -1.)
        {
          const double x2 = x * x;
          return (
            0.125 * (5. * x + x2) + 0.03125 * (-3. - 2. * x) * std::sqrt(-7. - 12. * x - 4. * x2) -
            0.0625 * std::asin(std::sqrt(2.) * (x + 1.5)) + 23. * 0.03125 - numbers::PI / 64.);
        }
      else
        {
          const double x2 = x * x;
          return (
            0.125 * (3. * x + x2) - 0.03125 * (-1. - 2. * x) * std::sqrt(1. - 4. * x - 4. * x2) +
            0.0625 * std::asin(std::sqrt(2.) * (x + 0.5)) + 15. * 0.03125 - numbers::PI / 64.);
        }
    }
  };
} // namespace MeltPoolDG::LevelSet
