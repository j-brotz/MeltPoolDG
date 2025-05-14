#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "../reinitialization_case.hpp"

namespace MeltPoolDG::Simulation::WallWetting
{
  /**
   * @brief Initial condition of the level-set field.
   * @tparam dim Number of spatial dimensions of the simulation
   * @tparam number Numeric type used for computations (e.g., float, double)
   */
  template <int dim, typename number>
  class InitialSignedDistance : public dealii::Function<dim>
  {
  public:
    /**
     * @brief Function describing the initial state of the level-set field. The function computes a
     * field with the interface as a vertical at a given position x = @p p_x_interface.
     *
     * @param[in] p_x_interface Position of the vertical interface (\f$ phi \f$ = 0).
     * @param[in] p_epsilon Interface thickness parameter.
     */
    InitialSignedDistance(const number p_x_interface, const number p_epsilon)
      : dealii::Function<dim>()
      , x_interface(p_x_interface)
      , epsilon(p_epsilon)
    {}

    /**
     * @brief Evaluate the value of the level-set indicator (\f$ phi \f$) at a given point @p p.
     *
     * @param[in] p Evaluation point coordinates
     *
     * @return Value of the level-set indicator
     */
    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const final
    {
      return -CharacteristicFunctions::tanh_characteristic_function(x_interface - p[0], epsilon);
    }

  private:
    /// Position of the vertical interface
    number x_interface;
    /// Interface thickness parameter
    number epsilon;
  };

  /**
   * @brief Function that imposes the bottom wall wetting boundary condition component-wise for
   * component-wise AffineConstraints.
   *
   * @tparam dim Number of spatial dimensions of the simulation
   * @tparam number Numeric type used for computations (e.g., float, double)
   */
  template <int dim, typename number>
  class BottomBoundaryNormalPerComponent : public dealii::Function<dim>
  {
  public:
    /**
     * @brief Function used to compute the normal vector at the bottom boundary considering wetting.
     *
     * @paramp[in] p_vector_component Component of the normal vector to compute
     *
     * @paramp[in] p_contact_angle Imposed static contact angle value in radians
     *
     * @note Here, opposed to BottomBoundaryNormal, the vector component is fixed in the constructor
     * to ensure that the component-wise AffineConstraints object holds the right values.
     */
    BottomBoundaryNormalPerComponent(const number p_vector_component, const number p_contact_angle)
      : dealii::Function<dim>()
      , vector_component(p_vector_component)
      , contact_angle(p_contact_angle)
    {}

    /**
     * @brief Evaluate the value of the normal vector component at a given point of the boundary.
     *
     * @return Value of the normal vector component
     */
    double
    value(const dealii::Point<dim> & /*p*/, const unsigned int /*component*/) const final
    {
      if (vector_component == 0)
        return std::sin(contact_angle);
      else /*component == 1*/
        return std::cos(contact_angle);
    }

  private:
    /// Component of the normal vector currently evaluated
    const number vector_component;
    /// Imposed static contact angle in radians
    const number contact_angle;
  };

  /**
   * @brief Function that computes the normal vector at bottom boundary considering wetting.
   *
   * @tparam dim Number of spatial dimensions of the simulation
   * @tparam number Numeric type used for computations (e.g., float, double)
   */
  template <int dim, typename number>
  class BottomBoundaryNormal : public dealii::Function<dim>
  {
  public:
    /**
     * @brief Function used to compute the normal vector at the bottom boundary considering wetting.
     *
     * @paramp[in] p_contact_angle Imposed static contact angle value in radians.
     */
    BottomBoundaryNormal(const number p_contact_angle)
      : dealii::Function<dim>()
      , contact_angle(p_contact_angle)
    {}

    /**
     * @brief Evaluate the value of the normal vector @p component at a given point @p
     * of the boundary.
     *
     * @param[in] p Evaluation point coordinates
     * @param[in] component Component of the normal vector to compute
     *
     * @return Value of the normal vector component
     */
    double
    value(const dealii::Point<dim> & /*p*/, const unsigned int component) const final
    {
      if (component == 0)
        return std::sin(contact_angle);
      else /*component == 1*/
        return std::cos(contact_angle);
    }

  private:
    /// Imposed static contact angle in radians
    const number contact_angle;
  };

  /**
   * @brief Simulation of wetting at the bottom wall. This simulates the modelling problem described
   * in "A conservative level set method for contact line dynamics" by Zahedi et al. (2009).
   * DOI: https://doi.org/10.1016/j.jcp.2009.05.043
   *
   * @tparam dim Number of spatial dimensions of the simulation
   * @tparam number Numeric type used for computations (e.g., float, double)
   */
  template <int dim, typename number>
  class SimulationWallWetting : public LevelSet::ReinitializationCase<dim, number>
  {
  public:
    /**
     * @brief Constructor of the wall wetting simulation using wetting boundary condition
     *
     * @param[in] parameter_file Parameter file associated to the simulation.
     *
     * @param[in] mpi_communicator MPI communicator.
     */
    SimulationWallWetting(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::ReinitializationCase<dim, number>(parameter_file, mpi_communicator)
    {}

    /**
     * @brief Add parameters that are specific to the simulation.
     *
     * @param[in, out] prm ParameterHandler object to which parameters can be added
     *
     * @return Boolean indicating if parameters should be printed
     */
    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) final
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("contact angle",
                          contact_angle_deg,
                          "contact angle at the bottom wall",
                          dealii::Patterns::Double(0, 180));
        prm.add_parameter(
          "gamma factor",
          gamma_factor,
          "factor multiplying epsilon_n in the computation of the normal vector filter parameter as defined by Zahedi et al. (2009)",
          dealii::Patterns::Double());
        prm.add_parameter(
          "wetting bc method",
          wetting_bc_method_string,
          "type of wetting approach used; options are <wetting_map|wetting_affine_constraints|contact_angle>",
          dealii::Patterns::Selection("wetting_map|wetting_affine_constraints|contact_angle"));
        prm.add_parameter(
          "time-step factor",
          time_step_factor,
          "time-step scaling factor; it is used to conduct a time-step sensitivity analysis",
          dealii::Patterns::Double());
      }
      prm.leave_subsection();
      return this->parameters.base.do_print_parameters;
    }

    /**
     * @brief Create the spatial discretization of the problem.
     */
    void
    create_spatial_discretization() final
    {
      // Check that the simulation is run in 2D
      AssertDimension(dim, 2);

      // Generate triangulation
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      // Create mesh
      std::vector<unsigned>  refinements(2, 25);
      const dealii::Point<2> bottom_left(x_min, y_min);
      const dealii::Point<2> top_right(x_max, y_max);
      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation, refinements, bottom_left, top_right, true);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * @brief Set boundary conditions.
     */
    void
    set_boundary_conditions() final
    {
      // Face numbering according to the deal.II colorize flag
      const auto [bottom_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      // Wetting boundary condition
      const number contact_angle_rad = compute_contact_angle_in_radians(contact_angle_deg);

      if (wetting_bc_method_string == "wetting_map")
        this->attach_boundary_condition(
          {bottom_bc, std::make_shared<BottomBoundaryNormal<dim, number>>(contact_angle_rad)},
          "wetting",
          "normal_vector");
      else if (wetting_bc_method_string == "wetting_affine_constraints")
        // This option requires component-wise definition of AffineConstraints objects. Therefore,
        // one function per component has to be instantiated.
        {
          this->attach_boundary_condition(
            {bottom_bc,
             std::make_shared<BottomBoundaryNormalPerComponent<dim, number>>(0, contact_angle_rad)},
            "nx",
            "normal_vector");
          this->attach_boundary_condition(
            {bottom_bc,
             std::make_shared<BottomBoundaryNormalPerComponent<dim, number>>(1, contact_angle_rad)},
            "ny",
            "normal_vector");
        }
      else if (wetting_bc_method_string == "contact_angle")
        this->attach_boundary_condition({bottom_bc,
                                         std::make_shared<dealii::Functions::ConstantFunction<dim>>(
                                           contact_angle_rad)},
                                        "contact_angle",
                                        "normal_vector");
      else
        AssertThrow(
          false,
          dealii::ExcMessage(
            "The selected wetting method is invalid; options are: <wetting_map|wetting_affine_constraints|contact_angle>"));
    }

    /**
     * @brief Set initial level-set field condition.
     */
    void
    set_field_conditions() final
    {
      // Compute normal diffusion factor (we consider a static mesh)
      number cell_size =
        dealii::GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim);
      epsilon = this->parameters.reinit.compute_interface_thickness_parameter_epsilon(
        cell_size / this->parameters.reinit.fe.get_n_subdivisions());

      // Compute time-step and change data value
      /* dt = h^2 / [2 * (epsilon_n + epsilon_t)] * time_step_factor */
      number time_step = time_step_factor * 0.5 * dealii::Utilities::fixed_power<2>(cell_size) /
                         (epsilon * (1 + this->parameters.reinit.tangential_diffusion_factor));
      this->parameters.time_stepping.time_step_size = time_step;

      // Compute normal vector filtering factor and change data value
      gamma = dealii::Utilities::fixed_power<2>(
        gamma_factor * this->parameters.reinit.interface_thickness_parameter.value);
      this->parameters.normal_vec.filter_parameter = gamma;

      // Initial level-set condition
      this->attach_initial_condition(
        std::make_shared<InitialSignedDistance<dim, number>>(x_interface, epsilon), "level_set");
    }

    /**
     * @brief Post-process the solution to extract the contact angle at the boundary for the current
     * time-step and write them in an output file.
     *
     * @param[in] generic_data_out GenericDataOut object containing solution information
     */
    void
    do_postprocessing(
      [[maybe_unused]] const GenericDataOut<dim, double> &generic_data_out) const final
    {
      // Do nothing if output is not requested
      if (not this->parameters.output.do_user_defined_postprocessing)
        return;

      // Face numbering according to the deal.II colorize flag
      const auto [bottom_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      number contact_angle;
      this->compute_contact_angle_with_grad_phi(contact_angle, generic_data_out, bottom_bc);

      if (dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        {
          std::cout << "| Static contact angle: " << this->contact_angle_deg << std::endl;
          std::cout << "| Computed contact angle: " << contact_angle << std::endl;

          // Add values to table
          postprocess_table.add_value("time", generic_data_out.get_time());
          postprocess_table.set_scientific("time", true);
          postprocess_table.set_scientific("time", 5);
          postprocess_table.add_value("contact_angle", contact_angle);

          // Write in output file
          namespace fs = std::filesystem;
          std::ofstream output(fs::path(this->parameters.output.directory) /
                               fs::path(this->parameters.output.paraview.filename + ".txt"));
          postprocess_table.write_text(output);
        } // End rank 0 scope
    }

    /**
     * @brief Compute the static contact angle in radians from degrees.
     *
     * @param contact_angle_deg Static contact angle in degrees
     *
     * @return Static contact angle in radians
     */
    inline number
    compute_contact_angle_in_radians(const number contact_angle_deg)
    {
      return contact_angle_deg * dealii::numbers::PI / 180.0;
    }

    /**
     * @brief Compute the contact angle at the contact point from level-set indicator gradients
     * (grad_phi). To evaluate at grad_phi at interface, a linear interpolation with the grad_phi
     * values of the two nearest quadrature points to the interface is done. Then, using the
     * y-component of the gradient vector, the contact angle is evaluated: n_y = cos(contact_angle).
     *
     * @param[out] contact_angle Computed contact angle value in degrees
     *
     * @param[in] generic_data_out GenericDataOut object containing solution information
     */
    void
    compute_contact_angle_with_grad_phi(number                            &contact_angle,
                                        const GenericDataOut<dim, double> &generic_data_out,
                                        const unsigned int                 bottom_bc) const
    {
      // Compute contact angle at the boundary
      dealii::FE_Q<dim>            fe(this->parameters.base.fe.degree);
      dealii::MappingQGeneric<dim> mapping(this->parameters.base.fe.degree);

      dealii::FEFaceValues<dim> phi_eval(
        mapping,
        fe,
        dealii::Quadrature<dim - 1>(fe.get_unit_face_support_points()),
        dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points);

      const unsigned int                          n_q_points = phi_eval.n_quadrature_points;
      std::vector<number>                         phi(n_q_points);
      std::vector<dealii::Tensor<1, dim, number>> grad_phi(n_q_points,
                                                           dealii::Tensor<1, dim, number>());

      // Set a tolerance to find points near the interface
      number phi_tolerance = 3 * epsilon;

      // Map containing local pair of level-set indicator absolute value and gradient
      std::map<number, dealii::Tensor<1, dim, number>> phi_positive_and_grad_phi_local_map;
      std::map<number, dealii::Tensor<1, dim, number>> phi_negative_and_grad_phi_local_map;
      std::map<number, dealii::Tensor<1, dim, number>> phi_zero_and_grad_phi_local_map;

      generic_data_out.get_vector("psi").update_ghost_values();

      for (const auto &cell : generic_data_out.get_dof_handler("psi").active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              unsigned int face_index = 0;
              // Loop over all faces of the cell
              for (const auto &face : cell->face_iterators())
                {
                  // Check if the face is at the boundary with the prescribed wetting boundary
                  // condition
                  if (face->at_boundary() and face->boundary_id() == bottom_bc)
                    {
                      // Update FEFaceValues with current face info
                      phi_eval.reinit(cell, face_index);

                      // Get phi values and gradients
                      phi_eval.get_function_values(generic_data_out.get_vector("psi"), phi);
                      phi_eval.get_function_gradients(generic_data_out.get_vector("psi"), grad_phi);

                      // Loop over quadrature points on the face
                      for (const auto &q : phi_eval.quadrature_point_indices())
                        {
                          number phi_q = phi[q];
                          if (std::abs(phi_q) < phi_tolerance)
                            {
                              if (phi_q > 1e-16)
                                phi_positive_and_grad_phi_local_map.insert(
                                  std::make_pair(phi[q], grad_phi[q]));
                              else if (phi_q < -1e-16)
                                phi_negative_and_grad_phi_local_map.insert(
                                  std::make_pair(phi[q], grad_phi[q]));
                              else
                                phi_zero_and_grad_phi_local_map.insert(
                                  std::make_pair(phi[q], grad_phi[q]));
                            }
                        } // End loop over quadrature points
                    }
                  ++face_index;
                } // End loop over faces
            }
        } // End loop over cells

      generic_data_out.get_vector("psi").zero_out_ghost_values();

      // Gather all local maps
      auto phi_positive_and_grad_phi_gathered_maps =
        dealii::Utilities::MPI::gather(this->mpi_communicator,
                                       phi_positive_and_grad_phi_local_map,
                                       0);
      auto phi_negative_and_grad_phi_gathered_maps =
        dealii::Utilities::MPI::gather(this->mpi_communicator,
                                       phi_negative_and_grad_phi_local_map,
                                       0);
      auto phi_zero_and_grad_phi_gathered_maps =
        dealii::Utilities::MPI::gather(this->mpi_communicator, phi_zero_and_grad_phi_local_map, 0);

      if (dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        {
          dealii::Tensor<1, dim, number> grad_phi_at_interface;

          // Build global map
          std::map<number, dealii::Tensor<1, dim, number>> phi_zero_and_grad_phi_global_map;

          for (const auto &local_map : phi_zero_and_grad_phi_gathered_maps)
            {
              phi_zero_and_grad_phi_global_map.insert(local_map.begin(), local_map.end());
            }
          if (phi_zero_and_grad_phi_global_map.empty())
            {
              std::map<number, dealii::Tensor<1, dim, number>> phi_positive_and_grad_phi_global_map;
              std::map<number, dealii::Tensor<1, dim, number>> phi_negative_and_grad_phi_global_map;
              for (const auto &local_map : phi_positive_and_grad_phi_gathered_maps)
                {
                  phi_positive_and_grad_phi_global_map.insert(local_map.begin(), local_map.end());
                }
              for (const auto &local_map : phi_negative_and_grad_phi_gathered_maps)
                {
                  phi_negative_and_grad_phi_global_map.insert(local_map.begin(), local_map.end());
                }

              // Check that the map is not empty
              AssertThrow(
                phi_positive_and_grad_phi_global_map.size() > 0,
                dealii::ExcMessage(
                  "Not enough points were found to evaluate the contact angle at the interface. "
                  "'phi_tolerance' is too restrictive."));
              AssertThrow(
                phi_negative_and_grad_phi_global_map.size() > 0,
                dealii::ExcMessage(
                  "Not enough points were found to evaluate the contact angle at the interface. "
                  "'phi_tolerance' is too restrictive."));

              // Recover the gradient at the closest points to the interface
              number phi_plus  = std::numeric_limits<number>::max();  // Smallest positive phi
              number phi_minus = -std::numeric_limits<number>::max(); // Largest negative phi
              dealii::Tensor<1, dim, number> grad_phi_at_interface_plus;
              dealii::Tensor<1, dim, number> grad_phi_at_interface_minus;

              for (const auto &[phi, grad_phi] : phi_positive_and_grad_phi_global_map)
                {
                  if (phi < phi_plus)
                    {
                      phi_plus                   = phi;
                      grad_phi_at_interface_plus = grad_phi;
                    }
                }
              for (const auto &[phi, grad_phi] : phi_negative_and_grad_phi_global_map)
                {
                  if (phi > phi_minus)
                    {
                      phi_minus                   = phi;
                      grad_phi_at_interface_minus = grad_phi;
                    }
                }

              // Interpolate gradient value at the interface
              number phi_interpolation = (-phi_minus) / (phi_plus - phi_minus);
              grad_phi_at_interface    = (1 - phi_interpolation) * grad_phi_at_interface_minus +
                                      phi_interpolation * grad_phi_at_interface_plus;
            }
          else
            {
              number phi_interface = std::numeric_limits<number>::max();
              for (const auto &[phi, grad_phi] : phi_zero_and_grad_phi_global_map)
                {
                  if (phi < phi_interface)
                    {
                      phi_interface         = phi;
                      grad_phi_at_interface = grad_phi;
                    }
                }
            }

          // Normalize vector
          grad_phi_at_interface = grad_phi_at_interface / (grad_phi_at_interface.norm() + 1e-16);

          // Recover contact angle from grad_phi [0 - 180] deg
          contact_angle = std::acos(grad_phi_at_interface[1]) * 180 /
                          dealii::numbers::PI; /* n_y = cos(contact_angle) */
        }
    }

  private:
    /* Domain dimensions */
    /// Minimal x value of the domain
    number x_min = 0.0;
    /// Maximal x value of the domain
    number x_max = 2.0;
    /// Minimal y value of the domain
    number y_min = x_min;
    /// Maximal y value of the domain
    number y_max = x_max;

    /// Interface location
    number x_interface = 1.0;

    /// Contact angle on bottom wall
    number contact_angle_deg = 45.0;

    /// Factor multiplying the normal diffusion coefficient in the computation of the normal vector
    /// filter parameter as defined in Zahedi et al. (2009)
    number gamma_factor = 1.0;

    /// Normal vector filter parameter
    number gamma;

    /// Interface thickness parameter
    number epsilon;

    /// Time-step scaling factor (for sensitivity analysis on the time-step)
    number time_step_factor = 1.0;

    /// Method used to impose the wetting boundary condition
    std::string wetting_bc_method_string = "wetting_affine_constraints";

    /// Contact angle post-processing table
    mutable dealii::TableHandler postprocess_table;
  };

} // namespace MeltPoolDG::Simulation::WallWetting