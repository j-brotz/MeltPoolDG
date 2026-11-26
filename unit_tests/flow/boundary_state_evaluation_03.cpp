#include <gtest/gtest.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/compressible_flow/boundary_condition_data.hpp>
#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.hpp>
#include <meltpooldg/compressible_flow/boundary_state_evaluation.templates.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/compressible_flow/material.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace MeltPoolDG;
using namespace MeltPoolDG::CompressibleFlow;

namespace
{
  /// The relative tolerance of all comparisons. A computed value is compared with an expected value
  /// up to this tolerance times the larger of 1 and the magnitude of the expected value.
  constexpr double tolerance = 1e-12;

  /// The ratio of specific heats of all species. Since it is the same for all species, the ratio of
  /// the mixture equals it, independently of the mass fractions.
  constexpr double ratio_of_specific_heats = 1.4;

  /// The quadrature rules supported by the non-reflecting outflow, both with N + 1 points per
  /// direction.
  enum class QuadratureRule
  {
    gauss,
    gauss_lobatto
  };

  /**
   * Parameters of a single test configuration.
   */
  template <int dim_, int n_species_, QuadratureRule quadrature_rule_>
  struct TestConfiguration
  {
    static constexpr int            dim             = dim_;
    static constexpr int            n_species       = n_species_;
    static constexpr QuadratureRule quadrature_rule = quadrature_rule_;
  };

  /**
   * The primitive state of the tests. Its density is constant, and its velocity, its pressure and
   * the mass fraction of the first species are linear in space. With polynomial degree 2, the
   * conserved variables are then interpolated exactly, so the interior values and gradients at the
   * boundary are those of this state, and its primitive variables are interpolated exactly at the
   * nodes. In the boxes of the tests, the velocity is subsonic, and the velocity, the pressure and
   * the mass fraction vary in all directions.
   */
  template <int dim>
  struct PrimitiveState
  {
    static double
    density()
    {
      return 1.2;
    }

    /// The gradient of the velocity. The entry (i, j) is the derivative of the velocity component i
    /// in the direction j.
    static dealii::Tensor<2, dim>
    velocity_gradient()
    {
      const double coefficients[3][3] = {{0.10, 0.05, -0.02},
                                         {-0.04, 0.12, 0.03},
                                         {0.02, -0.03, 0.08}};

      dealii::Tensor<2, dim> gradient;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          gradient[i][j] = coefficients[i][j];
      return gradient;
    }

    static dealii::Tensor<1, dim>
    velocity(const dealii::Point<dim> &p)
    {
      const double offsets[3] = {0.25, 0.15, -0.10};

      dealii::Tensor<1, dim> velocity;
      for (unsigned int d = 0; d < dim; ++d)
        velocity[d] = offsets[d];
      return velocity + velocity_gradient() * p;
    }

    static dealii::Tensor<1, dim>
    pressure_gradient()
    {
      const double coefficients[3] = {0.08, -0.05, 0.03};

      dealii::Tensor<1, dim> gradient;
      for (unsigned int d = 0; d < dim; ++d)
        gradient[d] = coefficients[d];
      return gradient;
    }

    static double
    pressure(const dealii::Point<dim> &p)
    {
      return 1. + pressure_gradient() * p;
    }

    static double
    mass_fraction(const dealii::Point<dim> &p)
    {
      const double coefficients[3] = {0.10, -0.05, 0.05};

      double mass_fraction = 0.4;
      for (unsigned int d = 0; d < dim; ++d)
        mass_fraction += coefficients[d] * p[d];
      return mass_fraction;
    }

    static double
    speed_of_sound(const dealii::Point<dim> &p)
    {
      return std::sqrt(ratio_of_specific_heats * pressure(p) / density());
    }
  };

  /**
   * Return the primitive variables of PrimitiveState at the point @p p. The entries are the
   * density, the velocity, the pressure and, for two species, the mass fraction of the first
   * species.
   */
  template <int dim, int n_species>
  dealii::Tensor<1, n_conserved_variables<dim, n_species>>
  primitive_variables(const dealii::Point<dim> &p)
  {
    dealii::Tensor<1, n_conserved_variables<dim, n_species>> primitive;
    primitive[0] = PrimitiveState<dim>::density();
    for (unsigned int d = 0; d < dim; ++d)
      primitive[1 + d] = PrimitiveState<dim>::velocity(p)[d];
    primitive[dim + 1] = PrimitiveState<dim>::pressure(p);
    if constexpr (n_species > 1)
      primitive[dim + 2] = PrimitiveState<dim>::mass_fraction(p);
    return primitive;
  }

  /**
   * Return the conserved variables of an ideal gas with the ratio of specific heats
   * #ratio_of_specific_heats from its primitive variables @p primitive, which are ordered as in
   * primitive_variables().
   */
  template <int dim, int n_species>
  ConservedVariablesType<dim, double, n_species, double>
  conserved_from_primitive(
    const dealii::Tensor<1, n_conserved_variables<dim, n_species>> &primitive)
  {
    ConservedVariablesType<dim, double, n_species, double> conserved;
    const double                                           density = primitive[0];

    double kinetic_energy = 0.;
    conserved[0]          = density;
    for (unsigned int d = 0; d < dim; ++d)
      {
        conserved[1 + d] = density * primitive[1 + d];
        kinetic_energy += 0.5 * density * primitive[1 + d] * primitive[1 + d];
      }
    conserved[dim + 1] = primitive[dim + 1] / (ratio_of_specific_heats - 1.) + kinetic_energy;
    for (unsigned int species = 0; species < n_species - 1; ++species)
      conserved[dim + 2 + species] = density * primitive[dim + 2 + species];
    return conserved;
  }

  /**
   * Return the primitive variables, ordered as in primitive_variables(), of the conserved variables
   * @p conserved of an ideal gas with the ratio of specific heats #ratio_of_specific_heats.
   */
  template <int dim, int n_species>
  dealii::Tensor<1, n_conserved_variables<dim, n_species>>
  primitive_from_conserved(const ConservedVariablesType<dim, double, n_species, double> &conserved)
  {
    dealii::Tensor<1, n_conserved_variables<dim, n_species>> primitive;
    const double                                             density = conserved[0];

    double kinetic_energy = 0.;
    primitive[0]          = density;
    for (unsigned int d = 0; d < dim; ++d)
      {
        primitive[1 + d] = conserved[1 + d] / density;
        kinetic_energy += 0.5 * density * primitive[1 + d] * primitive[1 + d];
      }
    primitive[dim + 1] = (ratio_of_specific_heats - 1.) * (conserved[dim + 1] - kinetic_energy);
    for (unsigned int species = 0; species < n_species - 1; ++species)
      primitive[dim + 2 + species] = conserved[dim + 2 + species] / density;
    return primitive;
  }

  /**
   * The conserved variables of PrimitiveState after the orthogonal transformation
   * x' = P x + b with a signed permutation matrix P and a shift b. The transformed state at x' is
   * the state at x = P^T (x' - b), with the velocity rotated by P.
   */
  template <int dim, int n_species>
  class ConservedState : public dealii::Function<dim>
  {
  public:
    ConservedState(const dealii::Tensor<2, dim> &transformation,
                   const dealii::Tensor<1, dim> &shift)
      : dealii::Function<dim>(n_conserved_variables<dim, n_species>)
      , transformation(transformation)
      , shift(shift)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int component) const override
    {
      const dealii::Point<dim> original_point(transpose(transformation) * (p - shift));

      dealii::Tensor<1, n_conserved_variables<dim, n_species>> primitive =
        primitive_variables<dim, n_species>(original_point);
      const dealii::Tensor<1, dim> velocity =
        transformation * PrimitiveState<dim>::velocity(original_point);
      for (unsigned int d = 0; d < dim; ++d)
        primitive[1 + d] = velocity[d];

      return conserved_from_primitive<dim, n_species>(primitive)[component];
    }

  private:
    const dealii::Tensor<2, dim> transformation;
    const dealii::Tensor<1, dim> shift;
  };

  /**
   * The target pressure of the outlet, which varies along the boundary and differs from the
   * pressure of PrimitiveState, transformed by x' = P x + b as in ConservedState.
   */
  template <int dim>
  class TargetPressure : public dealii::Function<dim>
  {
  public:
    TargetPressure(const dealii::Tensor<2, dim> &transformation,
                   const dealii::Tensor<1, dim> &shift)
      : dealii::Function<dim>(1)
      , transformation(transformation)
      , shift(shift)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int) const override
    {
      const dealii::Point<dim> original_point(transpose(transformation) * (p - shift));
      return 1. + 0.04 * original_point[0] - 0.03 * original_point[dim - 1];
    }

  private:
    const dealii::Tensor<2, dim> transformation;
    const dealii::Tensor<1, dim> shift;
  };

  /**
   * The data of one outlet face, i.e. of one active lane of a boundary face batch.
   */
  template <int dim, int n_species>
  struct OutletFace
  {
    /// The index of the boundary face batch that contains the face.
    unsigned int face_batch;

    /// The lane of the face within its face batch.
    unsigned int lane;

    /// The lower corner of the axis-aligned cell of the face.
    dealii::Point<dim> cell_lower_corner;

    /// The extents of the cell of the face in the coordinate directions.
    dealii::Tensor<1, dim> cell_extents;

    /// A vector storing the location of each face quadrature point. The index is the index of the
    /// quadrature point.
    std::vector<dealii::Point<dim>> locations;

    /// A vector storing the outward unit normal vector at each face quadrature point. The index is
    /// the index of the quadrature point.
    std::vector<dealii::Tensor<1, dim>> normals;

    /// A vector storing the JxW value of each face quadrature point. The index is the index of the
    /// quadrature point.
    std::vector<double> JxW;

    /// A vector storing the interior value at each face quadrature point. The index is the index of
    /// the quadrature point.
    std::vector<ConservedVariablesType<dim, double, n_species, double>> interior_values;

    /// A vector storing the exterior value computed by BoundaryStateEvaluation at each face
    /// quadrature point. The index is the index of the quadrature point.
    std::vector<ConservedVariablesType<dim, double, n_species, double>> exterior_values;
  };

  /**
   * A DG discretization of the conserved variables with polynomial degree 2 on a box whose lower
   * corner is the origin. The boundary is colorized, i.e. the boundary id of the lower and upper
   * boundary in direction d is 2d and 2d+1. The solution interpolates a given state.
   */
  template <int dim, int n_species>
  class OutletDiscretization
  {
  public:
    static constexpr unsigned int n_components = n_conserved_variables<dim, n_species>;
    static constexpr unsigned int degree       = 2;

    /**
     * Constructor. Sets up the discretization on the box with the upper corner @p upper_corner,
     * subdivided into @p subdivisions cells per direction, with N + 1 points of the quadrature rule
     * @p quadrature_rule per direction, and interpolates the state @p state.
     */
    OutletDiscretization(const dealii::Point<dim>        &upper_corner,
                         const std::vector<unsigned int> &subdivisions,
                         const QuadratureRule             quadrature_rule,
                         const dealii::Function<dim>     &state)
      : fe(dealii::FE_DGQ<dim>(degree), n_components)
    {
      dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation, subdivisions, dealii::Point<dim>(), upper_corner, /*colorize=*/true);

      dof_handler.reinit(triangulation);
      dof_handler.distribute_dofs(fe);
      constraints.close();

      const dealii::Quadrature<1> quadrature =
        quadrature_rule == QuadratureRule::gauss ?
          dealii::Quadrature<1>(dealii::QGauss<1>(degree + 1)) :
          dealii::Quadrature<1>(dealii::QGaussLobatto<1>(degree + 1));

      typename dealii::MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.mapping_update_flags = dealii::update_values | dealii::update_gradients |
                                             dealii::update_JxW_values |
                                             dealii::update_quadrature_points;
      additional_data.mapping_update_flags_boundary_faces =
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
        dealii::update_quadrature_points | dealii::update_normal_vectors;
      matrix_free.reinit(
        dealii::MappingQ1<dim>(), dof_handler, constraints, quadrature, additional_data);

      matrix_free.initialize_dof_vector(solution);
      dealii::VectorTools::interpolate(dof_handler, state, solution);
      solution.update_ghost_values();
    }

    /**
     * Evaluate the exterior values with BoundaryStateEvaluation on all boundary face batches and
     * return the data of all outlet faces, i.e. of all active lanes.
     */
    std::vector<OutletFace<dim, n_species>>
    evaluate_outlet(const BoundaryConditions<dim, double> &boundary_conditions,
                    const MaterialPhaseData<double>       &material) const
    {
      FEFaceIntegrator<dim, n_components, double>     phi_m(matrix_free, true);
      BoundaryStateEvaluation<dim, double, n_species> boundary_state(boundary_conditions,
                                                                     phi_m.n_q_points);

      std::vector<OutletFace<dim, n_species>> outlet_faces;
      for (unsigned int face = matrix_free.n_inner_face_batches();
           face < matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
           ++face)
        {
          phi_m.reinit(face);
          phi_m.gather_evaluate(solution,
                                dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);
          boundary_state.evaluate(phi_m, material, dealii::EvaluationFlags::values);

          for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_face_batch(face);
               ++lane)
            {
              OutletFace<dim, n_species> outlet_face;
              outlet_face.face_batch = face;
              outlet_face.lane       = lane;

              const std::pair<typename dealii::DoFHandler<dim>::cell_iterator, unsigned int>
                cell_and_face = matrix_free.get_face_iterator(face, lane, true);
              const std::pair<dealii::Point<dim>, dealii::Point<dim>> corners =
                cell_and_face.first->bounding_box().get_boundary_points();
              outlet_face.cell_lower_corner = corners.first;
              outlet_face.cell_extents      = corners.second - corners.first;

              for (const unsigned int q : phi_m.quadrature_point_indices())
                {
                  dealii::Point<dim>                                     location;
                  dealii::Tensor<1, dim>                                 normal;
                  ConservedVariablesType<dim, double, n_species, double> interior_value;
                  ConservedVariablesType<dim, double, n_species, double> exterior_value;
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      location[d] = phi_m.quadrature_point(q)[d][lane];
                      normal[d]   = phi_m.normal_vector(q)[d][lane];
                    }
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      interior_value[c] = phi_m.get_value(q)[c][lane];
                      exterior_value[c] = boundary_state.get_value(q)[c][lane];
                    }

                  outlet_face.locations.push_back(location);
                  outlet_face.normals.push_back(normal);
                  outlet_face.JxW.push_back(phi_m.JxW(q)[lane]);
                  outlet_face.interior_values.push_back(interior_value);
                  outlet_face.exterior_values.push_back(exterior_value);
                }

              outlet_faces.push_back(outlet_face);
            }
        }

      return outlet_faces;
    }

  private:
    /// The triangulation of the box.
    dealii::Triangulation<dim> triangulation;

    /// The DG element of all conserved variables.
    dealii::FESystem<dim> fe;

    /// The DoF handler of the conserved variables.
    dealii::DoFHandler<dim> dof_handler;

    /// The constraints of the conserved variables, which are empty.
    dealii::AffineConstraints<double> constraints;

    /// The matrix-free object with the quadrature of the selected rule.
    dealii::MatrixFree<dim, double> matrix_free;

    /// The interpolated conserved variables.
    dealii::LinearAlgebra::distributed::Vector<double> solution;
  };
} // namespace

/**
 * Test fixture providing the material, the coefficients of the outlet and the reference values of
 * the theory note for PrimitiveState. All species have the ratio of specific heats
 * #ratio_of_specific_heats.
 */
template <typename ConfigType>
class NonReflectingOutflowTest : public ::testing::Test
{
protected:
  static constexpr int            dim             = ConfigType::dim;
  static constexpr int            n_species       = ConfigType::n_species;
  static constexpr QuadratureRule quadrature_rule = ConfigType::quadrature_rule;
  static constexpr unsigned int   n_components    = n_conserved_variables<dim, n_species>;

  NonReflectingOutflowTest()
  {
    material.number_of_species       = n_species;
    material.gamma                   = ratio_of_specific_heats;
    material.specific_gas_constant   = 287.1;
    material.specific_isobaric_heat  = 1004.5;
    material.specific_isochoric_heat = 717.5;
    material.dynamic_viscosity       = 1.8e-5;
    material.thermal_conductivity    = 0.0257;
    for (unsigned int species = 0; species < n_species; ++species)
      {
        material.species_data[species].gamma                   = ratio_of_specific_heats;
        material.species_data[species].specific_gas_constant   = 287.1 + 100. * species;
        material.species_data[species].specific_isobaric_heat  = 1004.5 + 100. * species;
        material.species_data[species].specific_isochoric_heat = 717.5 + 50. * species;
        material.species_data[species].molar_mass              = 0.028 + 0.012 * species;
      }
  }

  /**
   * Return the coefficients of the outlet. They are chosen such that the relaxation towards the
   * target pressure and the transverse term both contribute to the correction.
   */
  static BoundaryConditionData<double>
  boundary_condition_data()
  {
    BoundaryConditionData<double> data;
    data.non_reflecting_outflow.reference_length       = 0.5;
    data.non_reflecting_outflow.relaxation_coefficient = 0.6;
    data.non_reflecting_outflow.transverse_coefficient = 0.4;
    return data;
  }

  /**
   * Return the face-averaged normal Mach number of PrimitiveState on @p outlet_face, with the JxW
   * values as weights.
   */
  static double
  reference_mean_mach_number(const OutletFace<dim, n_species> &outlet_face)
  {
    double mach_number_integral = 0.;
    double face_area            = 0.;
    for (unsigned int q = 0; q < outlet_face.locations.size(); ++q)
      {
        const dealii::Point<dim> &location = outlet_face.locations[q];
        mach_number_integral += outlet_face.JxW[q] *
                                (PrimitiveState<dim>::velocity(location) * outlet_face.normals[q]) /
                                PrimitiveState<dim>::speed_of_sound(location);
        face_area += outlet_face.JxW[q];
      }
    return mach_number_integral / face_area;
  }

  /**
   * Return the factor alpha = (L_1^+ - L_1) / (u_n - c) of the correction of the normal derivative
   * of PrimitiveState at the point @p location with the outward unit normal @p normal. It is
   * computed with Eqs. 3.3.3, 3.3.4 and 4.2.1 of the theory note, with the face-averaged normal
   * Mach number @p mean_mach_number and the target pressure @p target_pressure at the point.
   */
  static double
  reference_correction_factor(const dealii::Point<dim>     &location,
                              const dealii::Tensor<1, dim> &normal,
                              const double                  mean_mach_number,
                              const double                  target_pressure)
  {
    const typename BoundaryConditionData<double>::NonReflectingOutflow coefficients =
      boundary_condition_data().non_reflecting_outflow;

    const double                 density           = PrimitiveState<dim>::density();
    const dealii::Tensor<1, dim> velocity          = PrimitiveState<dim>::velocity(location);
    const double                 pressure          = PrimitiveState<dim>::pressure(location);
    const double                 speed_of_sound    = PrimitiveState<dim>::speed_of_sound(location);
    const dealii::Tensor<2, dim> velocity_gradient = PrimitiveState<dim>::velocity_gradient();
    const dealii::Tensor<1, dim> pressure_gradient = PrimitiveState<dim>::pressure_gradient();

    const double                 normal_velocity      = velocity * normal;
    const dealii::Tensor<1, dim> tangential_velocity  = velocity - normal_velocity * normal;
    const double normal_derivative_of_normal_velocity = normal * (velocity_gradient * normal);
    const double tangential_divergence =
      dealii::trace(velocity_gradient) - normal_derivative_of_normal_velocity;

    const double interior_amplitude =
      (normal_velocity - speed_of_sound) *
      ((pressure_gradient * normal) / (2. * density * speed_of_sound) -
       0.5 * normal_derivative_of_normal_velocity);
    const double transverse_term =
      (tangential_velocity * pressure_gradient) / (2. * density * speed_of_sound) -
      0.5 * (normal * (velocity_gradient * tangential_velocity)) +
      0.5 * speed_of_sound * tangential_divergence;
    const double relaxation_term = (1. - mean_mach_number * mean_mach_number) /
                                   (2. * density * coefficients.reference_length) *
                                   (pressure - target_pressure);
    const double prescribed_amplitude = coefficients.relaxation_coefficient * relaxation_term -
                                        coefficients.transverse_coefficient * transverse_term;

    return (prescribed_amplitude - interior_amplitude) / (normal_velocity - speed_of_sound);
  }

  /// The material of all tests.
  MaterialPhaseData<double> material;
};

using TestConfigs = ::testing::Types<TestConfiguration<2, 1, QuadratureRule::gauss>,
                                     TestConfiguration<2, 1, QuadratureRule::gauss_lobatto>,
                                     TestConfiguration<3, 1, QuadratureRule::gauss>,
                                     TestConfiguration<3, 1, QuadratureRule::gauss_lobatto>,
                                     TestConfiguration<2, 2, QuadratureRule::gauss>,
                                     TestConfiguration<2, 2, QuadratureRule::gauss_lobatto>,
                                     TestConfiguration<3, 2, QuadratureRule::gauss>,
                                     TestConfiguration<3, 2, QuadratureRule::gauss_lobatto>>;

TYPED_TEST_SUITE(NonReflectingOutflowTest, TestConfigs);



/**
 * This test checks that the jump between the exterior and the interior state of the non-reflecting
 * pressure outflow is a pure upstream acoustic wave, so that the outgoing waves pass the numerical
 * flux undisturbed (theory note, Section 5.4).
 * The test sets up a single axis-aligned cell with different extents in all directions and the
 * outlet on all of its faces, such that the faces have the normals +-x, +-y and, in 3D, +-z. The
 * interior state is PrimitiveState, and the target pressure differs from its pressure.
 * The computed jumps of the primitive variables are then checked to give zero with the left
 * eigenvectors of the entropy, vorticity, downstream acoustic and species waves, and the pressure
 * jump is checked to be nonzero.
 */
TYPED_TEST(NonReflectingOutflowTest, JumpIsIncomingAcousticWave)
{
  constexpr int dim       = TestFixture::dim;
  constexpr int n_species = TestFixture::n_species;

  dealii::Tensor<2, dim> identity;
  for (unsigned int d = 0; d < dim; ++d)
    identity[d][d] = 1.;
  const dealii::Tensor<1, dim> no_shift;

  const double       extents[3] = {0.9, 0.7, 0.5};
  dealii::Point<dim> upper_corner;
  for (unsigned int d = 0; d < dim; ++d)
    upper_corner[d] = extents[d];

  const OutletDiscretization<dim, n_species> discretization(
    upper_corner,
    std::vector<unsigned int>(dim, 1),
    TestFixture::quadrature_rule,
    ConservedState<dim, n_species>(identity, no_shift));

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = std::make_shared<TargetPressure<dim>>(identity, no_shift);
  BoundaryConditions<dim, double> boundary_conditions(TestFixture::boundary_condition_data());
  boundary_conditions.set_boundary_condition(
    BoundaryConditionType::subsonic_non_reflecting_pressure_outflow, boundary_functions);

  const std::vector<OutletFace<dim, n_species>> outlet_faces =
    discretization.evaluate_outlet(boundary_conditions, this->material);
  ASSERT_EQ(outlet_faces.size(), 2 * dim);

  double largest_pressure_jump = 0.;
  for (const OutletFace<dim, n_species> &outlet_face : outlet_faces)
    for (unsigned int q = 0; q < outlet_face.locations.size(); ++q)
      {
        SCOPED_TRACE(::testing::Message() << "face batch " << outlet_face.face_batch << ", lane "
                                          << outlet_face.lane << ", q " << q);

        const dealii::Tensor<1, TestFixture::n_components> interior =
          primitive_from_conserved<dim, n_species>(outlet_face.interior_values[q]);
        const dealii::Tensor<1, TestFixture::n_components> jump =
          primitive_from_conserved<dim, n_species>(outlet_face.exterior_values[q]) - interior;

        const dealii::Tensor<1, dim> &normal  = outlet_face.normals[q];
        const double                  density = interior[0];
        const double                  speed_of_sound =
          std::sqrt(ratio_of_specific_heats * interior[dim + 1] / density);

        dealii::Tensor<1, dim> velocity_jump;
        for (unsigned int d = 0; d < dim; ++d)
          velocity_jump[d] = jump[1 + d];
        const double                 normal_velocity_jump = velocity_jump * normal;
        const dealii::Tensor<1, dim> tangential_velocity_jump =
          velocity_jump - normal_velocity_jump * normal;

        EXPECT_NEAR(jump[0] - jump[dim + 1] / (speed_of_sound * speed_of_sound), 0., tolerance);
        for (unsigned int d = 0; d < dim; ++d)
          EXPECT_NEAR(tangential_velocity_jump[d], 0., tolerance);
        EXPECT_NEAR(jump[dim + 1] / (density * speed_of_sound) + normal_velocity_jump,
                    0.,
                    tolerance);
        for (unsigned int species = 0; species < n_species - 1; ++species)
          EXPECT_NEAR(jump[dim + 2 + species], 0., tolerance);

        largest_pressure_jump = std::max(largest_pressure_jump, std::abs(jump[dim + 1]));
      }

  EXPECT_GT(largest_pressure_jump, 1e-4);
}



/**
 * This test checks that the exterior values of the non-reflecting pressure outflow are those of the
 * polynomial correction, i.e. that the polynomial through the interior nodes and the new face
 * values has the corrected normal derivative of Eq. 5.1.5 at the face quadrature points (theory
 * note, Section 5.3.1).
 * The test sets up a single axis-aligned cell with different extents in all directions and the
 * outlet on all of its faces, such that the faces have the normals +-x, +-y and, in 3D, +-z. The
 * interior state is PrimitiveState, whose primitive variables are interpolated exactly at the
 * nodes. The test obtains the new face values by applying the inverse of the interpolation matrix
 * from the face nodes to the face quadrature points to the exterior primitive variables. The change
 * of the normal derivative then follows from the gradients of the basis functions of the face
 * nodes, since the values at all other nodes are kept. The expected change is alpha r_1 with
 * alpha from the formulas of the theory note and the face-averaged normal Mach number.
 * The computed changes of the normal derivative are then checked to agree with the expected ones
 * at all face quadrature points.
 */
TYPED_TEST(NonReflectingOutflowTest, CorrectedNormalDerivative)
{
  constexpr int          dim          = TestFixture::dim;
  constexpr int          n_species    = TestFixture::n_species;
  constexpr unsigned int n_components = TestFixture::n_components;

  dealii::Tensor<2, dim> identity;
  for (unsigned int d = 0; d < dim; ++d)
    identity[d][d] = 1.;
  const dealii::Tensor<1, dim> no_shift;

  const double       extents[3] = {0.9, 0.7, 0.5};
  dealii::Point<dim> upper_corner;
  for (unsigned int d = 0; d < dim; ++d)
    upper_corner[d] = extents[d];

  const OutletDiscretization<dim, n_species> discretization(
    upper_corner,
    std::vector<unsigned int>(dim, 1),
    TestFixture::quadrature_rule,
    ConservedState<dim, n_species>(identity, no_shift));

  const std::shared_ptr<TargetPressure<dim>> target_pressure =
    std::make_shared<TargetPressure<dim>>(identity, no_shift);
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = target_pressure;
  BoundaryConditions<dim, double> boundary_conditions(TestFixture::boundary_condition_data());
  boundary_conditions.set_boundary_condition(
    BoundaryConditionType::subsonic_non_reflecting_pressure_outflow, boundary_functions);

  const std::vector<OutletFace<dim, n_species>> outlet_faces =
    discretization.evaluate_outlet(boundary_conditions, this->material);
  ASSERT_EQ(outlet_faces.size(), 2 * dim);

  const dealii::FE_DGQ<dim>              scalar_fe(OutletDiscretization<dim, n_species>::degree);
  const std::vector<dealii::Point<dim>> &support_points = scalar_fe.get_unit_support_points();

  for (const OutletFace<dim, n_species> &outlet_face : outlet_faces)
    {
      const unsigned int n_q_points = outlet_face.locations.size();

      // The normal is constant on the face of an axis-aligned cell. The face lies at the reference
      // coordinate 1 in the normal direction if the normal points in the positive direction and at
      // 0 otherwise.
      const dealii::Tensor<1, dim> &normal           = outlet_face.normals[0];
      unsigned int                  normal_direction = 0;
      for (unsigned int d = 1; d < dim; ++d)
        if (std::abs(normal[d]) > std::abs(normal[normal_direction]))
          normal_direction = d;
      const double face_coordinate = normal[normal_direction] > 0. ? 1. : 0.;

      std::vector<unsigned int> face_nodes;
      for (unsigned int i = 0; i < scalar_fe.n_dofs_per_cell(); ++i)
        if (std::abs(support_points[i][normal_direction] - face_coordinate) < 1e-12)
          face_nodes.push_back(i);
      ASSERT_EQ(face_nodes.size(), n_q_points);

      std::vector<dealii::Point<dim>> reference_points(n_q_points);
      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int d = 0; d < dim; ++d)
          reference_points[q][d] =
            (outlet_face.locations[q][d] - outlet_face.cell_lower_corner[d]) /
            outlet_face.cell_extents[d];

      // The entry (q, b) of the interpolation matrix is the value of the basis function of the
      // face node b at the face quadrature point q.
      dealii::FullMatrix<double> interpolation_matrix(n_q_points, n_q_points);
      for (unsigned int q = 0; q < n_q_points; ++q)
        for (unsigned int b = 0; b < n_q_points; ++b)
          interpolation_matrix(q, b) = scalar_fe.shape_value(face_nodes[b], reference_points[q]);
      dealii::FullMatrix<double> inverse_interpolation_matrix(n_q_points, n_q_points);
      inverse_interpolation_matrix.invert(interpolation_matrix);

      // The new face values and the interior values at the face nodes.
      std::vector<dealii::Tensor<1, n_components>> new_face_values(n_q_points);
      std::vector<dealii::Tensor<1, n_components>> interior_face_values(n_q_points);
      for (unsigned int b = 0; b < n_q_points; ++b)
        {
          for (unsigned int q = 0; q < n_q_points; ++q)
            new_face_values[b] +=
              inverse_interpolation_matrix(b, q) *
              primitive_from_conserved<dim, n_species>(outlet_face.exterior_values[q]);

          dealii::Point<dim> node_location;
          for (unsigned int d = 0; d < dim; ++d)
            node_location[d] = outlet_face.cell_lower_corner[d] +
                               outlet_face.cell_extents[d] * support_points[face_nodes[b]][d];
          interior_face_values[b] = primitive_variables<dim, n_species>(node_location);
        }

      const double mean_mach_number = TestFixture::reference_mean_mach_number(outlet_face);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          SCOPED_TRACE(::testing::Message()
                       << "face batch " << outlet_face.face_batch << ", q " << q);

          dealii::Tensor<1, n_components> derivative_change;
          for (unsigned int b = 0; b < n_q_points; ++b)
            {
              const dealii::Tensor<1, dim> reference_gradient =
                scalar_fe.shape_grad(face_nodes[b], reference_points[q]);
              double normal_derivative = 0.;
              for (unsigned int d = 0; d < dim; ++d)
                normal_derivative +=
                  normal[d] * reference_gradient[d] / outlet_face.cell_extents[d];
              derivative_change +=
                normal_derivative * (new_face_values[b] - interior_face_values[b]);
            }

          const dealii::Point<dim> &location       = outlet_face.locations[q];
          const double              density        = PrimitiveState<dim>::density();
          const double              speed_of_sound = PrimitiveState<dim>::speed_of_sound(location);
          const double              correction_factor = TestFixture::reference_correction_factor(
            location, normal, mean_mach_number, target_pressure->value(location, 0));

          dealii::Tensor<1, n_components> expected_change;
          expected_change[0] = correction_factor * density / speed_of_sound;
          for (unsigned int d = 0; d < dim; ++d)
            expected_change[1 + d] = -correction_factor * normal[d];
          expected_change[dim + 1] = correction_factor * density * speed_of_sound;

          for (unsigned int c = 0; c < n_components; ++c)
            EXPECT_NEAR(derivative_change[c],
                        expected_change[c],
                        tolerance * std::max(1., std::abs(expected_change[c])))
              << "component " << c;
        }
    }
}



/**
 * This test checks that the faces of one face batch use the mean normal Mach number of their own
 * face, also if these differ between the faces (spec 001, R2).
 * The test sets up a box with several cells along each boundary and the outlet on all boundaries,
 * such that the faces of one boundary lie in one face batch and the normal velocity of
 * PrimitiveState differs between them. The expected exterior values are U(Q^- + delta alpha r_1)
 * with delta = h_n / 3 for polynomial degree 2 and with alpha from the formulas of the theory note
 * and the mean normal Mach number of each face.
 * The computed mean Mach numbers of the faces of at least one face batch are then checked to
 * differ, and the computed exterior values are checked to agree with the expected ones.
 */
TYPED_TEST(NonReflectingOutflowTest, FacesOfOneBatchUseTheirOwnMeanMachNumber)
{
  constexpr int          dim          = TestFixture::dim;
  constexpr int          n_species    = TestFixture::n_species;
  constexpr unsigned int n_components = TestFixture::n_components;

  dealii::Tensor<2, dim> identity;
  for (unsigned int d = 0; d < dim; ++d)
    identity[d][d] = 1.;
  const dealii::Tensor<1, dim> no_shift;

  const double       extents[3] = {0.9, 0.7, 0.5};
  dealii::Point<dim> upper_corner;
  for (unsigned int d = 0; d < dim; ++d)
    upper_corner[d] = extents[d];
  const std::vector<unsigned int> subdivisions =
    dim == 2 ? std::vector<unsigned int>{3, 4} : std::vector<unsigned int>{2, 3, 2};

  const OutletDiscretization<dim, n_species> discretization(
    upper_corner,
    subdivisions,
    TestFixture::quadrature_rule,
    ConservedState<dim, n_species>(identity, no_shift));

  const std::shared_ptr<TargetPressure<dim>> target_pressure =
    std::make_shared<TargetPressure<dim>>(identity, no_shift);
  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = target_pressure;
  BoundaryConditions<dim, double> boundary_conditions(TestFixture::boundary_condition_data());
  boundary_conditions.set_boundary_condition(
    BoundaryConditionType::subsonic_non_reflecting_pressure_outflow, boundary_functions);

  const std::vector<OutletFace<dim, n_species>> outlet_faces =
    discretization.evaluate_outlet(boundary_conditions, this->material);

  // A map storing the mean Mach numbers of the faces of each face batch by the index of the batch.
  std::map<unsigned int, std::vector<double>> mean_mach_numbers_of_batch;
  for (const OutletFace<dim, n_species> &outlet_face : outlet_faces)
    mean_mach_numbers_of_batch[outlet_face.face_batch].push_back(
      TestFixture::reference_mean_mach_number(outlet_face));

  double largest_spread = 0.;
  for (const std::pair<const unsigned int, std::vector<double>> &batch : mean_mach_numbers_of_batch)
    largest_spread = std::max(largest_spread,
                              *std::max_element(batch.second.begin(), batch.second.end()) -
                                *std::min_element(batch.second.begin(), batch.second.end()));
  ASSERT_GT(largest_spread, 1e-2);

  for (const OutletFace<dim, n_species> &outlet_face : outlet_faces)
    {
      const double mean_mach_number = TestFixture::reference_mean_mach_number(outlet_face);

      for (unsigned int q = 0; q < outlet_face.locations.size(); ++q)
        {
          SCOPED_TRACE(::testing::Message() << "face batch " << outlet_face.face_batch << ", lane "
                                            << outlet_face.lane << ", q " << q);

          const dealii::Point<dim>     &location = outlet_face.locations[q];
          const dealii::Tensor<1, dim> &normal   = outlet_face.normals[q];

          double normal_extent = 0.;
          for (unsigned int d = 0; d < dim; ++d)
            normal_extent += std::abs(normal[d]) * outlet_face.cell_extents[d];
          const double correction_length = normal_extent / 3.;

          const double density        = PrimitiveState<dim>::density();
          const double speed_of_sound = PrimitiveState<dim>::speed_of_sound(location);
          const double jump_amplitude =
            correction_length *
            TestFixture::reference_correction_factor(location,
                                                     normal,
                                                     mean_mach_number,
                                                     target_pressure->value(location, 0));

          dealii::Tensor<1, n_components> exterior_primitive =
            primitive_variables<dim, n_species>(location);
          exterior_primitive[0] += jump_amplitude * density / speed_of_sound;
          for (unsigned int d = 0; d < dim; ++d)
            exterior_primitive[1 + d] -= jump_amplitude * normal[d];
          exterior_primitive[dim + 1] += jump_amplitude * density * speed_of_sound;
          const ConservedVariablesType<dim, double, n_species, double> expected_value =
            conserved_from_primitive<dim, n_species>(exterior_primitive);

          for (unsigned int c = 0; c < n_components; ++c)
            EXPECT_NEAR(outlet_face.exterior_values[q][c],
                        expected_value[c],
                        tolerance * std::max(1., std::abs(expected_value[c])))
              << "component " << c;
        }
    }
}



/**
 * This test checks that the non-reflecting pressure outflow treats opposite sides of the domain
 * alike, i.e. that mirroring the cell and the state at a coordinate plane mirrors the exterior
 * values.
 * The test sets up a single axis-aligned cell with the outlet on all of its faces twice, once with
 * PrimitiveState and once with the state and the target pressure mirrored at the plane through the
 * center of the cell normal to x_0. The mirrored cell coincides with the original one.
 * The computed exterior values of the mirrored setup are then checked to agree with the mirrored
 * exterior values of the original setup at the mirrored points.
 */
TYPED_TEST(NonReflectingOutflowTest, MirroredState)
{
  constexpr int          dim          = TestFixture::dim;
  constexpr int          n_species    = TestFixture::n_species;
  constexpr unsigned int n_components = TestFixture::n_components;

  const double       extents[3] = {0.9, 0.7, 0.5};
  dealii::Point<dim> upper_corner;
  for (unsigned int d = 0; d < dim; ++d)
    upper_corner[d] = extents[d];

  // The original setup.
  dealii::Tensor<2, dim> identity;
  for (unsigned int d = 0; d < dim; ++d)
    identity[d][d] = 1.;
  const dealii::Tensor<1, dim> no_shift;

  const OutletDiscretization<dim, n_species> discretization(
    upper_corner,
    std::vector<unsigned int>(dim, 1),
    TestFixture::quadrature_rule,
    ConservedState<dim, n_species>(identity, no_shift));

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = std::make_shared<TargetPressure<dim>>(identity, no_shift);
  BoundaryConditions<dim, double> boundary_conditions(TestFixture::boundary_condition_data());
  boundary_conditions.set_boundary_condition(
    BoundaryConditionType::subsonic_non_reflecting_pressure_outflow, boundary_functions);

  const std::vector<OutletFace<dim, n_species>> outlet_faces =
    discretization.evaluate_outlet(boundary_conditions, this->material);

  // The mirrored setup with x' = P x + b, where P reverses x_0 and b shifts the cell back onto
  // itself.
  dealii::Tensor<2, dim> mirror = identity;
  mirror[0][0]                  = -1.;
  dealii::Tensor<1, dim> mirror_shift;
  mirror_shift[0] = extents[0];

  const OutletDiscretization<dim, n_species> mirrored_discretization(
    upper_corner,
    std::vector<unsigned int>(dim, 1),
    TestFixture::quadrature_rule,
    ConservedState<dim, n_species>(mirror, mirror_shift));

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    mirrored_boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    mirrored_boundary_functions[boundary_id] =
      std::make_shared<TargetPressure<dim>>(mirror, mirror_shift);
  BoundaryConditions<dim, double> mirrored_boundary_conditions(
    TestFixture::boundary_condition_data());
  mirrored_boundary_conditions.set_boundary_condition(
    BoundaryConditionType::subsonic_non_reflecting_pressure_outflow, mirrored_boundary_functions);

  const std::vector<OutletFace<dim, n_species>> mirrored_outlet_faces =
    mirrored_discretization.evaluate_outlet(mirrored_boundary_conditions, this->material);
  ASSERT_EQ(mirrored_outlet_faces.size(), outlet_faces.size());

  for (const OutletFace<dim, n_species> &mirrored_face : mirrored_outlet_faces)
    for (unsigned int q = 0; q < mirrored_face.locations.size(); ++q)
      {
        SCOPED_TRACE(::testing::Message()
                     << "mirrored face batch " << mirrored_face.face_batch << ", q " << q);

        // With Gauss-Lobatto quadrature, the points on the edges of a face are also points of the
        // neighboring faces, so the original point is identified by its location and its normal.
        const dealii::Point<dim>     original_point(transpose(mirror) *
                                                (mirrored_face.locations[q] - mirror_shift));
        const dealii::Tensor<1, dim> original_normal = transpose(mirror) * mirrored_face.normals[q];

        bool found = false;
        for (const OutletFace<dim, n_species> &outlet_face : outlet_faces)
          for (unsigned int p = 0; p < outlet_face.locations.size(); ++p)
            if (outlet_face.locations[p].distance(original_point) < 1e-12 and
                (outlet_face.normals[p] - original_normal).norm() < 1e-12)
              {
                found = true;

                const ConservedVariablesType<dim, double, n_species, double> &original_value =
                  outlet_face.exterior_values[p];
                ConservedVariablesType<dim, double, n_species, double> expected_value =
                  original_value;
                for (unsigned int i = 0; i < dim; ++i)
                  {
                    expected_value[1 + i] = 0.;
                    for (unsigned int j = 0; j < dim; ++j)
                      expected_value[1 + i] += mirror[i][j] * original_value[1 + j];
                  }

                for (unsigned int c = 0; c < n_components; ++c)
                  EXPECT_NEAR(mirrored_face.exterior_values[q][c],
                              expected_value[c],
                              tolerance * std::max(1., std::abs(expected_value[c])))
                    << "component " << c;
              }
        EXPECT_TRUE(found);
      }
}



/**
 * This test checks that the non-reflecting pressure outflow treats all coordinate directions alike,
 * i.e. that permuting the coordinate axes of the cell and the state permutes the exterior values.
 * The test sets up a single axis-aligned cell with different extents in all directions and the
 * outlet on all of its faces twice, once with PrimitiveState and once with the cell, the state and
 * the target pressure transformed by the cyclic permutation x'_i = x_{i+1} of the axes.
 * The computed exterior values of the permuted setup are then checked to agree with the permuted
 * exterior values of the original setup at the permuted points.
 */
TYPED_TEST(NonReflectingOutflowTest, PermutedAxes)
{
  constexpr int          dim          = TestFixture::dim;
  constexpr int          n_species    = TestFixture::n_species;
  constexpr unsigned int n_components = TestFixture::n_components;

  const double       extents[3] = {0.9, 0.7, 0.5};
  dealii::Point<dim> upper_corner;
  for (unsigned int d = 0; d < dim; ++d)
    upper_corner[d] = extents[d];

  // The original setup.
  dealii::Tensor<2, dim> identity;
  for (unsigned int d = 0; d < dim; ++d)
    identity[d][d] = 1.;
  const dealii::Tensor<1, dim> no_shift;

  const OutletDiscretization<dim, n_species> discretization(
    upper_corner,
    std::vector<unsigned int>(dim, 1),
    TestFixture::quadrature_rule,
    ConservedState<dim, n_species>(identity, no_shift));

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    boundary_functions[boundary_id] = std::make_shared<TargetPressure<dim>>(identity, no_shift);
  BoundaryConditions<dim, double> boundary_conditions(TestFixture::boundary_condition_data());
  boundary_conditions.set_boundary_condition(
    BoundaryConditionType::subsonic_non_reflecting_pressure_outflow, boundary_functions);

  const std::vector<OutletFace<dim, n_species>> outlet_faces =
    discretization.evaluate_outlet(boundary_conditions, this->material);

  // The permuted setup with x' = P x, where P maps x_{i+1} to x'_i.
  dealii::Tensor<2, dim> permutation;
  for (unsigned int i = 0; i < dim; ++i)
    permutation[i][(i + 1) % dim] = 1.;
  const dealii::Point<dim> permuted_upper_corner(permutation * upper_corner);

  const OutletDiscretization<dim, n_species> permuted_discretization(
    permuted_upper_corner,
    std::vector<unsigned int>(dim, 1),
    TestFixture::quadrature_rule,
    ConservedState<dim, n_species>(permutation, no_shift));

  std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
    permuted_boundary_functions;
  for (unsigned int boundary_id = 0; boundary_id < 2 * dim; ++boundary_id)
    permuted_boundary_functions[boundary_id] =
      std::make_shared<TargetPressure<dim>>(permutation, no_shift);
  BoundaryConditions<dim, double> permuted_boundary_conditions(
    TestFixture::boundary_condition_data());
  permuted_boundary_conditions.set_boundary_condition(
    BoundaryConditionType::subsonic_non_reflecting_pressure_outflow, permuted_boundary_functions);

  const std::vector<OutletFace<dim, n_species>> permuted_outlet_faces =
    permuted_discretization.evaluate_outlet(permuted_boundary_conditions, this->material);
  ASSERT_EQ(permuted_outlet_faces.size(), outlet_faces.size());

  for (const OutletFace<dim, n_species> &permuted_face : permuted_outlet_faces)
    for (unsigned int q = 0; q < permuted_face.locations.size(); ++q)
      {
        SCOPED_TRACE(::testing::Message()
                     << "permuted face batch " << permuted_face.face_batch << ", q " << q);

        // With Gauss-Lobatto quadrature, the points on the edges of a face are also points of the
        // neighboring faces, so the original point is identified by its location and its normal.
        const dealii::Point<dim>     original_point(transpose(permutation) *
                                                permuted_face.locations[q]);
        const dealii::Tensor<1, dim> original_normal =
          transpose(permutation) * permuted_face.normals[q];

        bool found = false;
        for (const OutletFace<dim, n_species> &outlet_face : outlet_faces)
          for (unsigned int p = 0; p < outlet_face.locations.size(); ++p)
            if (outlet_face.locations[p].distance(original_point) < 1e-12 and
                (outlet_face.normals[p] - original_normal).norm() < 1e-12)
              {
                found = true;

                const ConservedVariablesType<dim, double, n_species, double> &original_value =
                  outlet_face.exterior_values[p];
                ConservedVariablesType<dim, double, n_species, double> expected_value =
                  original_value;
                for (unsigned int i = 0; i < dim; ++i)
                  {
                    expected_value[1 + i] = 0.;
                    for (unsigned int j = 0; j < dim; ++j)
                      expected_value[1 + i] += permutation[i][j] * original_value[1 + j];
                  }

                for (unsigned int c = 0; c < n_components; ++c)
                  EXPECT_NEAR(permuted_face.exterior_values[q][c],
                              expected_value[c],
                              tolerance * std::max(1., std::abs(expected_value[c])))
                    << "component " << c;
              }
        EXPECT_TRUE(found);
      }
}
