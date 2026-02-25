#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/base/vectorization.h>

#include <meltpooldg/flow/compressible_flow_eos_utils_base.hpp>
#include <meltpooldg/flow/compressible_flow_kernels.hpp>
#include <meltpooldg/flow/compressible_flow_material_data.hpp>
#include <meltpooldg/flow/compressible_flow_types.hpp>

#include "../test_utils/utils.hpp"

constexpr int dim = 3;

template <int dim, typename number>
class EOSMock : public MeltPoolDG::Flow::EOS::EquationOfStateUtils<dim, number>
{
  using VectorizedArrayType = dealii::VectorizedArray<number>;
  using ValueType           = MeltPoolDG::CompressibleFlow::ConservedVariablesType<dim, number>;
  using GradientType = MeltPoolDG::CompressibleFlow::ConservedVariablesGradientType<dim, number>;
  using SingleValueGradientType = dealii::Tensor<1, dim, VectorizedArrayType>;

public:
  MOCK_METHOD(VectorizedArrayType,
              calculate_thermodynamic_pressure,
              (const ValueType &conserved_variables),
              (const, override));
  MOCK_METHOD(SingleValueGradientType,
              calculate_grad_T,
              (const ValueType &conserved_variables, const GradientType &grad_conserved_variables),
              (const, override));
  MOCK_METHOD(VectorizedArrayType,
              calculate_speed_of_sound,
              (const ValueType &conserved_variables),
              (const, override));
  MOCK_METHOD(VectorizedArrayType,
              calculate_temperature,
              (const ValueType &conserved_variables),
              (const, override));
  MOCK_METHOD(ValueType,
              convert_primitive_into_conservative_variables,
              (const ValueType &u_prim),
              (const, override));
  MOCK_METHOD(VectorizedArrayType,
              compute_inner_energy_from_pressure,
              (const VectorizedArrayType &pressure, const VectorizedArrayType &temperature),
              (const, override));

  EOSMock()
  {
    ON_CALL(*this, calculate_thermodynamic_pressure(::testing::_))
      .WillByDefault(testing::Return(VectorizedArrayType(101325.0)));
    ON_CALL(*this, calculate_grad_T(::testing::_, ::testing::_))
      .WillByDefault(testing::Return(dealii::Tensor<1, dim, VectorizedArrayType>()));
    ON_CALL(*this, calculate_speed_of_sound(::testing::_))
      .WillByDefault(testing::Return(VectorizedArrayType(343.0)));
    ON_CALL(*this, calculate_temperature(::testing::_))
      .WillByDefault(testing::Return(VectorizedArrayType(293.1)));
    ON_CALL(*this, convert_primitive_into_conservative_variables(::testing::_))
      .WillByDefault(testing::Return(ValueType()));
    ON_CALL(*this, compute_inner_energy_from_pressure(::testing::_, ::testing::_))
      .WillByDefault(testing::Return(VectorizedArrayType(15000.0)));
  }
};



void
set_material_data(MeltPoolDG::Flow::CompressibleFluidMaterialPhaseData<double> &material_data)
{
  material_data.gamma                 = 1.4;
  material_data.dynamic_viscosity     = 1.8e-5;
  material_data.thermal_conductivity  = 0.0257;
  material_data.reference_density     = 1.293;
  material_data.specific_gas_constant = 287.1;
}

TEST(CompressibleFlowKernelsTest, ConvectiveKernel)
{
  using VectorizedArrayType = dealii::VectorizedArray<double>;
  using FluxType            = MeltPoolDG::CompressibleFlow::FluxType<dim, double>;
  using ValueType           = MeltPoolDG::CompressibleFlow::ConservedVariablesType<dim, double>;

  MeltPoolDG::Flow::CompressibleFluidMaterialPhaseData<double> material_data;
  ::testing::NiceMock<EOSMock<dim, double>>                    eos_mock;

  set_material_data(material_data);

  MeltPoolDG::Flow::CompressibleConvectiveFlux<dim, double> convective_flux(&eos_mock,
                                                                            material_data);

  ValueType conserved_variables;
  conserved_variables[0] = VectorizedArrayType(1.293);
  for (unsigned int i = 1; i < dim + 1; i++)
    conserved_variables[i] = VectorizedArrayType(i + 2) * conserved_variables[0];
  conserved_variables[dim + 1] = VectorizedArrayType(20000.0);

  FluxType flux = convective_flux.flux(conserved_variables);

  {
    SCOPED_TRACE("Density flux computation");
    const typename FluxType::value_type expected_density_flux{
      {VectorizedArrayType(3.879), VectorizedArrayType(5.172), VectorizedArrayType(6.465)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[0], expected_density_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (x-direction)");
    const typename FluxType::value_type expected_momentum_x_flux{
      {VectorizedArrayType(101336.637), VectorizedArrayType(15.516), VectorizedArrayType(19.395)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[1], expected_momentum_x_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (y-direction)");
    const typename FluxType::value_type expected_momentum_y_flux{
      {VectorizedArrayType(15.516), VectorizedArrayType(101345.688), VectorizedArrayType(25.86)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[2], expected_momentum_y_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (z-direction)");
    const typename FluxType::value_type expected_momentum_z_flux{
      {VectorizedArrayType(19.395), VectorizedArrayType(25.86), VectorizedArrayType(101357.325)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[3], expected_momentum_z_flux);
  }

  {
    SCOPED_TRACE("Energy flux computation");
    const typename FluxType::value_type expected_energy_flux{
      {VectorizedArrayType(363975.), VectorizedArrayType(485300.), VectorizedArrayType(606625.)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[dim + 1], expected_energy_flux);
  }

  {
    SCOPED_TRACE("Lambda computation");
    const VectorizedArrayType lambda =
      convective_flux.lambda(conserved_variables, 2. * conserved_variables);
    VectorizedArrayType expected_lambda(171.53643927748996);
    MeltPoolDG::TestUtils::expect_double_eq(lambda, expected_lambda);
  }
}

TEST(CompressibleFlowKernelsTest, ViscousKernel)
{
  using VectorizedArrayType = dealii::VectorizedArray<double>;
  using FluxType            = MeltPoolDG::CompressibleFlow::FluxType<dim, double>;
  using ValueType           = MeltPoolDG::CompressibleFlow::ConservedVariablesType<dim, double>;
  using GradientType = MeltPoolDG::CompressibleFlow::ConservedVariablesGradientType<dim, double>;

  MeltPoolDG::Flow::CompressibleFluidMaterialPhaseData<double> material_data;
  ::testing::NiceMock<EOSMock<dim, double>>                    eos_mock;

  set_material_data(material_data);

  MeltPoolDG::Flow::CompressibleDiffusiveFlux<dim, double> viscous_flux(&eos_mock, material_data);

  ValueType conserved_variables;
  conserved_variables[0] = VectorizedArrayType(1.293);
  for (unsigned int i = 1; i < dim + 1; i++)
    conserved_variables[i] = VectorizedArrayType(i + 2) * conserved_variables[0];
  conserved_variables[dim + 1] = VectorizedArrayType(20000.0);

  GradientType grad_conserved_variables;
  for (unsigned int i = 0; i < dim + 2; i++)
    for (unsigned int d = 0; d < dim; d++)
      grad_conserved_variables[i][d] = VectorizedArrayType(0.1 * (i + 1) * (d + 1));

  FluxType flux = viscous_flux.flux(conserved_variables, grad_conserved_variables);

  {
    SCOPED_TRACE("Density flux computation");
    const typename FluxType::value_type expected_density_flux{
      {VectorizedArrayType(0.), VectorizedArrayType(0.), VectorizedArrayType(0.)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[0], expected_density_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (x-direction)");
    const typename FluxType::value_type expected_momentum_x_flux{
      {VectorizedArrayType(2.7842227378190243e-06),
       VectorizedArrayType(-4.176334106728539e-06),
       VectorizedArrayType(-5.5684454756380519e-06)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[1], expected_momentum_x_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (y-direction)");
    const typename FluxType::value_type expected_momentum_y_flux{
      {VectorizedArrayType(-4.176334106728539e-06),
       VectorizedArrayType(8.4703294725430034e-22),
       VectorizedArrayType(-6.9605568445475641e-06)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[2], expected_momentum_y_flux);
  }

  {
    SCOPED_TRACE("Momentum flux computation (z-direction)");
    const typename FluxType::value_type expected_momentum_z_flux{
      {VectorizedArrayType(-5.5684454756380519e-06),
       VectorizedArrayType(-6.9605568445475641e-06),
       VectorizedArrayType(-2.7842227378190268e-06)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[3], expected_momentum_z_flux);
  }

  {
    SCOPED_TRACE("Energy flux computation");
    const typename FluxType::value_type expected_energy_flux{
      {VectorizedArrayType(-3.6194895591647344e-05),
       VectorizedArrayType(-4.7331786542923428e-05),
       VectorizedArrayType(-5.8468677494199545e-05)}};
    MeltPoolDG::TestUtils::expect_double_eq(flux[dim + 1], expected_energy_flux);
  }
}