#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

using namespace VertexCFD::FluidProperties;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test
{
class IncompressibleConstantFluidPropertiesTest : public ::testing::Test
{
  protected:
    std::unique_ptr<ConstantFluidProperties> cfp;

    virtual void SetUpTemperature();
    virtual void SetUpBuoyancy();
};

// With temperature equation
void IncompressibleConstantFluidPropertiesTest::SetUpTemperature()
{
    Teuchos::ParameterList cfp_params;
    cfp_params.set("Artificial compressibility", 2.0);
    cfp_params.set("Build Temperature Equation", true);
    cfp_params.set("Heat Capacity Ratio", 1.4);
    cfp = std::make_unique<ConstantFluidProperties>(cfp_params);
}

// With buoyancy
void IncompressibleConstantFluidPropertiesTest::SetUpBuoyancy()
{
    Teuchos::ParameterList cfp_params;
    cfp_params.set("Artificial compressibility", 2.0);
    cfp_params.set("Build Temperature Equation", true);
    cfp_params.set("Build Buoyancy Source", true);
    cfp_params.set("Expansion coefficient", 0.5);
    cfp_params.set("Reference temperature", 0.6);
    cfp = std::make_unique<ConstantFluidProperties>(cfp_params);
}

// Temperature equation
TEST_F(IncompressibleConstantFluidPropertiesTest, temperature_equation)
{
    IncompressibleConstantFluidPropertiesTest::SetUpTemperature();

    const double beta_expect = 2.0;
    const double beta = cfp->artificialCompressibility();
    EXPECT_DOUBLE_EQ(beta_expect, beta);

    const double gamma_expect = 1.4;
    const double gamma = cfp->constantHeatCapacityRatio();
    EXPECT_DOUBLE_EQ(gamma_expect, gamma);
}

// Buoyancy
TEST_F(IncompressibleConstantFluidPropertiesTest, bouyancy)
{
    IncompressibleConstantFluidPropertiesTest::SetUpBuoyancy();
    const double beta_expect = 0.5;
    const double beta = cfp->expansionCoefficient();
    EXPECT_DOUBLE_EQ(beta_expect, beta);

    const double T0_expect = 0.6;
    const double T0 = cfp->referenceTemperature();
    EXPECT_DOUBLE_EQ(T0_expect, T0);
}
//---------------------------------------------------------------------------//

} // end namespace Test
