#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace VertexCFD::MHDProperties;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test
{
class FullInductionMHDPropertiesTest : public ::testing::Test
{
  protected:
    std::unique_ptr<FullInductionMHDProperties> mhd_props;

    virtual void SetUp() override;
    virtual void SetUpConstantProps(const bool set_alpha);
    virtual void SetUpVariableResistivity();
};

// Set up default (minimal inputs)
void FullInductionMHDPropertiesTest::SetUp()
{
    const Teuchos::ParameterList mhd_params;
    mhd_props = std::make_unique<FullInductionMHDProperties>(mhd_params);
}

// Set all properties with constant resistivity
void FullInductionMHDPropertiesTest::SetUpConstantProps(const bool set_alpha)
{
    Teuchos::ParameterList mhd_params;
    mhd_params.set("Build Magnetic Correction Potential Equation", true);
    mhd_params.set("Build Resistive Flux", true);
    mhd_params.set("Vacuum Magnetic Permeability", 0.5);
    mhd_params.set("Variable Resistivity", false);
    mhd_params.set("Resistivity", 1.5);
    mhd_params.set("Hyperbolic Divergence Cleaning Speed", 1.8);
    if (set_alpha)
    {
        mhd_params.set("Magnetic Correction Damping Factor", 2.0);
    }
    mhd_props = std::make_unique<FullInductionMHDProperties>(mhd_params);
}

// Set variable resistivity
void FullInductionMHDPropertiesTest::SetUpVariableResistivity()
{
    Teuchos::ParameterList mhd_params;
    mhd_params.set("Build Magnetic Correction Potential Equation", false);
    mhd_params.set("Build Resistive Flux", true);
    mhd_params.set("Variable Resistivity", true);
    mhd_props = std::make_unique<FullInductionMHDProperties>(mhd_params);
}
//---------------------------------------------------------------------------//

// Test default properties
TEST_F(FullInductionMHDPropertiesTest, defaults)
{
    FullInductionMHDPropertiesTest::SetUp();

    EXPECT_FALSE(mhd_props->variableResistivity());

    EXPECT_FALSE(mhd_props->buildMagnCorr());

    EXPECT_FALSE(mhd_props->buildResistiveFlux());

    const double mu_0 = mhd_props->vacuumMagneticPermeability();
    EXPECT_DOUBLE_EQ(1.0, mu_0);

    const double eta = mhd_props->resistivity();
    EXPECT_TRUE(std::isnan(eta));

    const double c_h = mhd_props->hyperbolicDivergenceCleaningSpeed();
    EXPECT_DOUBLE_EQ(0.0, c_h);

    const double alpha = mhd_props->magneticCorrectionDampingFactor();
    EXPECT_DOUBLE_EQ(0.0, alpha);
}

// Test constant properties with derived alpha
TEST_F(FullInductionMHDPropertiesTest, derived_alpha)
{
    FullInductionMHDPropertiesTest::SetUpConstantProps(false);

    EXPECT_FALSE(mhd_props->variableResistivity());

    EXPECT_TRUE(mhd_props->buildMagnCorr());

    EXPECT_TRUE(mhd_props->buildResistiveFlux());

    const double mu_0 = mhd_props->vacuumMagneticPermeability();
    EXPECT_DOUBLE_EQ(0.5, mu_0);

    const double eta = mhd_props->resistivity();
    EXPECT_EQ(1.5, eta);

    const double c_h = mhd_props->hyperbolicDivergenceCleaningSpeed();
    EXPECT_DOUBLE_EQ(1.8, c_h);

    const double alpha = mhd_props->magneticCorrectionDampingFactor();
    EXPECT_DOUBLE_EQ(10.0, alpha);
}

// Test constant properties with input alpha
TEST_F(FullInductionMHDPropertiesTest, set_alpha)
{
    FullInductionMHDPropertiesTest::SetUpConstantProps(true);

    EXPECT_FALSE(mhd_props->variableResistivity());

    EXPECT_TRUE(mhd_props->buildMagnCorr());

    EXPECT_TRUE(mhd_props->buildResistiveFlux());

    const double mu_0 = mhd_props->vacuumMagneticPermeability();
    EXPECT_DOUBLE_EQ(0.5, mu_0);

    const double eta = mhd_props->resistivity();
    EXPECT_EQ(1.5, eta);

    const double c_h = mhd_props->hyperbolicDivergenceCleaningSpeed();
    EXPECT_DOUBLE_EQ(1.8, c_h);

    const double alpha = mhd_props->magneticCorrectionDampingFactor();
    EXPECT_DOUBLE_EQ(2.0, alpha);
}

// Test variable resistivity
TEST_F(FullInductionMHDPropertiesTest, variable_resistivity)
{
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            FullInductionMHDPropertiesTest::SetUpVariableResistivity();
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
}

//---------------------------------------------------------------------------//

} // end namespace Test
