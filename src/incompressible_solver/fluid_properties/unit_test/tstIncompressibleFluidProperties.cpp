#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/fluid_properties/VertexCFD_Closure_IncompressibleFluidProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType>
void testEval(const bool unscaled_density,
              const bool solve_temp,
              const bool solve_mhd)
{
    const int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize class object to test
    Teuchos::ParameterList fluid_prop_list;
    if (unscaled_density)
        fluid_prop_list.set("Density", 1.1);
    fluid_prop_list.set("Kinematic viscosity", 1.2);
    if (solve_temp)
    {
        fluid_prop_list.set("Thermal conductivity", 1.3);
        fluid_prop_list.set("Specific heat capacity", 1.4);
    }
    if (solve_mhd)
        fluid_prop_list.set("Electrical conductivity", 1.5);

    auto eval = Teuchos::rcp(
        new FluidProperties::IncompressibleFluidProperties<EvalType,
                                                           panzer::Traits>(
            ir, fluid_prop_list));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_density);
    test_fixture.registerTestField<EvalType>(eval->_kinematic_viscosity);
    if (solve_temp)
    {
        test_fixture.registerTestField<EvalType>(eval->_thermal_conductivity);
        test_fixture.registerTestField<EvalType>(eval->_specific_heat_capacity);
    }
    if (solve_mhd)
        test_fixture.registerTestField<EvalType>(
            eval->_electrical_conductivity);

    test_fixture.evaluate<EvalType>();

    const auto rho = test_fixture.getTestFieldData<EvalType>(eval->_density);
    const auto nu
        = test_fixture.getTestFieldData<EvalType>(eval->_kinematic_viscosity);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(unscaled_density ? 1.1 : 1.0, fieldValue(rho, 0, qp));
        EXPECT_EQ(1.2, fieldValue(nu, 0, qp));
        if (solve_temp)
        {
            const auto k = test_fixture.getTestFieldData<EvalType>(
                eval->_thermal_conductivity);
            const auto cp = test_fixture.getTestFieldData<EvalType>(
                eval->_specific_heat_capacity);
            EXPECT_EQ(1.3, fieldValue(k, 0, qp));
            EXPECT_EQ(1.4, fieldValue(cp, 0, qp));
        }
        if (solve_mhd)
        {
            const auto sigma = test_fixture.getTestFieldData<EvalType>(
                eval->_electrical_conductivity);
            EXPECT_EQ(1.5, fieldValue(sigma, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleFluidPropertiesUnscaledIsothermalNS, Residual)
{
    testEval<panzer::Traits::Residual>(false, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleFluidPropertiesUnscaledIsothermalNS, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(false, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleFluidPropertiesHeatedNS, Residual)
{
    testEval<panzer::Traits::Residual>(true, true, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleFluidPropertiesHeatedNS, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true, true, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleFluidPropertiesMHDNS, Residual)
{
    testEval<panzer::Traits::Residual>(true, false, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleFluidPropertiesMHDNS, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true, false, true);
}

} // namespace Test
} // namespace VertexCFD
