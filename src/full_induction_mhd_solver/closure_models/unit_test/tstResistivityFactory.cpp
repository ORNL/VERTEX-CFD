#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace VertexCFD
{
namespace Test
{

//-----------------------------------------------------------------//
// Full Induction MHD closure model factory test for "Resistivity"
// using the constant scalar field closure model
//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory(const bool variable_resistivity)
{
    static constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "Resistivity";
    test_fixture.eval_name = "Constant Scalar Field \"resistivity\"";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.125)
        .set("Build Resistive Flux", true)
        .set("Resistivity", 1.25)
        .set("Variable Resistivity", variable_resistivity);
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.template buildAndTest<
        ClosureModel::ConstantScalarField<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(ResistivityFactory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>(false);
}

TEST(ResistivityFactory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>(false);
}

TEST(ResistivityFactory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>(false);
}

TEST(ResistivityFactory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>(false);
}

//-----------------------------------------------------------------//
// Full Induction MHD closure model factory exception test for
// "Resistivity" when "Variable Resistivity" is true.
//-----------------------------------------------------------------//
template<class EvalType>
void testFactoryException()
{
    const std::string msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            (testFactory<EvalType, 3>(true));
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::runtime_error);
}

TEST(ResistivityFactoryException, residual_test)
{
    testFactoryException<panzer::Traits::Residual>();
}

TEST(ResistivityFactoryException, jacobian_test)
{
    testFactoryException<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
