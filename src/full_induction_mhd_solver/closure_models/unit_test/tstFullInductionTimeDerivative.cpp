#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleVariableTimeDerivative.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory(const bool build_magn_corr)
{
    static constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "FullInductionTimeDerivative";
    std::vector<std::string> eval_names;
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        eval_names.push_back("induction_" + std::to_string(dim)
                             + " Incompressible Time Derivative 2D");
    }
    if (build_magn_corr)
    {
        eval_names.push_back(
            "magnetic_correction_potential Incompressible Time Derivative 2D");
    }
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Build Magnetic Correction Potential Equation", build_magn_corr)
        .set("Hyperbolic Divergence Cleaning Speed", 1.1);
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.num_evaluators = eval_names.size();
    for (int ind = 0; ind < test_fixture.num_evaluators; ++ind)
    {
        test_fixture.eval_index = ind;
        test_fixture.eval_name = eval_names[ind];
        test_fixture.template buildAndTest<
            ClosureModel::IncompressibleVariableTimeDerivative<EvalType,
                                                               panzer::Traits>,
            num_space_dim>();
    }
}

TEST(FullInductionTimeDerivative_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>(false);
}

TEST(FullInductionTimeDerivative_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>(false);
}

TEST(FullInductionTimeDerivative_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>(false);
}

TEST(FullInductionTimeDerivative_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>(false);
}

TEST(FullInductionDivCleaningTimeDerivative_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>(true);
}

TEST(FullInductionDivCleaningTimeDerivative_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>(true);
}

TEST(FullInductionDivCleaningTimeDerivative_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>(true);
}

TEST(FullInductionDivCleaningTimeDerivative_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>(true);
}
//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
