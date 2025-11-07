#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_VectorFieldDivergence.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConstantSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticCorrectionDampingSource.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

#include <string>

namespace VertexCFD
{
namespace Test
{

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory(const std::string& cm_name,
                 const bool build_magn_corr,
                 const bool build_resistive)
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", build_magn_corr)
        .set("Build Resistive Flux", build_resistive)
        .set("Resistivity", 1.0)
        .set("Hyperbolic Divergence Cleaning Speed", 1.0);
    test_fixture.factory_type = "Solid Full Induction";

    // solid full induction factory adds evaluators by default, including
    // external (and total) magnetic field, so user_params must contain
    // external magnetic field entry.
    test_fixture.user_params.set("External Magnetic Field Value",
                                 Teuchos::Array<double>({0, 0, 0}));

    // num_evaluators and index depend on the count of default evaluators
    test_fixture.num_evaluators = 5 + num_space_dim;
    if (build_magn_corr)
        test_fixture.num_evaluators += 2;
    if (build_resistive)
        test_fixture.num_evaluators += 3;
    test_fixture.eval_index = test_fixture.num_evaluators - 1;

    test_fixture.type_name = cm_name;
    if (cm_name == "InductionConstantSource")
    {
        test_fixture.eval_name = "Induction Constant Source "
                                 + std::to_string(num_space_dim) + "D";
        test_fixture.model_params.set("Induction Source",
                                      Teuchos::Array<double>({0, 0, 0}));
        test_fixture.template buildAndTest<
            ClosureModel::InductionConstantSource<EvalType, panzer::Traits, num_space_dim>,
            num_space_dim>();
    }
    else if (cm_name == "MagneticCorrectionDampingSource")
    {
        test_fixture.eval_name = "Magnetic Correction Damping Source";
        test_fixture.template buildAndTest<
            ClosureModel::MagneticCorrectionDampingSource<EvalType, panzer::Traits>,
            num_space_dim>();
    }
    else if (cm_name == "VectorFieldDivergence")
    {
        test_fixture.eval_name = "Vector Field Divergence 2D";
        test_fixture.model_params.set("Field Names", "foo");
        test_fixture.template buildAndTest<
            ClosureModel::VectorFieldDivergence<EvalType, panzer::Traits, num_space_dim>,
            num_space_dim>();
    }
}

//-----------------------------------------------------------------//
struct SolidFullInductionFactoryTestParams
{
    std::string cm_name;
    bool build_magn_corr;
    bool build_resistive;
};

class SolidFullInductionFactory
    : public testing::TestWithParam<std::tuple<std::string, bool, bool>>
{
  public:
    struct PrintToStringParamName
    {
        std::string operator()(
            const testing::TestParamInfo<std::tuple<std::string, bool, bool>>&
                info) const
        {
            SolidFullInductionFactoryTestParams test_param;
            test_param.cm_name = std::get<0>(info.param);
            test_param.build_magn_corr = std::get<1>(info.param);
            test_param.build_resistive = std::get<2>(info.param);
            std::string test_name = test_param.build_magn_corr ? "Cleaning"
                                                               : "NoCleaning";
            test_name += test_param.build_resistive ? "Resistive" : "Ideal";
            return test_param.cm_name + test_name;
        }
    };
};

//-----------------------------------------------------------------//
TEST_P(SolidFullInductionFactory, Residual2D)
{
    auto params = GetParam();
    testFactory<panzer::Traits::Residual, 2>(
        std::get<0>(params), std::get<1>(params), std::get<2>(params));
}

TEST_P(SolidFullInductionFactory, Jacobian2D)
{
    auto params = GetParam();
    testFactory<panzer::Traits::Jacobian, 2>(
        std::get<0>(params), std::get<1>(params), std::get<2>(params));
}

//-----------------------------------------------------------------//
TEST_P(SolidFullInductionFactory, Residual3D)
{
    auto params = GetParam();
    testFactory<panzer::Traits::Residual, 3>(
        std::get<0>(params), std::get<1>(params), std::get<2>(params));
}

TEST_P(SolidFullInductionFactory, Jacobian3D)
{
    auto params = GetParam();
    testFactory<panzer::Traits::Jacobian, 3>(
        std::get<0>(params), std::get<1>(params), std::get<2>(params));
}

//-----------------------------------------------------------------//
INSTANTIATE_TEST_SUITE_P(TEST,
                         SolidFullInductionFactory,
                         testing::Combine(testing::Values("InductionConstantSo"
                                                          "urce",
                                                          "MagneticCorrectionD"
                                                          "ampingSource",
                                                          "VectorFieldDivergen"
                                                          "ce"),
                                          testing::Bool(),
                                          testing::Bool()),
                         SolidFullInductionFactory::PrintToStringParamName());

//-----------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
