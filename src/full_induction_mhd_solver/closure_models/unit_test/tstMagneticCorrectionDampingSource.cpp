#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticCorrectionDampingSource.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        scalar_magnetic_potential;

    Dependencies(const panzer::IntegrationRule& ir)
        : scalar_magnetic_potential("scalar_magnetic_potential", ir.dl_scalar)
    {
        this->addEvaluatedField(scalar_magnetic_potential);

        this->setName(
            "Magnetic Correction Damping Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        scalar_magnetic_potential.deep_copy(0.3);
    }
};

template<class EvalType>
void testEval(const bool use_default_alpha)
{
    static constexpr int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    const double alpha = 2.0;
    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Build Magnetic Correction Potential Equation",
                              true);
    if (use_default_alpha)
    {
        full_induction_params.set("Hyperbolic Divergence Cleaning Speed",
                                  alpha * 0.18);
    }
    else
    {
        full_induction_params.set("Hyperbolic Divergence Cleaning Speed", 1.1);
        full_induction_params.set("Magnetic Correction Damping Factor", alpha);
    }
    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);

    // Initialize class object to test
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::MagneticCorrectionDampingSource<EvalType,
                                                          panzer::Traits>(
            ir, mhd_props));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_damping_potential_source);

    test_fixture.evaluate<EvalType>();

    const auto src_magn_pot = test_fixture.getTestFieldData<EvalType>(
        eval->_damping_potential_source);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(-0.6, fieldValue(src_magn_pot, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(MagneticCorrectionDampingSource, Residual)
{
    testEval<panzer::Traits::Residual>(false);
}

//-----------------------------------------------------------------//
TEST(MagneticCorrectionDampingSource, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(false);
}

//-----------------------------------------------------------------//
TEST(MagneticCorrectionDampingSourceDefaultAlpha, Residual)
{
    testEval<panzer::Traits::Residual>(true);
}

//-----------------------------------------------------------------//
TEST(MagneticCorrectionDampingSourceDefaultAlpha, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", false)
        .set("Hyperbolic Divergence Cleaning Speed", 1.0);
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.type_name = "MagneticCorrectionDampingSource";
    test_fixture.eval_name = "Magnetic Correction Damping Source";
    test_fixture.template buildAndTest<
        ClosureModel::MagneticCorrectionDampingSource<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(MagneticCorrectionDampingSourceFactory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(MagneticCorrectionDampingSourceFactory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(MagneticCorrectionDampingSourceFactory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(MagneticCorrectionDampingSourceFactory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
