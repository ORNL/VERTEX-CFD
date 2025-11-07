#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKTauEddyViscosity.hpp"

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
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_kinetic_energy;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        turb_specific_dissipation_rate;

    Dependencies(const panzer::IntegrationRule& ir)
        : turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                         ir.dl_scalar)
    {
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_specific_dissipation_rate);
        this->setName(
            "Incompressible K-Tau Eddy Viscosity Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        turb_kinetic_energy.deep_copy(2.5);
        turb_specific_dissipation_rate.deep_copy(10.0);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleKTauEddyViscosity<EvalType,
                                                          panzer::Traits,
                                                          NumSpaceDim>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_t);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_nu_t = test_fixture.getTestFieldData<EvalType>(eval->_nu_t);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_var = 25.0;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_var, fieldValue(fv_nu_t, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleKTauEddyViscosity2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleKTauEddyViscosity2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleKTauEddyViscosity3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleKTauEddyViscosity3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
