#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"
#include "utils/VertexCFD_Utils_PhaseLayout.hpp"
#include "utils/VertexCFD_Utils_ScalarToVector.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFNthPhaseFraction.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _alpha_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _alpha_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : _alpha_1("alpha_pacifico", ir.dl_scalar)
        , _alpha_2("alpha_modelo", ir.dl_scalar)
    {
        this->addEvaluatedField(_alpha_1);
        this->addEvaluatedField(_alpha_2);

        this->setName(
            "Incompressible LSVOF Nth Phase Fraction Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        _alpha_1.deep_copy(0.5);
        _alpha_2.deep_copy(0.3);
    }
};

template<class EvalType>
void testEval()
{
    constexpr int num_space_dim = 3;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));

    test_fixture.registerEvaluator<EvalType>(deps);

    // Phase list
    std::vector<std::string> phase_list;
    phase_list.push_back("alpha_pacifico");
    phase_list.push_back("alpha_modelo");

    // Create combined array of phase fraction fields
    const auto phase_vec
        = Utils::ScalarToVector<EvalType, PhaseIndex>::createFromList(
            ir, "volume_fractions", phase_list, false, false);

    test_fixture.registerEvaluator<EvalType>(phase_vec);

    phase_list.push_back("alpha_corona");

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFNthPhaseFraction<EvalType,
                                                              panzer::Traits>(
            ir, phase_list));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_alpha_n);

    test_fixture.evaluate<EvalType>();

    const auto fc_alpha_n
        = test_fixture.getTestFieldData<EvalType>(eval->_alpha_n);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(0.2, fieldValue(fc_alpha_n, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFNthPhaseFraction, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFNthPhaseFraction, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

} // namespace Test
} // namespace VertexCFD
