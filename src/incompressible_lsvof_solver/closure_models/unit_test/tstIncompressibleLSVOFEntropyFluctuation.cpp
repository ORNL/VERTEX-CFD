#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFEntropyFluctuation.hpp"

#include <Panzer_GlobalData.hpp>
#include <Panzer_ParameterLibrary.hpp>
#include <Panzer_ScalarParameterEntry.hpp>
#include <Teuchos_RCP.hpp>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> entropy_function;

    Dependencies(const panzer::IntegrationRule& ir)
        : entropy_function("entropy_function", ir.dl_scalar)

    {
        this->addEvaluatedField(entropy_function);

        this->setName(
            "Incompressible LSVOF Entropy Fluctuation for Scalar Equation "
            "Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF entropy viscosity test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = entropy_function.extent(1);

        for (int qp = 0; qp < num_point; ++qp)
        {
            entropy_function(c, qp) = 0.5;
        }
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

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Domain Volume", 2.0);

    // Entropy function
    auto global_data = panzer::createGlobalData();
    auto& pl = *global_data->pl;
    pl.addParameterFamily("Entropy - entropy_function", true, false);
    Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> entry;
    entry = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
    entry->setValue(4.0);
    pl.addEntry<EvalType>("Entropy - entropy_function", entry);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFEntropyFluctuation<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
            ir, closure_params, "alpha_air_equation", global_data));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_entropy_fluctuation);

    test_fixture.evaluate<EvalType>();

    const auto fc_entropy
        = test_fixture.getTestFieldData<EvalType>(eval->_entropy_fluctuation);

    const int num_point = ir.num_points;

    // Expected values

    double exp_entropy_fluc = 1.5;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_entropy_fluc, fieldValue(fc_entropy, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFEntropyFluctuation2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

TEST(TestLSVOFEntropyFluctuation2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

TEST(TestLSVOFEntropyFluctuation3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

TEST(TestLSVOFEntropyFluctuation3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
