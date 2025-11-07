#include "VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include <Panzer_Traits.hpp>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
void testDefaultFixture()
{
    ClosureModelFactoryTestFixture<EvalType> test_fixture;

    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 1.5)
        .set("Artificial compressibility", 0.1);

    EXPECT_EQ("!!! UNDEFINED !!!", test_fixture.type_name);
    EXPECT_EQ("!!! UNDEFINED !!!", test_fixture.eval_name);

    // The default type_name should not match any closure model and throw an
    // exception.
    EXPECT_THROW((test_fixture.template buildAndTest<void, 2>()),
                 std::runtime_error);
}

TEST(ClosureModelFactoryHarness, residual_test)
{
    testDefaultFixture<panzer::Traits::Residual>();
}

TEST(ClosureModelFactoryHarness, jacobian_test)
{
    testDefaultFixture<panzer::Traits::Jacobian>();
}

} // namespace Test
} // namespace VertexCFD
