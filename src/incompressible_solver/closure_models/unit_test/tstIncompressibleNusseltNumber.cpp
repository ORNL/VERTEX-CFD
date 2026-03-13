#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleNusseltNumber.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_temperature;

    Dependencies(const panzer::IntegrationRule& ir)
        : normals("Side Normal", ir.dl_vector)
        , grad_temperature("GRAD_temperature", ir.dl_vector)
    {
        this->addEvaluatedField(normals);

        this->addEvaluatedField(grad_temperature);

        this->setName("Incompressible Nusselt Number Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "nusselt number test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_temperature.extent(1);
        const int num_space_dim = grad_temperature.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = (dim % 2 == 0) ? -1 : 1;
                const int dimqp = (dim + 1) * sign;
                grad_temperature(c, qp, dim) = 1.5 * dimqp;
                normals(c, qp, dim) = 0.75 * dimqp;
            }
        }
    }
};

template<class EvalType>
void testEval(const int num_space_dim)
{
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleNusseltNumber<EvalType, panzer::Traits>(
            ir));
    test_fixture.registerEvaluator<EvalType>(eval);

    test_fixture.registerTestField<EvalType>(eval->_nusselt_number);
    test_fixture.evaluate<EvalType>();

    const auto calc_nusselt_number
        = test_fixture.getTestFieldData<EvalType>(eval->_nusselt_number);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_nusselt_number_2d = 5.625;
    const double exp_nusselt_number_3d = 15.75;
    const double exp_nusselt_number
        = num_space_dim == 3 ? exp_nusselt_number_3d : exp_nusselt_number_2d;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_nusselt_number, fieldValue(calc_nusselt_number, 0, qp));
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleNusseltNumber2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleNusseltNumber2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleNusseltNumber3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleNusseltNumber3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleNusseltNumber";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.eval_name = "Incompressible Nusselt Number 2D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleNusseltNumber<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(IncompressibleNusseltNumber_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleNusseltNumber_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
