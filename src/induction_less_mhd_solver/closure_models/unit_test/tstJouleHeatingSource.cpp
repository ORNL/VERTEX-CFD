#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_JouleHeatingSource.hpp"

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

    int num_space_dim;

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sigma;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _electric_current_density_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _electric_current_density_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _electric_current_density_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : num_space_dim(ir.spatial_dimension)
        , _sigma("electrical_conductivity", ir.dl_scalar)
        , _electric_current_density_0("electric_current_density_0",
                                      ir.dl_scalar)
        , _electric_current_density_1("electric_current_density_1",
                                      ir.dl_scalar)
        , _electric_current_density_2("electric_current_density_2",
                                      ir.dl_scalar)
    {
        this->addEvaluatedField(_sigma);
        this->addEvaluatedField(_electric_current_density_0);
        this->addEvaluatedField(_electric_current_density_1);
        this->addEvaluatedField(_electric_current_density_2);
        this->setName("Joule Heating Source Term Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "joule heating source term test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _electric_current_density_0.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            _sigma(c, qp) = 0.5;
            _electric_current_density_0(c, qp) = 0.1;
            _electric_current_density_1(c, qp) = -0.2;
            _electric_current_density_2(c, qp) = num_space_dim == 3 ? 0.3
                                                                    : _nanval;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::
            JouleHeatingSource<EvalType, panzer::Traits, num_space_dim>(ir));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_jh_source);
    test_fixture.evaluate<EvalType>();
    const auto joule_heating
        = test_fixture.getTestFieldData<EvalType>(eval->_jh_source);

    // Assert values
    const double exp_values = num_space_dim == 3 ? 0.28 : 0.1;

    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_values, fieldValue(joule_heating, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(JouleHeatingSource2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(JouleHeatingSource2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(JouleHeatingSource3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(JouleHeatingSource3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.user_params.set("Build Inductionless MHD Equation", true);
    test_fixture.user_params.set("Build Temperature Equation", true);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0)
        .set("Electrical conductivity", 3.0)
        .set("Thermal conductivity", 0.5)
        .set("Specific heat capacity", 0.6);
    test_fixture.type_name = "JouleHeatingSource";
    test_fixture.eval_name = "Joule Heating Source Term "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::JouleHeatingSource<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(JouleHeatingSource_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(JouleHeatingSource_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
