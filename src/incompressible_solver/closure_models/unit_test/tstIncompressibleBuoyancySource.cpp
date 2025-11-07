#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleBuoyancySource.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

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

    double _T;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> temperature;

    Dependencies(const panzer::IntegrationRule& ir, const double T)
        : _T(T)
        , temperature("temperature", ir.dl_scalar)
    {
        this->addEvaluatedField(temperature);

        this->setName("Incompressible Buoyancy Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "buoyancy source test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = temperature.extent(1);

        for (int qp = 0; qp < num_point; ++qp)
        {
            temperature(c, qp) = _T;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize class object to test
    const double T = 330.0;
    Teuchos::Array<double> gravity(num_space_dim);
    gravity[0] = 2.0;
    gravity[1] = 3.0;
    if (num_space_dim == 3)
        gravity[2] = 4.0;
    Teuchos::ParameterList user_params;
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", true);
    fluid_prop_list.set("Thermal conductivity", 0.5);
    fluid_prop_list.set("Specific heat capacity", 5.0);
    fluid_prop_list.set("Build Buoyancy Source", true);
    fluid_prop_list.set("Reference temperature", 300.0);
    fluid_prop_list.set("Expansion coefficient", 1.0e-2);
    user_params.set("Build Buoyancy Source", true);
    user_params.set("Gravity", gravity);

    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, T));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleBuoyancySource<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            ir, fluid_prop, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_buoyancy_continuity_source);
    test_fixture.registerTestField<EvalType>(eval->_buoyancy_energy_source);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(
            eval->_buoyancy_momentum_source[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_cont = test_fixture.getTestFieldData<EvalType>(
        eval->_buoyancy_continuity_source);
    const auto fc_energy = test_fixture.getTestFieldData<EvalType>(
        eval->_buoyancy_energy_source);
    const auto fc_mom_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_buoyancy_momentum_source[0]);
    const auto fc_mom_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_buoyancy_momentum_source[1]);

    const int num_point = ir.num_points;

    const double exp_mom_source[3] = {-0.6, -0.9, -1.2};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.0, fieldValue(fc_cont, 0, qp));
        EXPECT_DOUBLE_EQ(0.0, fieldValue(fc_energy, 0, qp));

        EXPECT_DOUBLE_EQ(exp_mom_source[0], fieldValue(fc_mom_0, 0, qp));
        EXPECT_DOUBLE_EQ(exp_mom_source[1], fieldValue(fc_mom_1, 0, qp));
        if (num_space_dim > 2) // 3D mesh
        {
            const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_buoyancy_momentum_source[2]);
            EXPECT_DOUBLE_EQ(exp_mom_source[2], fieldValue(fc_mom_2, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleBuoyancySource2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleBuoyancySource2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleBuoyancySource3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleBuoyancySource3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    const Teuchos::Array<double> gravity(num_space_dim);
    test_fixture.user_params.set("Gravity", gravity);
    test_fixture.user_params.set("Build Temperature Equation", false);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleBuoyancySource";
    test_fixture.eval_name = "Incompressible Buoyancy Source "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleBuoyancySource<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleBuoyancySource_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleBuoyancySource_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleBuoyancySource_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleBuoyancySource_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
