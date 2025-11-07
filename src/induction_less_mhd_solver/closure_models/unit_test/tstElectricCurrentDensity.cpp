#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricCurrentDensity.hpp"

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

    int num_grad_dim;

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> sigma;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> ext_magn_field_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> ext_magn_field_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> ext_magn_field_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_electric_potential;

    Dependencies(const panzer::IntegrationRule& ir)
        : num_grad_dim(ir.spatial_dimension)
        , sigma("electrical_conductivity", ir.dl_scalar)
        , velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , ext_magn_field_0("external_magnetic_field_0", ir.dl_scalar)
        , ext_magn_field_1("external_magnetic_field_1", ir.dl_scalar)
        , ext_magn_field_2("external_magnetic_field_2", ir.dl_scalar)
        , grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
    {
        this->addEvaluatedField(sigma);
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);
        this->addEvaluatedField(ext_magn_field_0);
        this->addEvaluatedField(ext_magn_field_1);
        this->addEvaluatedField(ext_magn_field_2);
        this->addEvaluatedField(grad_electric_potential);
        this->setName(
            "Electric Potential Electric Current Density Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "electric potential electric current density test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = velocity_0.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            sigma(c, qp) = 6.25;
            velocity_0(c, qp) = 0.5;
            velocity_1(c, qp) = 1.5;
            grad_electric_potential(c, qp, 0) = 0.1;
            grad_electric_potential(c, qp, 1) = -0.2;
            if (num_grad_dim == 3)
            {
                velocity_2(c, qp) = 2.5;
                grad_electric_potential(c, qp, 2) = 0.3;
                ext_magn_field_0(c, qp) = 1.1;
                ext_magn_field_1(c, qp) = 2.0;
                ext_magn_field_2(c, qp) = -0.3;
            }
            else
            {
                velocity_2(c, qp) = _nanval;
                ext_magn_field_0(c, qp) = 0.0;
                ext_magn_field_1(c, qp) = 0.0;
                ext_magn_field_2(c, qp) = 0.3;
            }
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
        new ClosureModel::ElectricCurrentDensity<EvalType,
                                                 panzer::Traits,
                                                 num_space_dim>(ir));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            eval->_electric_current_density[dim]);
    }

    test_fixture.evaluate<EvalType>();

    const int num_point = ir.num_points;

    // Assert values
    double exp_values[num_space_dim];
    if (num_space_dim == 2)
    {
        exp_values[0] = 2.1875;
        exp_values[1] = 0.3125;
    }
    else
    {
        exp_values[0] = -34.6875;
        exp_values[1] = 19.375;
        exp_values[2] = -5.9375;
    }

    for (int qp = 0; qp < num_point; ++qp)
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            const auto Jdim = test_fixture.getTestFieldData<EvalType>(
                eval->_electric_current_density[dim]);
            EXPECT_NEAR(exp_values[dim], fieldValue(Jdim, 0, qp), 1.0e-15);
        }
}

//-----------------------------------------------------------------//
TEST(ElectricCurrentDensity2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(ElectricCurrentDensity2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(ElectricCurrentDensity3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(ElectricCurrentDensity3D, jacobian_test)
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
    test_fixture.user_params.set("Build Temperature Equation", false);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 2.0)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "ElectricCurrentDensity";
    test_fixture.eval_name = "Electric Potential Electric Current Density "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::ElectricCurrentDensity<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(ElectricCurrentDensity_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(ElectricCurrentDensity_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
