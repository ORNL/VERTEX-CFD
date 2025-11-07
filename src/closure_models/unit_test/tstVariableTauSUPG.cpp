#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_VariableTauSUPG.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
// Tau model
enum class TauModel
{
    Steady,
    Transient,
    NoSUPG,
};

//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> ele_lngth;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> diffusivity;

    Dependencies(const panzer::IntegrationRule& ir)
        : velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , ele_lngth("element_length", ir.dl_vector)
        , diffusivity("diffusivity_test_variable", ir.dl_scalar)
    {
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);
        this->addEvaluatedField(ele_lngth);
        this->addEvaluatedField(diffusivity);

        this->setName("Variable Tau SUPG Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "variable tau supg test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = ele_lngth.extent(1);
        const int num_space_dim = ele_lngth.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            velocity_0(c, qp) = 0.5;
            velocity_1(c, qp) = 1.5;
            velocity_2(c, qp) = num_space_dim == 3 ? 2.5 : _nanval;
            diffusivity(c, qp) = 1.8;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                ele_lngth(c, qp, dim) = 3.0 * (dim + 1);
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const TauModel tau_model)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList closure_params;
    closure_params.set("Field Name", "test_variable");
    double tau_supg_exp;
    Teuchos::ParameterList supg_params;

    // Temperature equation
    if (tau_model == TauModel::Steady)
    {
        supg_params.set("Tau model", "Steady");
        supg_params.set("SUPG coefficient", 0.2);
        tau_supg_exp = 0.26004502055758866;
    }
    else if (tau_model == TauModel::Transient)
    {
        test_fixture.setStepSize(0.2);
        supg_params.set("Tau model", "Transient");
        tau_supg_exp = 0.04982612597850825;
    }
    else
    {
        supg_params.set("Tau model", "NoSUPG");
        tau_supg_exp = 0.0;
    }

    auto eval = Teuchos::rcp(
        new ClosureModel::VariableTauSUPG<EvalType, panzer::Traits, num_space_dim>(
            ir, closure_params, supg_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_tau_supg);
    test_fixture.evaluate<EvalType>();

    const int num_point = ir.num_points;
    const auto tau_supg
        = test_fixture.getTestFieldData<EvalType>(eval->_tau_supg);

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(tau_supg_exp, fieldValue(tau_supg, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(VariableTauSUPGSteady2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(TauModel::Steady);
}

//-----------------------------------------------------------------//
TEST(VariableTauSUPGSteady2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(TauModel::Steady);
}

//-----------------------------------------------------------------//
TEST(VariableTauSUPGTransient3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(TauModel::Transient);
}

//-----------------------------------------------------------------//
TEST(VariableTauSUPGTransient3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(TauModel::Transient);
}

//-----------------------------------------------------------------//
TEST(VariableTauSUPGNoSUPG2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(TauModel::NoSUPG);
}

//-----------------------------------------------------------------//
TEST(VariableTauSUPGNoSUPG2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(TauModel::NoSUPG);
}

} // namespace Test
} // namespace VertexCFD
