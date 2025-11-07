#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_VariableSUPGFlux.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> var_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> tau_supg_var;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        var_diff_flux;

    Dependencies(const panzer::IntegrationRule& ir)
        : velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , dxdt_var("DXDT_test_variable", ir.dl_scalar)
        , var("test_variable", ir.dl_scalar)
        , var_source("SOURCE_test_variable_equation", ir.dl_scalar)
        , tau_supg_var("tau_supg_test_variable", ir.dl_scalar)
        , grad_var("GRAD_test_variable", ir.dl_vector)
        , var_diff_flux("DIFFUSION_FLUX_test_variable_equation", ir.dl_vector)
    {
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);
        this->addEvaluatedField(dxdt_var);
        this->addEvaluatedField(var);
        this->addEvaluatedField(var_source);
        this->addEvaluatedField(tau_supg_var);
        this->addEvaluatedField(grad_var);
        this->addEvaluatedField(var_diff_flux);

        this->setName("Variable SUPG Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "variable supg flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_var.extent(1);
        const int num_space_dim = grad_var.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            velocity_0(c, qp) = 1.5;
            velocity_1(c, qp) = 2.5;
            velocity_2(c, qp) = num_space_dim == 3 ? 3.5 : _nanval;
            dxdt_var(c, qp) = 4.0;
            var(c, qp) = 5.0;
            var_source(c, qp) = 6.0;
            tau_supg_var(c, qp) = 7.0;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                grad_var(c, qp, dim) = 8.0 * (dim + 1) * sign;
                var_diff_flux(c, qp, dim) = 9.0 * (dim + 1) * sign;
            }
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

    auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList closure_params;
    closure_params.set("Field Name", "test_variable");
    closure_params.set("Equation Name", "test_variable_equation");

    auto eval = Teuchos::rcp(
        new ClosureModel::VariableSUPGFlux<EvalType, panzer::Traits, num_space_dim>(
            ir, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_var_diff_flux);
    test_fixture.evaluate<EvalType>();

    const auto fc_var
        = test_fixture.getTestFieldData<EvalType>(eval->_var_diff_flux);

    const int num_point = ir.num_points;

    const double exp_var_flux_2d[2] = {264.0, 473.0};
    const double exp_var_flux_3d[3] = {-618.0, -997.0, -1448.0};
    const double* exp_var_flux = num_space_dim == 3 ? exp_var_flux_3d
                                                    : exp_var_flux_2d;
    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_DOUBLE_EQ(exp_var_flux[dim], fieldValue(fc_var, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(VariableSUPGFlux2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(VariableSUPGFlux2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(VariableSUPGFlux3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(VariableSUPGFlux3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
