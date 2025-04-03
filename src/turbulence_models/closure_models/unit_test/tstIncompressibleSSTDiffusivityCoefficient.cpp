#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSSTDiffusivityCoefficient.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sst_blending_function;

    Dependencies(const panzer::IntegrationRule& ir)
        : _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
        , _sst_blending_function("sst_blending_function", ir.dl_scalar)
    {
        this->addEvaluatedField(_nu_t);
        this->addEvaluatedField(_sst_blending_function);
        this->setName(
            "Incompressible SST Diffusivity Coefficient Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _sst_blending_function.deep_copy(0.75);
        _nu_t.deep_copy(1.5);
    }
};

template<class EvalType>
void testEval()
{
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set turbulent quantities
    const double sigma_k1 = 0.8;
    const double sigma_k2 = 0.1;
    const double sigma_w1 = 0.4;
    const double sigma_w2 = 0.8;
    const double sigma_k = 0.75 * sigma_k1 + 0.25 * sigma_k2;
    const double sigma_w = 0.75 * sigma_w1 + 0.25 * sigma_w2;

    // Create parameter list for user-defined constants
    Teuchos::ParameterList user_params;
    user_params.sublist("Turbulence Parameters")
        .sublist("SST Parameters")
        .set<double>("sigma_k1", sigma_k1)
        .set<double>("sigma_k2", sigma_k2)
        .set<double>("sigma_w1", sigma_w1)
        .set<double>("sigma_w2", sigma_w2);

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Fluid properties
    const double nu = 0.25;
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", nu);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", false);
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleSSTDiffusivityCoefficient<EvalType,
                                                                  panzer::Traits>(
            ir, fluid_prop, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_diffusivity_var_k);
    test_fixture.registerTestField<EvalType>(eval->_diffusivity_var_w);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_var_k
        = test_fixture.getTestFieldData<EvalType>(eval->_diffusivity_var_k);
    const auto fv_var_w
        = test_fixture.getTestFieldData<EvalType>(eval->_diffusivity_var_w);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_diffusivity_var_k = nu + sigma_k * 1.5;
    const double exp_diffusivity_var_w = nu + sigma_w * 1.5;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_diffusivity_var_k, fieldValue(fv_var_k, 0, qp));
        EXPECT_DOUBLE_EQ(exp_diffusivity_var_w, fieldValue(fv_var_w, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTDiffusivityCoefficient, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTDiffusivityCoefficient, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
