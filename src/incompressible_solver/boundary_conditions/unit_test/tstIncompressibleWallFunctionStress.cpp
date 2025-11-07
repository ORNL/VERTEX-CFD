#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleWallFunctionStress.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _u_tau;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _y_plus;

    Dependencies(const panzer::IntegrationRule& ir)
        : _velocity_0("velocity_0", ir.dl_scalar)
        , _velocity_1("velocity_1", ir.dl_scalar)
        , _velocity_2("velocity_2", ir.dl_scalar)
        , _u_tau("BOUNDARY_friction_velocity", ir.dl_scalar)
        , _y_plus("BOUNDARY_y_plus", ir.dl_scalar)
    {
        this->addEvaluatedField(_velocity_0);
        this->addEvaluatedField(_velocity_1);
        this->addEvaluatedField(_velocity_2);
        this->addEvaluatedField(_u_tau);
        this->addEvaluatedField(_y_plus);

        this->setName(
            "Incompressible Wall Function Stress Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        // Set scalar variables
        _velocity_0.deep_copy(2.4);
        _velocity_1.deep_copy(3.6);
        _velocity_2.deep_copy(4.8);
        _u_tau.deep_copy(5.1);
        _y_plus.deep_copy(11.06);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Evaluate dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create wall function stress evaluator.
    const auto wf_eval = Teuchos::rcp(
        new BoundaryCondition::IncompressibleWallFunctionStress<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
            *test_fixture.ir, "test_prefix"));
    test_fixture.registerEvaluator<EvalType>(wf_eval);

    // Add required test fields.
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            wf_eval->_wall_func_shear_stress[dim]);
    }

    test_fixture.evaluate<EvalType>();

    // Get result
    auto wall_func_stress_result = test_fixture.getTestFieldData<EvalType>(
        wf_eval->_wall_func_shear_stress[0]);

    // Create expected stresses
    const double exp_stress[3]
        = {1.1066907775768533, 1.6600361663652801, 2.2133815551537066};

    // Loop over quadrature points and mesh dimension
    const int num_point = wall_func_stress_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int d = 0; d < num_space_dim; ++d)
        {
            wall_func_stress_result = test_fixture.getTestFieldData<EvalType>(
                wf_eval->_wall_func_shear_stress[d]);
            EXPECT_DOUBLE_EQ(exp_stress[d],
                             fieldValue(wall_func_stress_result, 0, qp));
        }
    }
}

//---------------------------------------------------------------------------//
// 2D wall function stress residual
TEST(WallFunctionStress2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

// 2D wall function stress jacobian
TEST(WallFunctionStress2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

// 3D wall function stress residual
TEST(WallFunctionStress3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

// 3D wall function stress jacobian
TEST(WallFunctionStress3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
