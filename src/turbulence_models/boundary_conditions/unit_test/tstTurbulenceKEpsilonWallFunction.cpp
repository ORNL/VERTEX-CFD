#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceKEpsilonWallFunction.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _prod_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dest_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _source_k;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _grad_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _grad_e;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu;

    const bool _low_k;
    const double _nanval;

    Dependencies(const panzer::IntegrationRule& ir, const bool low_k)
        : _k("turb_kinetic_energy", ir.dl_scalar)
        , _e("turb_dissipation_rate", ir.dl_scalar)
        , _vel_0("velocity_0", ir.dl_scalar)
        , _vel_1("velocity_1", ir.dl_scalar)
        , _vel_2("velocity_2", ir.dl_scalar)
        , _prod_k("PRODUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
        , _dest_k("DESTRUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
        , _source_k("SOURCE_turb_kinetic_energy_equation", ir.dl_scalar)
        , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
        , _grad_k("GRAD_turb_kinetic_energy", ir.dl_vector)
        , _grad_e("GRAD_turb_dissipation_rate", ir.dl_vector)
        , _normals("Side Normal", ir.dl_vector)
        , _nu("kinematic_viscosity", ir.dl_scalar)
        , _low_k(low_k)
        , _nanval(std::numeric_limits<double>::quiet_NaN())
    {
        this->addEvaluatedField(_k);
        this->addEvaluatedField(_e);
        this->addEvaluatedField(_vel_0);
        this->addEvaluatedField(_vel_1);
        this->addEvaluatedField(_vel_2);
        this->addEvaluatedField(_prod_k);
        this->addEvaluatedField(_dest_k);
        this->addEvaluatedField(_source_k);
        this->addEvaluatedField(_nu_t);
        this->addEvaluatedField(_grad_k);
        this->addEvaluatedField(_grad_e);
        this->addEvaluatedField(_normals);
        this->addEvaluatedField(_nu);
        this->setName(
            "Turbulence Model K-Epsilon Wall Function Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "turbulence k-epsilon wall function test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        using std::pow;

        const int num_point = _k.extent(1);
        const int num_space_dim = _normals.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            _k(c, qp) = _low_k ? 0.25 : 2.5;
            _e(c, qp) = 3.1;
            _vel_0(c, qp) = 1.5;
            _vel_1(c, qp) = -3.0;
            _vel_2(c, qp) = num_space_dim == 3 ? 4.5 : _nanval;

            _prod_k(c, qp) = 7.5;
            _dest_k(c, qp) = -2.2;
            _source_k(c, qp) = 5.3;

            _nu_t(c, qp) = 4.8;
            _nu(c, qp) = 0.0001;

            for (int d = 0; d < num_space_dim; ++d)
            {
                _grad_k(c, qp, d) = 0.02 * (d + 1.0);
                _grad_e(c, qp, d) = 0.03 * (d + 1.0);
                _normals(c, qp, d) = 0.33 * (d + 1.0);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool low_k, const bool neumann)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create boundary parameters
    Teuchos::ParameterList bc_params;
    bc_params.set("Epsilon Condition Type", neumann ? "Neumann" : "Dirichlet");

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir, low_k));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create wall function evaluator
    const auto wall_eval = Teuchos::rcp(
        new BoundaryCondition::TurbulenceKEpsilonWallFunction<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
            *test_fixture.ir, bc_params));
    test_fixture.registerEvaluator<EvalType>(wall_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_k);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_e);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_grad_k);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_grad_e);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_u_tau);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_y_plus);
    test_fixture.registerTestField<EvalType>(wall_eval->_wall_func_nu_t);

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto boundary_k_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_k);
    const auto boundary_e_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_e);
    const auto boundary_grad_k_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_grad_k);
    const auto boundary_grad_e_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_grad_e);
    const auto boundary_u_tau_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_u_tau);
    const auto boundary_y_plus_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_y_plus);
    const auto wall_func_nu_t_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_wall_func_nu_t);

    // Set expected values (see doc script)
    const double exp_k = low_k ? 0.25 : 2.5;

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    const double exp_grad_k_2D[3] = {0.00911, 0.01822, nan_val};
    const double exp_grad_k_3D[3] = {-0.010492, -0.020984, -0.031476};
    const auto exp_grad_k = (num_space_dim == 3) ? exp_grad_k_3D
                                                 : exp_grad_k_2D;

    // Set expected epsilon boundary value for Dirichlet condition (see doc
    // file)
    const double exp_e_dir_2D = low_k ? 18.65286527037969 : 1240.4622237904111;
    const double exp_e_dir_3D = low_k ? 146.23846371977686 : 1240.4622237904111;
    const double exp_e_dir = num_space_dim == 3 ? exp_e_dir_3D : exp_e_dir_2D;

    // Expected epsilon boundary value for Neumann condition is interior value
    const double exp_e_neu = 3.1;

    // Set expected boundary epsilon based on condition type
    const double exp_e = neumann ? exp_e_neu : exp_e_dir;

    // Expected boundary epsilon gradient equal to interior gradient for
    // Dirichlet condition
    const double exp_grad_e_dir[3]
        = {0.03, 0.06, num_space_dim == 3 ? 0.09 : nan_val};

    // Expected boundary gradient calculated for Neumann condition (see doc
    // file)
    const double exp_grad_e_2D[3]
        = {320533.19035428844, 641066.3807085769, nan_val};
    const double exp_grad_e_2D_low_k[3]
        = {1687.8312232028663, 3375.6624464057327, nan_val};
    const double exp_grad_e_3D[3]
        = {320533.1609512885, 641066.321902577, 961599.4828538651};
    const double exp_grad_e_3D_low_k[3]
        = {22142.174555921218, 44284.349111842435, 66426.52366776364};
    const auto exp_grad_e_neu
        = (num_space_dim == 3) ? (low_k ? exp_grad_e_3D_low_k : exp_grad_e_3D)
                               : (low_k ? exp_grad_e_2D_low_k : exp_grad_e_2D);

    // Set expected epsilon values based on condition type
    const auto exp_grad_e = neumann ? exp_grad_e_neu : exp_grad_e_dir;

    // Set other boundary turbulence quantities
    const auto exp_u_tau_2D = low_k ? 0.3032641922468069 : 0.8660254037844386;
    const auto exp_u_tau_3D = low_k ? 0.507458054264097 : 0.8660254037844386;
    const auto exp_u_tau = num_space_dim == 3 ? exp_u_tau_3D : exp_u_tau_2D;
    const auto exp_y_plus = 11.06;
    const auto neu_nu_t = 0.00045346000000000004;
    const auto dir_nu_t = low_k ? 0.001814516129032258 : 0.1814516129032258;
    const auto exp_wall_func_nu_t = neumann ? neu_nu_t : dir_nu_t;

    // Check boundary values
    const int num_point = boundary_k_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_k, fieldValue(boundary_k_result, 0, qp));
        EXPECT_DOUBLE_EQ(exp_e, fieldValue(boundary_e_result, 0, qp));

        for (int d = 0; d < num_space_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(exp_grad_k[d],
                             fieldValue(boundary_grad_k_result, 0, qp, d));
            EXPECT_DOUBLE_EQ(exp_grad_e[d],
                             fieldValue(boundary_grad_e_result, 0, qp, d));
        }

        EXPECT_DOUBLE_EQ(exp_u_tau, fieldValue(boundary_u_tau_result, 0, qp));
        EXPECT_DOUBLE_EQ(exp_y_plus, fieldValue(boundary_y_plus_result, 0, qp));
        EXPECT_DOUBLE_EQ(exp_wall_func_nu_t,
                         fieldValue(wall_func_nu_t_result, 0, qp));
    }
}

//---------------------------------------------------------------------------//
// 2-D case
TEST(Test2DTurbulenceKEpsilonWallFunctionDirichlet, residual)
{
    testEval<panzer::Traits::Residual, 2>(false, false);
}

TEST(Test2DTurbulenceKEpsilonWallFunctionDirichlet, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false);
}

TEST(Test2DTurbulenceKEpsilonWallFunctionLowKDirichlet, residual)
{
    testEval<panzer::Traits::Residual, 2>(true, false);
}

TEST(Test2DTurbulenceKEpsilonWallFunctionLowKDirichlet, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, false);
}

TEST(Test2DTurbulenceKEpsilonWallFunctionNeumann, residual)
{
    testEval<panzer::Traits::Residual, 2>(false, true);
}

TEST(Test2DTurbulenceKEpsilonWallFunctionNeumann, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true);
}

TEST(Test2DTurbulenceKEpsilonWallFunctionLowKNeumann, residual)
{
    testEval<panzer::Traits::Residual, 2>(true, true);
}

TEST(Test2DTurbulenceKEpsilonWallFunctionLowKNeumann, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, true);
}

//---------------------------------------------------------------------------//
// 3-D case
TEST(Test3DTurbulenceKEpsilonWallFunctionDirichlet, residual)
{
    testEval<panzer::Traits::Residual, 3>(false, false);
}

TEST(Test3DTurbulenceKEpsilonWallFunctionDirichlet, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, false);
}

TEST(Test3DTurbulenceKEpsilonWallFunctionLowKDirichlet, residual)
{
    testEval<panzer::Traits::Residual, 3>(true, false);
}

TEST(Test3DTurbulenceKEpsilonWallFunctionLowKDirichlet, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, false);
}

TEST(Test3DTurbulenceKEpsilonWallFunctionNeumann, residual)
{
    testEval<panzer::Traits::Residual, 3>(false, true);
}

TEST(Test3DTurbulenceKEpsilonWallFunctionNeumann, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, true);
}

TEST(Test3DTurbulenceKEpsilonWallFunctionLowKNeumann, residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true);
}

TEST(Test3DTurbulenceKEpsilonWallFunctionLowKNeumann, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true);
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
