#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_lsvof_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleLSVOFNoSlip.hpp"

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"
#include "utils/VertexCFD_Utils_PhaseLayout.hpp"
#include "utils/VertexCFD_Utils_ScalarToVector.hpp"

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
// Continuity Equation cases
enum class ContinuityModel
{
    AC,
    EDAC
};
//---------------------------------------------------------------------------//
// LSVOF model types
enum class LSVOFModel
{
    VOF,
    CLS
};
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _phi, _u_0, _u_1, _u_2;
    ContinuityModel _continuity_model;
    LSVOFModel _lsvof_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _alpha_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _alpha_1;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _level_set;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;

    Dependencies(const panzer::IntegrationRule& ir,
                 double phi,
                 double u_0,
                 double u_1,
                 double u_2,
                 const ContinuityModel continuity_model,
                 const LSVOFModel lsvof_model)
        : _phi(phi)
        , _u_0(u_0)
        , _u_1(u_1)
        , _u_2(u_2)
        , _continuity_model(continuity_model)
        , _lsvof_model(lsvof_model)
        , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _velocity_0("velocity_0", ir.dl_scalar)
        , _velocity_1("velocity_1", ir.dl_scalar)
        , _velocity_2("velocity_2", ir.dl_scalar)
        , _alpha_0("alpha_salt", ir.dl_scalar)
        , _alpha_1("alpha_lime", ir.dl_scalar)
        , _level_set("level_set", ir.dl_scalar)
        , _normals("Side Normal", ir.dl_vector)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        this->addEvaluatedField(_velocity_0);
        this->addEvaluatedField(_velocity_1);
        this->addEvaluatedField(_velocity_2);

        if (_lsvof_model == LSVOFModel::VOF)
        {
            this->addEvaluatedField(_alpha_0);
            this->addEvaluatedField(_alpha_1);
        }
        else if (_lsvof_model == LSVOFModel::CLS)
        {
            this->addEvaluatedField(_level_set);
        }

        this->addEvaluatedField(_normals);

        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);
        if (_continuity_model == ContinuityModel::EDAC)
            this->addEvaluatedField(_grad_lagrange_pressure);

        this->setName("IncompressibleLSVOF NoSlip Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        // Set scalar variables
        _lagrange_pressure.deep_copy(_phi);
        _velocity_0.deep_copy(_u_0);
        _velocity_1.deep_copy(_u_1);
        _velocity_2.deep_copy(_u_2);
        if (_lsvof_model == LSVOFModel::VOF)
        {
            _alpha_0.deep_copy(0.123);
            _alpha_1.deep_copy(0.456);
        }
        else if (_lsvof_model == LSVOFModel::CLS)
        {
            _level_set.deep_copy(42.1);
        }

        Kokkos::parallel_for(
            "incompressible lsvof no slip test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int c) const
    {
        const int num_point = _lagrange_pressure.extent(1);
        const int num_space_dim = _grad_velocity_0.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            // Set gradient and normal vectors
            for (int d = 0; d < num_space_dim; ++d)
            {
                const int sign = pow(-1, d + 1);
                const int dimqp = (d + 1) * sign;

                _normals(c, qp, d) = (_u_0 + _u_1) * dimqp;

                _grad_velocity_0(c, qp, d) = _u_0 * dimqp;
                _grad_velocity_1(c, qp, d) = _u_1 * dimqp;
                _grad_velocity_2(c, qp, d) = _u_2 * dimqp;

                if (_continuity_model == ContinuityModel::EDAC)
                    _grad_lagrange_pressure(c, qp, d) = (_u_0 + _u_1) * dimqp;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const ContinuityModel continuity_model,
              const LSVOFModel lsvof_model)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    std::string continuity_model_name = "";
    switch (continuity_model)
    {
        case (ContinuityModel::AC):
            continuity_model_name = "AC";
            break;
        case (ContinuityModel::EDAC):
            continuity_model_name = "EDAC";
            break;
    }

    // Initialize values and create dependencies
    const int num_lsvof_dofs = 2;
    const double phi = 1.5;
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim == 3 ? 0.125 : nan_val;
    const auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, phi, u, v, w, continuity_model, lsvof_model));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create phase array
    std::vector<std::string> dof_phase_list;
    dof_phase_list.push_back("alpha_salt");
    dof_phase_list.push_back("alpha_lime");

    // Create combined array of phase fraction fields
    const auto phase_vec
        = Utils::ScalarToVector<EvalType, PhaseIndex>::createFromList(
            *test_fixture.ir, "volume_fractions", dof_phase_list, false, false);

    test_fixture.registerEvaluator<EvalType>(phase_vec);

    const std::string lsvof_model_name
        = (lsvof_model == LSVOFModel::VOF) ? "VOF" : "CLS";

    // Create no slip evaluator.
    auto no_slip_eval = Teuchos::rcp(
        new BoundaryCondition::
            IncompressibleLSVOFNoSlip<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir,
                num_lsvof_dofs,
                lsvof_model_name,
                continuity_model_name,
                true));
    test_fixture.registerEvaluator<EvalType>(no_slip_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        no_slip_eval->_boundary_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_velocity[dim]);
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_grad_velocity[dim]);
    }

    if (continuity_model == ContinuityModel::EDAC)
    {
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_grad_lagrange_pressure);
    }

    if (lsvof_model_name == "VOF")
    {
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_alphas);
    }
    else if (lsvof_model_name == "CLS")
    {
        test_fixture.registerTestField<EvalType>(no_slip_eval->_boundary_phi);
    }
    // Evaluate incompressible LSVOF no-slip
    test_fixture.evaluate<EvalType>();

    // Expected values
    const double exp_lagrange_pres = 1.5;
    const double exp_grad_lagrange_pres[3]
        = {num_space_dim == 3 ? 5.15625 : 1.359375,
           num_space_dim == 3 ? -10.3125 : -2.71875,
           num_space_dim == 3 ? 15.46875 : nan_val};

    const double exp_grad_vel_0_3d[3] = {-0.25, 0.5, -0.75};
    const double exp_grad_vel_1_3d[3] = {-0.5, 1.0, -1.5};
    const double exp_grad_vel_2_3d[3] = {-0.125, 0.25, -0.375};

    const double exp_grad_vel_0_2d[2]
        = {exp_grad_vel_0_3d[0], exp_grad_vel_0_3d[1]};
    const double exp_grad_vel_1_2d[2]
        = {exp_grad_vel_1_3d[0], exp_grad_vel_1_3d[1]};

    const double* exp_grad_vel_0 = num_space_dim == 3 ? exp_grad_vel_0_3d
                                                      : exp_grad_vel_0_2d;
    const double* exp_grad_vel_1 = num_space_dim == 3 ? exp_grad_vel_1_3d
                                                      : exp_grad_vel_1_2d;
    const double* exp_grad_vel_2 = exp_grad_vel_2_3d;

    const double exp_alphas[2] = {0.123, 0.456};

    // Get no slip fields
    auto boundary_lagrange_pressure_result
        = test_fixture.getTestFieldData<EvalType>(
            no_slip_eval->_boundary_lagrange_pressure);

    // Loop over quadrature points and mesh dimension
    const int num_point = boundary_lagrange_pressure_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        // Lagrange pressure
        EXPECT_DOUBLE_EQ(exp_lagrange_pres,
                         fieldValue(boundary_lagrange_pressure_result, 0, qp));

        // Loop over mesh dimension to assert velocity and gradient vectors
        for (int d = 0; d < num_space_dim; d++)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    no_slip_eval->_boundary_velocity[d]);
            EXPECT_DOUBLE_EQ(0.0,
                             fieldValue(boundary_velocity_d_result, 0, qp));

            const auto boundary_grad_velocity_0_result
                = test_fixture.getTestFieldData<EvalType>(
                    no_slip_eval->_boundary_grad_velocity[0]);
            EXPECT_DOUBLE_EQ(
                exp_grad_vel_0[d],
                fieldValue(boundary_grad_velocity_0_result, 0, qp, d));

            const auto boundary_grad_velocity_1_result
                = test_fixture.getTestFieldData<EvalType>(
                    no_slip_eval->_boundary_grad_velocity[1]);
            EXPECT_DOUBLE_EQ(
                exp_grad_vel_1[d],
                fieldValue(boundary_grad_velocity_1_result, 0, qp, d));

            if (num_space_dim > 2)
            {
                const auto boundary_grad_velocity_2_result
                    = test_fixture.getTestFieldData<EvalType>(
                        no_slip_eval->_boundary_grad_velocity[2]);
                EXPECT_DOUBLE_EQ(
                    exp_grad_vel_2[d],
                    fieldValue(boundary_grad_velocity_2_result, 0, qp, d));
            }

            if (continuity_model == ContinuityModel::EDAC)
            {
                const auto boundary_grad_lagrange_pressure_result
                    = test_fixture.getTestFieldData<EvalType>(
                        no_slip_eval->_boundary_grad_lagrange_pressure);
                EXPECT_DOUBLE_EQ(
                    exp_grad_lagrange_pres[d],
                    fieldValue(
                        boundary_grad_lagrange_pressure_result, 0, qp, d));
            }
        }

        if (lsvof_model_name == "VOF")
        {
            auto boundary_alphas = test_fixture.getTestFieldData<EvalType>(
                no_slip_eval->_boundary_alphas);

            for (int phase = 0; phase < num_lsvof_dofs; ++phase)
            {
                EXPECT_DOUBLE_EQ(exp_alphas[phase],
                                 fieldValue(boundary_alphas, 0, qp, phase));
            }
        }
        else if (lsvof_model_name == "CLS")
        {
            auto boundary_phi = test_fixture.getTestFieldData<EvalType>(
                no_slip_eval->_boundary_phi);

            EXPECT_DOUBLE_EQ(42.1, fieldValue(boundary_phi, 0, qp));
        }
    }
}

//---------------------------------------------------------------------------//
// Value parameterized test fixture
struct EvaluationTest : public testing::TestWithParam<ContinuityModel>
{
    // Case generator for parameterized test
    struct ParamNameGenerator
    {
        std::string
        operator()(const testing::TestParamInfo<ContinuityModel>& info) const
        {
            const auto continuity_model = info.param;
            switch (continuity_model)
            {
                case (ContinuityModel::AC):
                    return "AC";
                case (ContinuityModel::EDAC):
                    return "EDAC";
                default:
                    return "INVALID_NAME";
            }
        }
    };
};

//---------------------------------------------------------------------------//
// 2-D incompressible lsvof no slip
TEST_P(EvaluationTest, VOFResidual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(continuity_model, LSVOFModel::VOF);
}

TEST_P(EvaluationTest, VOFJacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(continuity_model, LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
// 2-D incompressible CLS no slip
TEST_P(EvaluationTest, CLSResidual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(continuity_model, LSVOFModel::CLS);
}

TEST_P(EvaluationTest, CLSJacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(continuity_model, LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
// 3-D incompressible lsvof no slip
TEST_P(EvaluationTest, VOFResidual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(continuity_model, LSVOFModel::VOF);
}

TEST_P(EvaluationTest, VOFJacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(continuity_model, LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
// Generate test suite with continuity models
INSTANTIATE_TEST_SUITE_P(ContinuityModelType,
                         EvaluationTest,
                         testing::Values(ContinuityModel::AC,
                                         ContinuityModel::EDAC),
                         EvaluationTest::ParamNameGenerator{});

} // end namespace Test
} // end namespace VertexCFD
