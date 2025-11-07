#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_FullInductionFixed.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType, int NumSpaceDim>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr int num_magnetic_field_dim = 3;

    Kokkos::Array<double, num_magnetic_field_dim> _grad_b;
    double _scalar_magn_pot;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_induced_magnetic_field;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _scalar_magnetic_potential;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, num_magnetic_field_dim> grad_b,
                 const double scalar_magn_pot)
        : _grad_b(grad_b)
        , _scalar_magn_pot(scalar_magn_pot)
        , _scalar_magnetic_potential("scalar_magnetic_potential", ir.dl_scalar)
    {
        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_vector,
                                       _grad_induced_magnetic_field,
                                       "GRAD_induced_magnetic_field_");

        this->addEvaluatedField(_scalar_magnetic_potential);

        this->setName("Full Induction Model Fixed Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            _grad_induced_magnetic_field[dim].deep_copy(_grad_b[dim]);
        }
        _scalar_magnetic_potential.deep_copy(_scalar_magn_pot);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(bool build_magn_corr, bool dirichlet_scalar_magn_pot)
{
    // Test fixture
    static constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    static constexpr int num_magnetic_field_dim = 3;
    const double nanval = std::numeric_limits<double>::signaling_NaN();
    const Kokkos::Array<double, num_magnetic_field_dim> grad_b
        = {1.1, 2.2, num_space_dim == 2 ? nanval : 3.3};
    const double scalar_magn_pot
        = (build_magn_corr && !dirichlet_scalar_magn_pot) ? 4.4 : nanval;

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(
            *test_fixture.ir, grad_b, scalar_magn_pot));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create fixed evaluator.
    const Kokkos::Array<double, num_magnetic_field_dim> bnd_b
        = {0.3, 0.4, num_space_dim == 2 ? nanval : 0.5};
    const double bnd_scalar_magn_pot
        = (build_magn_corr && dirichlet_scalar_magn_pot) ? 5.5 : nanval;
    const double exp_scalar_magn_pot
        = dirichlet_scalar_magn_pot ? bnd_scalar_magn_pot : scalar_magn_pot;

    Teuchos::ParameterList bc_params;
    bc_params.set("induced_magnetic_field_0", bnd_b[0]);
    bc_params.set("induced_magnetic_field_1", bnd_b[1]);
    if (num_space_dim == 3)
        bc_params.set("induced_magnetic_field_2", bnd_b[2]);
    if (build_magn_corr && dirichlet_scalar_magn_pot)
        bc_params.set("scalar_magnetic_potential", bnd_scalar_magn_pot);

    Teuchos::ParameterList full_indu_params;
    full_indu_params.set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", build_magn_corr)
        .set("Hyperbolic Divergence Cleaning Speed", 1.1);

    const MHDProperties::FullInductionMHDProperties mhd_props(full_indu_params);

    const auto fixed_eval = Teuchos::rcp(
        new BoundaryCondition::
            FullInductionFixed<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, bc_params, mhd_props));

    test_fixture.registerEvaluator<EvalType>(fixed_eval);

    // Add required test fields.
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            fixed_eval->_boundary_induced_magnetic_field[dim]);
        test_fixture.registerTestField<EvalType>(
            fixed_eval->_boundary_grad_induced_magnetic_field[dim]);
    }
    if (build_magn_corr)
    {
        test_fixture.registerTestField<EvalType>(
            fixed_eval->_boundary_scalar_magnetic_potential);
    }

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto bnd_magn_field_0_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_induced_magnetic_field[0]);
    const auto bnd_grad_magn_field_0_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_grad_induced_magnetic_field[0]);
    const auto bnd_magn_field_1_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_induced_magnetic_field[1]);
    const auto bnd_grad_magn_field_1_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_grad_induced_magnetic_field[1]);

    const int num_point = bnd_magn_field_0_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(bnd_b[0], fieldValue(bnd_magn_field_0_result, 0, qp));
        EXPECT_DOUBLE_EQ(bnd_b[1], fieldValue(bnd_magn_field_1_result, 0, qp));
        if (num_space_dim == 3)
        {
            const auto bnd_magn_field_2_result
                = test_fixture.getTestFieldData<EvalType>(
                    fixed_eval->_boundary_induced_magnetic_field[2]);
            EXPECT_DOUBLE_EQ(bnd_b[2],
                             fieldValue(bnd_magn_field_2_result, 0, qp));
        }
        if (build_magn_corr)
        {
            const auto bnd_scalar_magn_pot_result
                = test_fixture.getTestFieldData<EvalType>(
                    fixed_eval->_boundary_scalar_magnetic_potential);
            EXPECT_DOUBLE_EQ(exp_scalar_magn_pot,
                             fieldValue(bnd_scalar_magn_pot_result, 0, qp));
        }

        for (int d = 0; d < num_space_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(
                grad_b[0], fieldValue(bnd_grad_magn_field_0_result, 0, qp, d));
            EXPECT_DOUBLE_EQ(
                grad_b[1], fieldValue(bnd_grad_magn_field_1_result, 0, qp, d));
            if (num_space_dim == 3)
            {
                const auto bnd_grad_magn_field_2_result
                    = test_fixture.getTestFieldData<EvalType>(
                        fixed_eval->_boundary_grad_induced_magnetic_field[2]);
                EXPECT_DOUBLE_EQ(
                    grad_b[2],
                    fieldValue(bnd_grad_magn_field_2_result, 0, qp, d));
            }
        }
    }
}

//---------------------------------------------------------------------------//
struct FullInductionFixedTestParams
{
    bool build_magn_corr;
    bool dirichlet_scalar_magn_pot;
};

class FullInductionFixed
    : public testing::TestWithParam<FullInductionFixedTestParams>
{
  public:
    struct PrintNameString
    {
        template<class T>
        std::string operator()(const testing::TestParamInfo<T>& info) const
        {
            auto p = static_cast<FullInductionFixedTestParams>(info.param);
            const std::string test_name = !p.build_magn_corr ? "NoCleaning"
                                          : p.dirichlet_scalar_magn_pot
                                              ? "FixedScalarMagneticPotential"
                                              : "FreeScalarMagneticPotential";
            return test_name;
        }
    };
};

//---------------------------------------------------------------------------//
TEST_P(FullInductionFixed, residual_2d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Residual, 2>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot);
}

//---------------------------------------------------------------------------//
TEST_P(FullInductionFixed, jacobian_2d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot);
}

//---------------------------------------------------------------------------//
TEST_P(FullInductionFixed, residual_3d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Residual, 3>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot);
}

//---------------------------------------------------------------------------//
TEST_P(FullInductionFixed, jacobian_3d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot);
}

//---------------------------------------------------------------------------//
INSTANTIATE_TEST_SUITE_P(
    Test,
    FullInductionFixed,
    testing::Values(FullInductionFixedTestParams{false, false},
                    FullInductionFixedTestParams{true, false},
                    FullInductionFixedTestParams{true, true}),
    FullInductionFixed::PrintNameString());

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
