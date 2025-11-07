#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFArtificialCompression.hpp"

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

    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> scalar;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_scalar;

    Dependencies(const panzer::IntegrationRule& ir)
        : vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , scalar("alpha_air", ir.dl_scalar)
        , grad_scalar("GRAD_alpha_air", ir.dl_vector)

    {
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        this->addEvaluatedField(scalar);
        this->addEvaluatedField(grad_scalar);

        this->setName(
            "Incompressible LSVOF Artificial Compression for Scalar Equation "
            "Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF artificial compression test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_scalar.extent(1);
        const int num_space_dim = grad_scalar.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            vel_0(c, qp) = 0.25;
            vel_1(c, qp) = 0.5;
            vel_2(c, qp) = num_space_dim > 2 ? 0.125 : _nanval;
            scalar(c, qp) = 0.3;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_scalar(c, qp, dim) = (vel_0(c, qp) + vel_1(c, qp)) * dimqp;
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    using std::pow;
    using std::sqrt;
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    // Initialize velocity components and dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Compression Factor", 20.0);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFArtificialCompression<EvalType,
                                                                   panzer::Traits,
                                                                   num_space_dim>(
            ir, closure_params, "alpha_air", "alpha_air_equation"));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_scalar_flux);

    test_fixture.evaluate<EvalType>();

    const auto fc_scalar
        = test_fixture.getTestFieldData<EvalType>(eval->_scalar_flux);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_scalar_flux_3d[3]
        = {-0.642991057480584, 1.285982114961168, -1.9289731724417523};

    const double exp_scalar_flux_2d[3] = {-1.05, 2.1, _nanval};

    const double* exp_scalar_flux = num_space_dim == 3 ? exp_scalar_flux_3d
                                                       : exp_scalar_flux_2d;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_DOUBLE_EQ(exp_scalar_flux[dim],
                             fieldValue(fc_scalar, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFArtificialCompression2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFArtificialCompression2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFArtificialCompression3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFArtificialCompression3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
