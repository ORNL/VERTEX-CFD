#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include <turbulence_models/closure_models/VertexCFD_Closure_IncompressibleWALEEddyViscosity.hpp>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> element_length;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , element_length("les_element_length", ir.dl_vector)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(element_length);
        this->setName(
            "Incompressible WALE Eddy Viscosity Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "Incompressible WALE eddy viscosity test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_vel_0.extent(1);
        const int num_space_dim = grad_vel_0.extent(2);
        using std::pow;

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.125 * dimqp
                                                            : _nanval;
                element_length(c, qp, dim) = 0.750 * dimqp;
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    const int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Create parameter list for user-defined constants
    Teuchos::ParameterList turb_params;
    turb_params.set<double>("C_w", 0.500).set<double>("C_k", 0.094);

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleWALEEddyViscosity<EvalType,
                                                          panzer::Traits,
                                                          NumSpaceDim>(
            ir, turb_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_t);
    test_fixture.registerTestField<EvalType>(eval->_k_sgs);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_nu_t = test_fixture.getTestFieldData<EvalType>(eval->_nu_t);
    const auto fv_k_sgs = test_fixture.getTestFieldData<EvalType>(eval->_k_sgs);

    // Expected values
    const int num_point = ir.num_points;

    const double exp_nu_t = num_space_dim == 3 ? 0.04137844429829328
                                               : 0.14112104197373945;
    const double exp_k_sgs = num_space_dim == 3 ? 0.024606058225685103
                                                : 0.8013741154973666;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_nu_t, fieldValue(fv_nu_t, 0, qp));
        EXPECT_DOUBLE_EQ(exp_k_sgs, fieldValue(fv_k_sgs, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleWALEEddyViscosity2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleWALEEddyViscosity2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleWALEEddyViscosity3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleWALEEddyViscosity3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
