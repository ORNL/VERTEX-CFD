#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_DivergenceCleaningSource.hpp"

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    static constexpr int num_space_dim = NumSpaceDim;
    using scalar_type = typename EvalType::ScalarT;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_scalar_magnetic_potential;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_scalar_magnetic_potential("GRAD_scalar_magnetic_potential",
                                         ir.dl_vector)
    {
        this->addEvaluatedField(grad_scalar_magnetic_potential);
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, velocity, "velocity_");

        this->setName("Divergence Cleaning Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "divergence cleaning source test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_scalar_magnetic_potential.extent(1);
        const int num_grad_dim = grad_scalar_magnetic_potential.extent(2);
        using std::pow;

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                grad_scalar_magnetic_potential(c, qp, dim) = pow(-0.9, dim + 1)
                                                             * (qp + dim + 1);
            }
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                velocity[dim](c, qp) = pow(-0.6, dim) * (qp + dim + 1);
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const int num_grad_dim)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize class object to test
    auto deps = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::DivergenceCleaningSource<EvalType,
                                                   panzer::Traits,
                                                   num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(
        eval->_div_cleaning_potential_source);

    test_fixture.evaluate<EvalType>();

    const auto src_magn_pot = test_fixture.getTestFieldData<EvalType>(
        eval->_div_cleaning_potential_source);

    const int num_point = ir.num_points;

    const double exp_src_magn_pot_2d[8]
        = {2.844, 7.974, 15.876, 26.55, 39.996, 56.214, 75.204, 96.966};
    const double exp_src_magn_pot_3d[8] = {
        5.20596, 12.17304, 22.437, 35.99784, 52.85556, 73.01016, 96.46164, 123.21};
    const auto exp_src_magn_pot = num_grad_dim == 2 ? exp_src_magn_pot_2d
                                                    : exp_src_magn_pot_3d;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_src_magn_pot[qp], fieldValue(src_magn_pot, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(DivergenceCleaningSource2Vel2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(2);
}

//-----------------------------------------------------------------//
TEST(DivergenceCleaningSource2Vel2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(2);
}

//-----------------------------------------------------------------//
TEST(DivergenceCleaningSource3Vel2D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(2);
}

//-----------------------------------------------------------------//
TEST(DivergenceCleaningSource3Vel2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(2);
}
//-----------------------------------------------------------------//
TEST(DivergenceCleaningSource3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(3);
}

//-----------------------------------------------------------------//
TEST(DivergenceCleaningSource3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(3);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", false);
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.type_name = "DivergenceCleaningSource";
    test_fixture.eval_name = "Divergence Cleaning Source "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::DivergenceCleaningSource<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(DivergenceCleaningSource_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(DivergenceCleaningSource_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(DivergenceCleaningSource_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(DivergenceCleaningSource_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
