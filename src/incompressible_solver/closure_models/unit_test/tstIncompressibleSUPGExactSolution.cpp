#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleSUPGExactSolution.hpp"

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

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> cp;

    Dependencies(const panzer::IntegrationRule& ir)
        : rho("density", ir.dl_scalar)
        , k("thermal_conductivity", ir.dl_scalar)
        , cp("specific_heat_capacity", ir.dl_scalar)

    {
        this->addEvaluatedField(rho);
        this->addEvaluatedField(k);
        this->addEvaluatedField(cp);

        this->setName("Incompressible Viscous Heat Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "viscous heat test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = rho.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            rho(c, qp) = 2.0;
            k(c, qp) = 3.0;
            cp(c, qp) = 4.0;
        }
    }
};

template<class EvalType>
void testEval()
{
    const int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;

    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);
    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Set non-trivial coordinates for the degree of freedom
    const int num_point = ir.num_points;
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        for (int qp = 0; qp < num_point; ++qp)
        {
            ip_coord_mirror(0, qp, dim) = 0.1 * (dim + 1) * (qp + 1) - 0.25;
        }
    }
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Initialize class object to test
    Teuchos::ParameterList closure_params;
    closure_params.set("Uniform advection velocity", 5.0);
    closure_params.set("Thermal diffusivity", 3.0 / (2.0 * 4.0));
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleSUPGExactSolution<EvalType,
                                                          panzer::Traits,
                                                          num_space_dim>(
            ir, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_velocity[dim]);
    test_fixture.registerTestField<EvalType>(eval->_temperature);
    test_fixture.evaluate<EvalType>();

    const auto exact_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto exact_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto exact_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);
    const auto exact_temp
        = test_fixture.getTestFieldData<EvalType>(eval->_temperature);

    // Reference values
    const double ns_ref = 0.0;
    const double temp_ref[4] = {-0.029999719917906037,
                                -0.009999842386130088,
                                0.009999693010100228,
                                0.029997930457695426};

    // Assert values
    const double tol = 1.0e-12;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(ns_ref, fieldValue(exact_cont, 0, qp));
        EXPECT_DOUBLE_EQ(ns_ref, fieldValue(exact_mom_0, 0, qp));
        EXPECT_DOUBLE_EQ(ns_ref, fieldValue(exact_mom_1, 0, qp));
        EXPECT_NEAR(temp_ref[qp], fieldValue(exact_temp, 0, qp), tol);
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSUPGExactSolution, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleSUPGExactSolution, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleSUPGExactSolution";
    test_fixture.model_params.set("Uniform advection velocity", -2.0)
        .set("Thermal diffusivity", 1.0);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.eval_name = "Incompressible SUPG Exact Solution";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleSUPGExactSolution<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleSUPGExactSolution_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleSUPGExactSolution_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
