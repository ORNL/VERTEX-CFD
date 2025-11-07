#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADDiffusionFlux.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_species_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_species_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_species_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_species_0("GRAD_species_0", ir.dl_vector)
        , grad_species_1("GRAD_species_1", ir.dl_vector)
        , grad_species_2("GRAD_species_2", ir.dl_vector)
    {
        this->addEvaluatedField(grad_species_0);
        this->addEvaluatedField(grad_species_1);
        this->addEvaluatedField(grad_species_2);

        this->setName("RAD Diffusion Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "rad diffusion flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_species_0.extent(1);
        const int num_space_dim = grad_species_0.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_species_0(c, qp, dim) = 1.25 * dimqp;
                grad_species_1(c, qp, dim) = 2.5 * dimqp;
                grad_species_2(c, qp, dim) = 3.125 * dimqp;
            }
        }
    }
};

template<class EvalType>
void testEval(int num_space_dim)
{
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;
    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::ParameterList rad_params;
    rad_params.set("Number of Species", 3);
    rad_params.set("Build Diffusion", true);
    rad_params.set("Diffusion Coefficient", 0.5);
    const Teuchos::ParameterList reaction_params;

    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Initialize class object to test
    auto eval = Teuchos::rcp(
        new ClosureModel::RADDiffusionFlux<EvalType, panzer::Traits>(
            ir, species_prop));
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int num = 0; num < 3; ++num)
        test_fixture.registerTestField<EvalType>(eval->_species_flux[num]);

    test_fixture.evaluate<EvalType>();

    auto calc_species_flux_0
        = test_fixture.getTestFieldData<EvalType>(eval->_species_flux[0]);
    auto calc_species_flux_1
        = test_fixture.getTestFieldData<EvalType>(eval->_species_flux[1]);
    auto calc_species_flux_2
        = test_fixture.getTestFieldData<EvalType>(eval->_species_flux[2]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_species_0_flux[3]
        = {-0.625, 1.25, num_space_dim == 3 ? -1.875 : nan_val};
    const double exp_species_1_flux[3]
        = {-1.25, 2.5, num_space_dim == 3 ? -3.75 : nan_val};
    const double exp_species_2_flux[3]
        = {-1.5625, 3.125, num_space_dim == 3 ? -4.6875 : nan_val};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_species_0_flux[dim],
                      fieldValue(calc_species_flux_0, 0, qp, dim));
            EXPECT_EQ(exp_species_1_flux[dim],
                      fieldValue(calc_species_flux_1, 0, qp, dim));
            EXPECT_EQ(exp_species_2_flux[dim],
                      fieldValue(calc_species_flux_2, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(RADDiffusionFlux2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(RADDiffusionFlux2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(RADDiffusionFlux3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(RADDiffusionFlux3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

} // namespace Test
} // namespace VertexCFD
