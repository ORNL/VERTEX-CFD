#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFEntropyViscosity.hpp"

#include <Panzer_GlobalData.hpp>
#include <Panzer_ParameterLibrary.hpp>
#include <Panzer_ScalarParameterEntry.hpp>
#include <Teuchos_RCP.hpp>

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
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> element_length;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> entropy_residual;

    Dependencies(const panzer::IntegrationRule& ir)
        : vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , element_length("element_length", ir.dl_vector)
        , entropy_residual("entropy_residual", ir.dl_scalar)

    {
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        this->addEvaluatedField(element_length);
        this->addEvaluatedField(entropy_residual);

        this->setName(
            "Incompressible LSVOF Entropy Viscosity for Scalar Equation "
            "Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF entropy viscosity test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = element_length.extent(1);
        const int num_space_dim = element_length.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            vel_0(c, qp) = 0.25;
            vel_1(c, qp) = 0.5;
            vel_2(c, qp) = num_space_dim > 2 ? 0.125 : _nanval;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                element_length(c, qp, dim) = std::pow(0.1, dim);
            }

            entropy_residual(c, qp) = 0.1;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Ceiling viscosity parameter", 0.5);
    closure_params.set("Local entropy parameter", 0.5);

    // Entropy function and residual
    auto global_data = panzer::createGlobalData();
    auto& pl = *global_data->pl;
    pl.addParameterFamily("Max Entropy - entropy_fluctuation", true, false);
    Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> entry;
    entry = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
    entry->setValue(0.1);
    pl.addEntry<EvalType>("Max Entropy - entropy_fluctuation", entry);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFEntropyViscosity<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
            ir, closure_params, "alpha_air_equation", global_data));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_max);
    test_fixture.registerTestField<EvalType>(eval->_nu_e);
    test_fixture.registerTestField<EvalType>(eval->_artificial_viscosity);

    test_fixture.evaluate<EvalType>();

    const auto fc_nu_max
        = test_fixture.getTestFieldData<EvalType>(eval->_nu_max);
    const auto fc_nu_e = test_fixture.getTestFieldData<EvalType>(eval->_nu_e);
    const auto fc_artificial_viscosity
        = test_fixture.getTestFieldData<EvalType>(eval->_artificial_viscosity);

    const int num_point = ir.num_points;

    // Expected values

    double exp_nu_max = (num_space_dim == 3) ? 0.0028641098093474
                                             : 0.027950849718747374;
    double exp_local_viscosity = (num_space_dim == 3) ? 5.0e-5
                                                      : 0.005000000000000001;
    double exp_artificial_viscosity
        = (num_space_dim == 3) ? 5.0e-5 : 0.005000000000000001;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_nu_max, fieldValue(fc_nu_max, 0, qp));

        EXPECT_DOUBLE_EQ(exp_local_viscosity, fieldValue(fc_nu_e, 0, qp));

        EXPECT_DOUBLE_EQ(exp_artificial_viscosity,
                         fieldValue(fc_artificial_viscosity, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFEntropyViscosity2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

TEST(TestLSVOFEntropyViscosity2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

TEST(TestLSVOFEntropyViscosity3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

TEST(TestLSVOFEntropyViscosity3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
