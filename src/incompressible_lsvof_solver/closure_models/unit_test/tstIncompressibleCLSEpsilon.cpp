#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSEpsilon.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
enum class EpsilonMeasurementMethod
{
    Minimum,
    Maximum,
    Magnitude,
    Average
};

//---------------------------------------------------------------------------//
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> element_length;

    Dependencies(const panzer::IntegrationRule& ir)
        : element_length("element_length", ir.dl_vector)
    {
        this->addEvaluatedField(element_length);
        this->setName("Incompressible CLS Epsilon Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "Incompressible CLS epsilon test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = element_length.extent(1);
        const int num_space_dim = element_length.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                element_length(c, qp, dim) = 0.25 * (dim + 1);
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const EpsilonMeasurementMethod method)
{
    const int integration_order = 2;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Create parameter list and expected value
    Teuchos::ParameterList params;
    double exp_eps = 0.0;

    params.set("Interface Thickness Coefficient", 1.0);

    if (method == EpsilonMeasurementMethod::Minimum)
    {
        params.set<std::string>("Epsilon Measurement Method", "Minimum");
        exp_eps = 0.25;
    }
    else if (method == EpsilonMeasurementMethod::Maximum)
    {
        params.set<std::string>("Epsilon Measurement Method", "Maximum");
        exp_eps = num_space_dim == 2 ? 0.50 : 0.75;
    }
    else if (method == EpsilonMeasurementMethod::Magnitude)
    {
        params.set<std::string>("Epsilon Measurement Method", "Magnitude");
        exp_eps = num_space_dim == 2 ? 0.55901699437494745
                                     : 0.93541434669348533;
    }
    else
    {
        params.set<std::string>("Epsilon Measurement Method", "Average");
        exp_eps = num_space_dim == 2 ? 0.375 : 0.5;
    }

    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleCLSEpsilon<EvalType,
                                                   panzer::Traits,
                                                   num_space_dim>(ir, params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_epsilon);

    test_fixture.evaluate<EvalType>();

    const auto fc_eps = test_fixture.getTestFieldData<EvalType>(eval->_epsilon);

    const int num_point = ir.num_points;

    // Check on values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_eps, fieldValue(fc_eps, 0, qp));
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMinimum2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(EpsilonMeasurementMethod::Minimum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMinimum2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(EpsilonMeasurementMethod::Minimum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMinimum3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(EpsilonMeasurementMethod::Minimum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMinimum3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(EpsilonMeasurementMethod::Minimum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMaximum2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(EpsilonMeasurementMethod::Maximum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMaximum2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(EpsilonMeasurementMethod::Maximum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMaximum3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(EpsilonMeasurementMethod::Maximum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMaximum3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(EpsilonMeasurementMethod::Maximum);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMagnitude2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(EpsilonMeasurementMethod::Magnitude);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMagnitude2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(EpsilonMeasurementMethod::Magnitude);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMagnitude3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(EpsilonMeasurementMethod::Magnitude);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonMagnitude3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(EpsilonMeasurementMethod::Magnitude);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonAverage2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(EpsilonMeasurementMethod::Average);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonAverage2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(EpsilonMeasurementMethod::Average);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonAverage3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(EpsilonMeasurementMethod::Average);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSEpsilonAverage3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(EpsilonMeasurementMethod::Average);
}

} // namespace Test
} // namespace VertexCFD
