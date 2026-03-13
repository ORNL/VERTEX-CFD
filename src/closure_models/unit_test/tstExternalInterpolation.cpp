#include <drivers/unit_test/VertexCFD_DriverUnitTestConfig.hpp>

#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include <closure_models/VertexCFD_Closure_ExternalInterpolation.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

// Check interpolation of sin(4*x)+sin(2*y)
template<class EvalType, int NumSpaceDim>
void testInterpolation()
{
    const int integration_order = 3;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        NumSpaceDim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    Teuchos::ParameterList closure_params("closure parameters");
    closure_params.set("output field name", "external_scalar_field");
    closure_params.set("input field name", "solution");
    const std::string location = VERTEXCFD_DRIVER_TEST_INPUT_DIR;
    const std::string file = "rectangle_field.exo";
    const std::string file_path = location + file;
    closure_params.set("file name", file_path);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new VertexCFD::ClosureModel::
            ExternalInterpolation<EvalType, panzer::Traits, NumSpaceDim>(
                ir, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_scalar_field);
    test_fixture.setTime(1);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_scalar_field
        = test_fixture.getTestFieldData<EvalType>(eval->_scalar_field);

    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto coordinates = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           ip_coord_view);

// ArborX improved the interpolation implementation past the 2.0.1 release
#if ARBORX_VERSION_MAJOR >= 2    \
    && (ARBORX_VERSION_MINOR > 0 \
        || (ARBORX_VERSION_MINOR == 0 && ARBORX_VERSION_PATCH > 1))
    const double tolerance = .015;
#else
    const double tolerance = .15;
#endif
    for (std::size_t i = 0; i < coordinates.extent(1); ++i)
        EXPECT_NEAR(
            sin(4.0 * coordinates(0, i, 0)) + sin(2.0 * coordinates(0, i, 1)),
            fieldValue(fv_scalar_field, 0, i),
            tolerance)
            << " coordinates (" << coordinates(0, i, 0) << ','
            << coordinates(0, i, 1) << ')';

#if ARBORX_VERSION_MAJOR >= 2    \
    && (ARBORX_VERSION_MINOR > 0 \
        || (ARBORX_VERSION_MINOR == 0 && ARBORX_VERSION_PATCH > 1))
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 0), 0.9774091503117741, 1.e-12);
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 1), 1.7340436324852422, 1.e-12);
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 2), 0.3986279754566450, 1.e-12);
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 3), 1.1569758030062278, 1.e-12);
#else
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 0), 1.1147379340872794, 1.e-12);
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 1), 1.8606353374168845, 1.e-12);
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 2), 0.3990041504336132, 1.e-12);
    EXPECT_NEAR(fieldValue(fv_scalar_field, 0, 3), 1.1569638890895977, 1.e-12);
#endif
}

// Check timestep selection
template<class EvalType, int NumSpaceDim>
void testTimestep()
{
    const int integration_order = 3;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        NumSpaceDim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    Teuchos::ParameterList closure_params("closure parameters");
    closure_params.set("output field name", "external_scalar_field");
    closure_params.set("input field name", "solution");
    const std::string location = VERTEXCFD_DRIVER_TEST_INPUT_DIR;
    const std::string file = "constant_per_time_step.exo";
    const std::string file_path = location + file;
    closure_params.set("file name", file_path);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new VertexCFD::ClosureModel::
            ExternalInterpolation<EvalType, panzer::Traits, NumSpaceDim>(
                ir, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_scalar_field);

// ArborX improved the interpolation implementation past the 2.0.1 release
#if ARBORX_VERSION_MAJOR >= 2    \
    && (ARBORX_VERSION_MINOR > 0 \
        || (ARBORX_VERSION_MINOR == 0 && ARBORX_VERSION_PATCH > 1))
    const double tolerance = 2.e-8;
#else
    const double tolerance = 8.e-3;
#endif

    // The test file has time steps 0., 0.2, 0.5, 0.7 we expect to choose the
    // closest
    std::vector<double> time_steps;
    const int n_time_steps = 13;
    for (int i = 0; i < n_time_steps + 1; ++i)
        time_steps.push_back(1. / n_time_steps * i);
    std::vector<double> expected_time{
        0, 0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7};

    for (unsigned int i = 0; i < time_steps.size(); ++i)
    {
        test_fixture.setTime(time_steps[i]);
        test_fixture.evaluate<EvalType>();

        // Evaluate test fields
        const auto fv_scalar_field
            = test_fixture.getTestFieldData<EvalType>(eval->_scalar_field);

        auto ip_coord_view
            = test_fixture.int_values->ip_coordinates.get_static_view();
        auto coordinates = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, ip_coord_view);
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(
                fieldValue(fv_scalar_field, 0, j), expected_time[i], tolerance);
    }

    // Test with shuffled time
    std::vector<int> indices(n_time_steps);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());
    for (const int i : indices)
    {
        test_fixture.setTime(time_steps[i]);
        test_fixture.evaluate<EvalType>();

        // Evaluate test fields
        const auto fv_scalar_field
            = test_fixture.getTestFieldData<EvalType>(eval->_scalar_field);

        auto ip_coord_view
            = test_fixture.int_values->ip_coordinates.get_static_view();
        auto coordinates = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, ip_coord_view);
        for (int j = 0; j < 4; ++j)
            EXPECT_NEAR(
                fieldValue(fv_scalar_field, 0, j), expected_time[i], tolerance);
    }
}

//-----------------------------------------------------------------//
TEST(ExternalInterpolation, Residual)
{
    testInterpolation<panzer::Traits::Residual, 2>();
    testTimestep<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(ExternalInterpolation, Jacobian)
{
    testInterpolation<panzer::Traits::Jacobian, 2>();
    testTimestep<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
