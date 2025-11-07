#include <VertexCFD_ClosureModelFactoryTestHarness.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <closure_models/VertexCFD_Closure_MetricTensor.hpp>

#include <Panzer_IntegrationDescriptor.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Traits.hpp>

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopology.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, class CellTopo>
void testEval(const double (&coords)[CellTopo::node_count][CellTopo::dimension],
              const double (&exp)[CellTopo::dimension][CellTopo::dimension])
{
    constexpr int num_cell = 1;
    constexpr int nodes_per_cell = CellTopo::node_count;
    constexpr int num_point = 1;
    constexpr int num_space_dim = CellTopo::dimension;

    const EvaluatorTestFixture::host_coords_view coords_view(
        "coords", num_cell, nodes_per_cell, num_space_dim);
    for (int i = 0; i < nodes_per_cell; ++i)
        for (int j = 0; j < num_space_dim; ++j)
            coords_view(0, i, j) = coords[i][j];

    const int integration_order = 0;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(shards::getCellTopologyData<CellTopo>(),
                                      coords_view,
                                      integration_order,
                                      basis_order);

    EXPECT_EQ(nodes_per_cell, test_fixture.cardinality());
    EXPECT_EQ(num_point, test_fixture.numPoint());
    if (::testing::Test::HasFailure())
    {
        GTEST_FAIL() << "Unexpected test fixture parameters";
    }

    // Create evaluator.
    auto metric_tensor_eval = Teuchos::rcp(
        new ClosureModel::MetricTensor<EvalType, panzer::Traits>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(metric_tensor_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        metric_tensor_eval->_metric_tensor);

    // Evaluate metric tensor.
    test_fixture.evaluate<EvalType>();

    // Check the metric tensor.
    auto metric_tensor_result = test_fixture.getTestFieldData<EvalType>(
        metric_tensor_eval->_metric_tensor);

    EXPECT_EQ(num_cell, metric_tensor_result.extent(0));
    EXPECT_EQ(num_point, metric_tensor_result.extent(1));
    EXPECT_EQ(num_space_dim, metric_tensor_result.extent(2));
    if (::testing::Test::HasFailure())
    {
        GTEST_FAIL() << "Unexpected metric tensor result extents";
    }

    constexpr double tol = 1.0e-14;

    for (int i = 0; i < num_space_dim; ++i)
    {
        for (int j = 0; j < num_space_dim; ++j)
        {
            // For zero exptected values, use absolute tolerance.
            if (exp[i][j] < tol)
            {
                EXPECT_NEAR(exp[i][j], metric_tensor_result(0, 0, i, j), tol)
                    << "M[" << i << ", " << j << ']';
            }
            // For non-zero expected values, use relative (ULP) tolerance.
            else
            {
                EXPECT_DOUBLE_EQ(exp[i][j], metric_tensor_result(0, 0, i, j))
                    << "M[" << i << ", " << j << ']';
            }
        }
    }
}

//---------------------------------------------------------------------------//
// Apply a coordinate transformation given by a rotation/scaling matrix and
// displacement vector.
template<int N, int D>
void applyTransformation(double (&coords)[N][D],
                         const double (&mtx)[D][D],
                         const double (&disp)[D])
{
    for (int node = 0; node < N; ++node)
    {
        double new_coord[D]{};
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j)
                new_coord[i] += mtx[i][j] * coords[node][j];
        for (int i = 0; i < D; ++i)
            coords[node][i] = new_coord[i] + disp[i];
    }
}

//---------------------------------------------------------------------------//
// Test the ideal cell - all edges are length 1.
// The result should be the identity.
template<class EvalType, class CellTopo>
struct IdealCellTest
{
    static constexpr int N = CellTopo::node_count;
    static constexpr int D = CellTopo::dimension;

    static void eval(const double (&coords)[N][D])
    {
        double exp[D][D]{};
        for (int i = 0; i < D; ++i)
            exp[i][i] = 1.0;

        testEval<EvalType, CellTopo>(coords, exp);
    }
};

//---------------------------------------------------------------------------//
// Test an isotropic cell - scale, rotate, and displace the ideal cell.
// The result should be scaled but not impacted by rotation or displacement.
template<class EvalType, class CellTopo>
struct IsotropicCellTest
{
    static constexpr int N = CellTopo::node_count;
    static constexpr int D = CellTopo::dimension;

    // Transform 1-D coordinates.
    static void transformCoords(double (&coords)[N][1])
    {
        // Scale by 5.
        constexpr double mtx[1][1] = {
            {5.0},
        };

        // Displace.
        constexpr double disp[1] = {1.0};

        applyTransformation(coords, mtx, disp);
    }

    // Transform 2-D coordinates.
    static void transformCoords(double (&coords)[N][2])
    {
        // Scale by 5 and rotate.
        constexpr double mtx[2][2] = {
            {4.0, -3.0},
            {3.0, 4.0},
        };

        // Displace in all directions.
        constexpr double disp[2] = {1.0, -2.0};

        applyTransformation(coords, mtx, disp);
    }

    // Transform 3-D coordinates.
    static void transformCoords(double (&coords)[N][3])
    {
        // Scale by 5 and rotate about three axes.
        constexpr double mtx[3][3] = {
            {0.12, -3.84, 3.2},
            {3.84, 2.12, 2.4},
            {-3.2, 2.4, 3.0},
        };

        // Displace in all directions.
        constexpr double disp[3] = {1.0, -2.0, 0.5};

        applyTransformation(coords, mtx, disp);
    }

    static void eval(double (&coords)[N][D])
    {
        transformCoords(coords);

        double exp[D][D]{};
        for (int i = 0; i < D; ++i)
            exp[i][i] = 25.0;

        testEval<EvalType, CellTopo>(coords, exp);
    }
};

//---------------------------------------------------------------------------//
// Test an anisotropic cell - scale, rotate, and displace the ideal cell.
// The result should be impacted by scaling and rotation, but not displacement.
template<class EvalType, class CellTopo>
struct AnisotropicCellTest
{
    static constexpr int N = CellTopo::node_count;
    static constexpr int D = CellTopo::dimension;

    // Transform 2-D coordinates.
    static void transformCoords(double (&coords)[N][2])
    {
        // Scale (anisotropically) and rotate.
        constexpr double mtx[2][2] = {
            {4.0, -1.5},
            {3.0, 2.0},
        };

        // Displace in all directions.
        constexpr double disp[2] = {1.0, -2.0};

        applyTransformation(coords, mtx, disp);
    }

    // Expected 2-D metric.
    static void setExpected(double (&exp)[2][2])
    {
        exp[0][0] = 18.25;
        exp[0][1] = 9.0;

        exp[1][0] = 9.0;
        exp[1][1] = 13.0;
    }

    // Transform 3-D coordinates.
    static void transformCoords(double (&coords)[N][3])
    {
        // Scale (anisotropically) and rotate about three axes.
        constexpr double mtx[3][3] = {
            {0.12, -1.92, 6.4},
            {3.84, 1.06, 4.8},
            {-3.2, 1.2, 6.0},
        };

        // Displace in all directions.
        constexpr double disp[3] = {1.0, -2.0, 0.5};

        applyTransformation(coords, mtx, disp);
    }

    // Expected 3-D metric.
    static void setExpected(double (&exp)[3][3])
    {
        exp[0][0] = 44.6608;
        exp[0][1] = 29.1456;
        exp[0][2] = 35.712;

        exp[1][0] = 29.1456;
        exp[1][1] = 38.9092;
        exp[1][2] = 17.784;

        exp[2][0] = 35.712;
        exp[2][1] = 17.784;
        exp[2][2] = 47.68;
    }

    static void eval(double (&coords)[N][D])
    {
        transformCoords(coords);

        double exp[D][D]{};
        setExpected(exp);

        testEval<EvalType, CellTopo>(coords, exp);
    }
};

//---------------------------------------------------------------------------//
// Test residual and Jacobian evaluations.
template<typename EvalType>
class MetricTensorTest : public ::testing::Test
{
};
using EvalTypes
    = ::testing::Types<panzer::Traits::Residual, panzer::Traits::Jacobian>;
class EvalTypeNames
{
  public:
    template<typename T>
    static std::string GetName(const int i)
    {
        if (std::is_same<T, panzer::Traits::Residual>())
            return "Residual";
        if (std::is_same<T, panzer::Traits::Jacobian>())
            return "Jacobian";
        return std::to_string(i);
    }
};
TYPED_TEST_SUITE(MetricTensorTest, EvalTypes, EvalTypeNames);

//---------------------------------------------------------------------------//
// Test a 1-D line.
template<class EvalType, template<class, class> class CellTest>
void testLine()
{
    // Ideal line.
    // NOLINTNEXTLINE(misc-const-correctness)
    double coords[2][1] = {
        {0.0},
        {1.0},
    };

    CellTest<EvalType, shards::Line<2>>::eval(coords);
}

TYPED_TEST(MetricTensorTest, IdealLine)
{
    testLine<TypeParam, IdealCellTest>();
}

TYPED_TEST(MetricTensorTest, IsotropicLine)
{
    testLine<TypeParam, IsotropicCellTest>();
}

//---------------------------------------------------------------------------//
// Test a 2-D triangle.
template<class EvalType, template<class, class> class CellTest>
void testTriangle()
{
    // Ideal triangle.
    // NOLINTNEXTLINE(misc-const-correctness)
    double coords[3][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.5, 0.5 * std::sqrt(3.0)},
    };

    CellTest<EvalType, shards::Triangle<3>>::eval(coords);
}

TYPED_TEST(MetricTensorTest, IdealTriangle)
{
    testTriangle<TypeParam, IdealCellTest>();
}

TYPED_TEST(MetricTensorTest, IsotropicTriangle)
{
    testTriangle<TypeParam, IsotropicCellTest>();
}

TYPED_TEST(MetricTensorTest, AnisotropicTriangle)
{
    testTriangle<TypeParam, AnisotropicCellTest>();
}

//---------------------------------------------------------------------------//
// Test a 2-D quadrilateral.
template<class EvalType, template<class, class> class CellTest>
void testQuadrilateral()
{
    // Ideal quadrilateral.
    // NOLINTNEXTLINE(misc-const-correctness)
    double coords[4][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 1.0},
    };

    CellTest<EvalType, shards::Quadrilateral<4>>::eval(coords);
}

TYPED_TEST(MetricTensorTest, IdealQuadrilateral)
{
    testQuadrilateral<TypeParam, IdealCellTest>();
}

TYPED_TEST(MetricTensorTest, IsotropicQuadrilateral)
{
    testQuadrilateral<TypeParam, IsotropicCellTest>();
}

TYPED_TEST(MetricTensorTest, AnisotropicQuadrilateral)
{
    testQuadrilateral<TypeParam, AnisotropicCellTest>();
}

//---------------------------------------------------------------------------//
// Test a 3-D tetrahedron.
template<class EvalType, template<class, class> class CellTest>
void testTetrahedron()
{
    // Ideal tetrahedron.
    // NOLINTNEXTLINE(misc-const-correctness)
    double coords[4][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.5, 0.5 * std::sqrt(3.0), 0.0},
        {0.5, 0.5 / std::sqrt(3.0), std::sqrt(2.0 / 3.0)},
    };

    CellTest<EvalType, shards::Tetrahedron<4>>::eval(coords);
}

TYPED_TEST(MetricTensorTest, IdealTetrahedron)
{
    testTetrahedron<TypeParam, IdealCellTest>();
}

TYPED_TEST(MetricTensorTest, IsotropicTetrahedron)
{
    testTetrahedron<TypeParam, IsotropicCellTest>();
}

TYPED_TEST(MetricTensorTest, AnisotropicTetrahedron)
{
    testTetrahedron<TypeParam, AnisotropicCellTest>();
}

//---------------------------------------------------------------------------//
// Test a 3-D hexahedron.
template<class EvalType, template<class, class> class CellTest>
void testHexahedron()
{
    // Ideal hexahedron.
    // NOLINTNEXTLINE(misc-const-correctness)
    double coords[8][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    CellTest<EvalType, shards::Hexahedron<8>>::eval(coords);
}

TYPED_TEST(MetricTensorTest, IdealHexahedron)
{
    testHexahedron<TypeParam, IdealCellTest>();
}

TYPED_TEST(MetricTensorTest, IsotropicHexahedron)
{
    testHexahedron<TypeParam, IsotropicCellTest>();
}

TYPED_TEST(MetricTensorTest, AnisotropicHexahedron)
{
    testHexahedron<TypeParam, AnisotropicCellTest>();
}

//---------------------------------------------------------------------------//
// Test a 3-D pyramid.
template<class EvalType, template<class, class> class CellTest>
void testPyramid()
{
    // Ideal pyramid.
    // NOLINTNEXTLINE(misc-const-correctness)
    double coords[5][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.5, 0.5, 1.0 / std::sqrt(2.0)},
    };

    // Pyramid cells are not currently supported by panzer.
    ASSERT_THROW((CellTest<EvalType, shards::Pyramid<5>>::eval(coords)),
                 std::runtime_error);
}

TYPED_TEST(MetricTensorTest, IdealPyramid)
{
    testPyramid<TypeParam, IdealCellTest>();
}

TYPED_TEST(MetricTensorTest, IsotropicPyramid)
{
    testPyramid<TypeParam, IsotropicCellTest>();
}

TYPED_TEST(MetricTensorTest, AnisotropicPyramid)
{
    testPyramid<TypeParam, AnisotropicCellTest>();
}

//---------------------------------------------------------------------------//
// Test a 3-D wedge.
template<class EvalType, template<class, class> class CellTest>
void testWedge()
{
    // Ideal wedge.
    // NOLINTNEXTLINE(misc-const-correctness)
    double coords[6][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.5, 0.5 * std::sqrt(3.0), 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 1.0},
        {0.5, 0.5 * std::sqrt(3.0), 1.0},
    };

    // Wedge cells are not currently supported by panzer.
    ASSERT_THROW((CellTest<EvalType, shards::Wedge<6>>::eval(coords)),
                 std::runtime_error);
}

TYPED_TEST(MetricTensorTest, IdealWedge)
{
    testWedge<TypeParam, IdealCellTest>();
}

TYPED_TEST(MetricTensorTest, IsotropicWedge)
{
    testWedge<TypeParam, IsotropicCellTest>();
}

TYPED_TEST(MetricTensorTest, AnisotropicWedge)
{
    testWedge<TypeParam, AnisotropicCellTest>();
}

//---------------------------------------------------------------------------//
// Try to construct the evaluator for an unsupported cell topology.
template<class EvalType>
void testUnsupportedTopology()
{
    // Set up an integration_rule for an unsupported cell topology.
    // ShellQuadrilateral is the only topology supported by Intrepid but not
    // by this closure model.
    const panzer::IntegrationDescriptor integration_desc(
        0, panzer::IntegrationDescriptor::VOLUME);
    auto cell_topo = Teuchos::rcp(new shards::CellTopology(
        shards::getCellTopologyData<shards::ShellQuadrilateral<4>>()));
    const int num_cells = 0;
    auto integration_rule = Teuchos::rcp(
        new panzer::IntegrationRule(integration_desc, cell_topo, num_cells));

    using MetricTensorEval
        = ClosureModel::MetricTensor<EvalType, panzer::Traits>;

    const std::string msg = "Invalid base cell topology: ShellQuadrilateral_4";
    ASSERT_THROW(
        try {
            auto metric_tensor_eval = MetricTensorEval(*integration_rule);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::runtime_error);
}

TYPED_TEST(MetricTensorTest, UnsupportedTopology)
{
    testUnsupportedTopology<TypeParam>();
}

//---------------------------------------------------------------------------//
// Test construction from the closure model factory.
template<class EvalType>
void testFactory()
{
    constexpr int num_space_dim = 2;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MetricTensor";
    test_fixture.eval_name = "Metric Tensor";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::MetricTensor<EvalType, panzer::Traits>,
        num_space_dim>();
}

TYPED_TEST(MetricTensorTest, Factory)
{
    testFactory<TypeParam>();
}

} // namespace Test
} // namespace VertexCFD
