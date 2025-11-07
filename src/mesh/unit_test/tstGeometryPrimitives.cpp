#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <mesh/VertexCFD_Mesh_GeometryPrimitives.hpp>

#include <Panzer_Traits.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class SideViewType, class PointViewType, class ResultViewType, class NormalViewType>
KOKKOS_INLINE_FUNCTION void faceDistanceFunc(SideViewType sides,
                                             PointViewType point,
                                             ResultViewType result,
                                             NormalViewType normal,
                                             int index)
{
    int orient[3] = {0, 1, 2};
    result(index, 0) = VertexCFD::GeometryPrimitives::tetCheck(
        sides, normal, 0, point, 0, 0, orient);
    result(index, 1) = VertexCFD::GeometryPrimitives::planeIntersect(
        sides, normal, 0, point, 0, 0);
    result(index, 2) = VertexCFD::GeometryPrimitives::distanceToTriangleFace(
        sides, normal, 0, point, 0, 0, orient);
}

template<class EvalType>
void testEval()
{
    const double tol = 1e-6;

    // Define kokkos views needed for all tests
    const Kokkos::View<double***, Kokkos::LayoutRight, PHX::Device> point(
        Kokkos::ViewAllocateWithoutInitializing("pointView"), 1, 1, 3);
    const Kokkos::View<double***, Kokkos::LayoutRight, PHX::Device> sides(
        Kokkos::ViewAllocateWithoutInitializing("sidesView"), 1, 3, 3);
    const Kokkos::View<double**, Kokkos::LayoutRight, PHX::Device> normal(
        Kokkos::ViewAllocateWithoutInitializing("normalView"), 1, 3);
    const Kokkos::View<double[6], Kokkos::LayoutRight, PHX::Device> dist(
        Kokkos::ViewAllocateWithoutInitializing("distanceView"));

    // Test of linearEdge function
    Kokkos::parallel_for(
        "linearEdgeTest",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            // Define the line segment to test against
            for (int dim = 0; dim < 3; ++dim)
            {
                sides(0, 0, dim) = 0.0;
                sides(0, 1, dim) = 1.0;
            }
            sides(0, 1, 2) = 0.0;

            // Test for 3d test_point that is closest to some point on the line
            // segment
            point(0, 0, 0) = 0.1;
            point(0, 0, 1) = 0.1;
            point(0, 0, 2) = 1.0;
            int index[2] = {0, 1};
            dist(0) = VertexCFD::GeometryPrimitives::distanceToLinearEdge(
                sides, 0, point, 0, 0, 3, index);

            // Test for 3d test_point that is closest to the end of the line
            point(0, 0, 0) = 2.5;
            point(0, 0, 1) = 1.0;
            point(0, 0, 2) = 2.0;
            dist(1) = VertexCFD::GeometryPrimitives::distanceToLinearEdge(
                sides, 0, point, 0, 0, 3, index);

            // Test for 3d test_point that is closest to the start of the line
            point(0, 0, 0) = -1.0;
            point(0, 0, 1) = -2.4;
            point(0, 0, 2) = 0.0;
            dist(2) = VertexCFD::GeometryPrimitives::distanceToLinearEdge(
                sides, 0, point, 0, 0, 3, index);

            // Test for 2d test_point that is closest to the end of the line
            point(0, 0, 0) = 1.0;
            point(0, 0, 1) = 3.0;
            dist(3) = VertexCFD::GeometryPrimitives::distanceToLinearEdge(
                sides, 0, point, 0, 0, 2, index);

            // Test for 2d test_point that is closest to the some point on the
            // line
            point(0, 0, 0) = 0.2;
            point(0, 0, 1) = 0.8;
            dist(4) = VertexCFD::GeometryPrimitives::distanceToLinearEdge(
                sides, 0, point, 0, 0, 2, index);

            // Test for 2d test_point that is closest to the start of the line
            point(0, 0, 0) = -0.8;
            point(0, 0, 1) = 0.6;
            dist(5) = VertexCFD::GeometryPrimitives::distanceToLinearEdge(
                sides, 0, point, 0, 0, 2, index);
        });

    auto dist_mirror
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dist);
    EXPECT_DOUBLE_EQ(1.0, dist_mirror(0));
    EXPECT_DOUBLE_EQ(2.5, dist_mirror(1));
    EXPECT_DOUBLE_EQ(2.6, dist_mirror(2));
    EXPECT_DOUBLE_EQ(2.0, dist_mirror(3));
    EXPECT_NEAR(0.424264, dist_mirror(4), tol);
    EXPECT_DOUBLE_EQ(1.0, dist_mirror(5));

    const Kokkos::View<double[7][3], Kokkos::LayoutRight, PHX::Device> result(
        Kokkos::ViewAllocateWithoutInitializing("ResultView"));

    // Test of distanceToTriangleFace, tetCheck and planeIntersect functions
    Kokkos::parallel_for(
        "tetIntersectTest",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            // Define the unit triangle to test against
            for (int dim = 0; dim < 3; ++dim)
            {
                sides(0, 0, dim) = 0.0;
                sides(0, 1, dim) = 0.0;
                sides(0, 2, dim) = 0.0;
                normal(0, dim) = 0.0;
            }
            sides(0, 1, 0) = 1.0;
            sides(0, 2, 1) = 1.0;
            normal(0, 2) = 1.0;

            // Test for a point that intersects the triangle
            int index = 0;
            point(0, 0, 0) = 0.1;
            point(0, 0, 1) = 0.1;
            point(0, 0, 2) = 1.0;
            faceDistanceFunc(sides, point, result, normal, index);

            // Test for a point that is closest to edge ab
            ++index;
            point(0, 0, 0) = 0.5;
            point(0, 0, 1) = -0.5;
            point(0, 0, 2) = 1.0;
            faceDistanceFunc(sides, point, result, normal, index);

            // Test for a point that is closest to edge bc
            ++index;
            point(0, 0, 0) = 1.0;
            point(0, 0, 1) = 1.0;
            point(0, 0, 2) = 1.0;
            faceDistanceFunc(sides, point, result, normal, index);

            // Test for a point that is closest to edge ca
            ++index;
            point(0, 0, 0) = -0.5;
            point(0, 0, 1) = 0.5;
            point(0, 0, 2) = 1.0;
            faceDistanceFunc(sides, point, result, normal, index);

            // Test for a point that is closest to edge ab or bc
            ++index;
            point(0, 0, 0) = 2.0;
            point(0, 0, 1) = -1.0;
            point(0, 0, 2) = 1.0;
            faceDistanceFunc(sides, point, result, normal, index);

            // Test for a point that is closest to edge bc or ca
            ++index;
            point(0, 0, 0) = -1.0;
            point(0, 0, 1) = 4.0;
            point(0, 0, 2) = 1.0;
            faceDistanceFunc(sides, point, result, normal, index);

            // Test for a point that is closest to edge ca or ab
            ++index;
            point(0, 0, 0) = -2.0;
            point(0, 0, 1) = -1.0;
            point(0, 0, 2) = 1.0;
            faceDistanceFunc(sides, point, result, normal, index);
        });

    auto result_mirror
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    EXPECT_EQ(0, result_mirror(0, 0));
    EXPECT_DOUBLE_EQ(1.0, result_mirror(0, 1));
    EXPECT_NEAR(1.0, result_mirror(0, 2), tol);

    EXPECT_EQ(1, result_mirror(1, 0));
    EXPECT_DOUBLE_EQ(1.0, result_mirror(1, 1));
    EXPECT_NEAR(1.118034, result_mirror(1, 2), tol);

    EXPECT_EQ(2, result_mirror(2, 0));
    EXPECT_DOUBLE_EQ(1.0, result_mirror(2, 1));
    EXPECT_NEAR(1.22474487, result_mirror(2, 2), tol);

    EXPECT_EQ(3, result_mirror(3, 0));
    EXPECT_DOUBLE_EQ(1.0, result_mirror(3, 1));
    EXPECT_NEAR(1.118034, result_mirror(3, 2), tol);

    EXPECT_EQ(4, result_mirror(4, 0));
    EXPECT_DOUBLE_EQ(1.0, result_mirror(4, 1));
    EXPECT_NEAR(1.732050807, result_mirror(4, 2), tol);

    EXPECT_EQ(5, result_mirror(5, 0));
    EXPECT_DOUBLE_EQ(1.0, result_mirror(5, 1));
    EXPECT_NEAR(3.31662479, result_mirror(5, 2), tol);

    EXPECT_EQ(6, result_mirror(6, 0));
    EXPECT_DOUBLE_EQ(1.0, result_mirror(6, 1));
    EXPECT_NEAR(2.44948974, result_mirror(6, 2), tol);

    // Test the crossProduct function
    const Kokkos::View<double[5][3], Kokkos::LayoutRight, PHX::Device>
        cross_product(Kokkos::ViewAllocateWithoutInitializing("CrossProductVie"
                                                              "w"));

    Kokkos::parallel_for(
        "CrossProductTest",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            double array1[3];
            double array2[3];
            double array_out[3];
            array1[0] = 1.0;
            array1[1] = 0.0;
            array1[2] = 0.0;

            array2[0] = 0.0;
            array2[1] = 1.0;
            array2[2] = 0.0;
            VertexCFD::GeometryPrimitives::crossProduct(
                array1, array2, array_out);
            cross_product(0, 0) = array_out[0];
            cross_product(0, 1) = array_out[1];
            cross_product(0, 2) = array_out[2];

            array2[0] = 0.0;
            array2[1] = 0.0;
            array2[2] = 1.0;
            VertexCFD::GeometryPrimitives::crossProduct(
                array1, array2, array_out);
            cross_product(1, 0) = array_out[0];
            cross_product(1, 1) = array_out[1];
            cross_product(1, 2) = array_out[2];

            array2[0] = 1.0;
            array2[1] = 0.0;
            array2[2] = 0.0;
            VertexCFD::GeometryPrimitives::crossProduct(
                array1, array2, array_out);
            cross_product(2, 0) = array_out[0];
            cross_product(2, 1) = array_out[1];
            cross_product(2, 2) = array_out[2];

            array2[0] = 1.0;
            array2[1] = 1.0;
            array2[2] = 1.0;
            VertexCFD::GeometryPrimitives::crossProduct(
                array1, array2, array_out);
            cross_product(3, 0) = array_out[0];
            cross_product(3, 1) = array_out[1];
            cross_product(3, 2) = array_out[2];

            array2[0] = 0.0;
            array2[1] = 0.0;
            array2[2] = 0.0;
            VertexCFD::GeometryPrimitives::crossProduct(
                array1, array2, array_out);
            cross_product(4, 0) = array_out[0];
            cross_product(4, 1) = array_out[1];
            cross_product(4, 2) = array_out[2];
        });

    auto cross_product_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cross_product);
    EXPECT_EQ(0, cross_product_mirror(0, 0));
    EXPECT_EQ(0, cross_product_mirror(0, 1));
    EXPECT_EQ(1, cross_product_mirror(0, 2));

    EXPECT_EQ(0, cross_product_mirror(1, 0));
    EXPECT_EQ(-1, cross_product_mirror(1, 1));
    EXPECT_EQ(0, cross_product_mirror(1, 2));

    EXPECT_EQ(0, cross_product_mirror(2, 0));
    EXPECT_EQ(0, cross_product_mirror(2, 1));
    EXPECT_EQ(0, cross_product_mirror(2, 2));

    EXPECT_EQ(0, cross_product_mirror(3, 0));
    EXPECT_EQ(-1, cross_product_mirror(3, 1));
    EXPECT_EQ(1, cross_product_mirror(3, 2));

    EXPECT_EQ(0, cross_product_mirror(4, 0));
    EXPECT_EQ(0, cross_product_mirror(4, 1));
    EXPECT_EQ(0, cross_product_mirror(4, 2));
}

//---------------------------------------------------------------------------//
TEST(GeometryPrimitives, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(GeometryPrimitives, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
