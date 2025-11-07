/*
  Function for calculating the wall distance from an arbitrary point
  to an arbitrary (triangular) surface and/or (linear) edge
*/
#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace VertexCFD
{
namespace GeometryPrimitives
{

/*
    Calculates the cross product of two 3x1 vectors
*/
KOKKOS_INLINE_FUNCTION void
crossProduct(double arr1[3], double arr2[3], double arr_out[3])
{
    arr_out[0] = arr1[1] * arr2[2] - arr1[2] * arr2[1];
    arr_out[1] = arr1[2] * arr2[0] - arr1[0] * arr2[2];
    arr_out[2] = arr1[0] * arr2[1] - arr1[1] * arr2[0];
}

/*
  Checks if the intersection of a point, p, with a plane, defined ba * ca, lies
  within the triangle abc, and if it is outside the triangle what edge it lies
  closest to
*/
template<class SideViewType, class NormalViewType, class PointViewType>
KOKKOS_INLINE_FUNCTION int tetCheck(SideViewType& sides_view,
                                    NormalViewType& normals_view,
                                    int side,
                                    PointViewType& ip_view,
                                    int cell,
                                    int point,
                                    int index[3])
{
    // calculate the intersection location of the point and plane along the
    // normal vector
    const double t1 = normals_view(side, 0) * sides_view(side, index[0], 0)
                      + normals_view(side, 1) * sides_view(side, index[0], 1)
                      + normals_view(side, 2) * sides_view(side, index[0], 2);
    const double t2 = normals_view(side, 0) * ip_view(cell, point, 0)
                      + normals_view(side, 1) * ip_view(cell, point, 1)
                      + normals_view(side, 2) * ip_view(cell, point, 2);
    const double param = t1 - t2;

    double intersection_point[3];
    double line_a_int[3];
    double line_b_int[3];
    double line_c_int[3];
    double line_ba[3];
    double line_cb[3];
    double line_ac[3];

    for (int dim = 0; dim < 3; ++dim)
    {
        // Intersection point with the plane the side lies in
        intersection_point[dim] = ip_view(cell, point, dim)
                                  + param * normals_view(side, dim);
        // Line segment from intersection point to node A
        line_a_int[dim] = intersection_point[dim]
                          - sides_view(side, index[0], dim);
        // Line segment from node B to node A
        line_ba[dim] = sides_view(side, index[1], dim)
                       - sides_view(side, index[0], dim);
        // Line segment from intersection point to node B
        line_b_int[dim] = intersection_point[dim]
                          - sides_view(side, index[1], dim);
        // Line segment from node C to node B
        line_cb[dim] = sides_view(side, index[2], dim)
                       - sides_view(side, index[1], dim);
        // Line segment from intersection point to node C
        line_c_int[dim] = intersection_point[dim]
                          - sides_view(side, index[2], dim);
        // Line segment from node A to node C
        line_ac[dim] = sides_view(side, index[0], dim)
                       - sides_view(side, index[2], dim);
    }

    // calculate unit normal vector of triangle abp
    double norm_abp[3];
    crossProduct(line_ba, line_b_int, norm_abp);
    const double n_abp_mag = sqrt(norm_abp[0] * norm_abp[0]
                                  + norm_abp[1] * norm_abp[1]
                                  + norm_abp[2] * norm_abp[2]);
    if (n_abp_mag != 0)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            norm_abp[dim] /= n_abp_mag;
        }
    }

    // calculate unit normal vector of triangle bcp
    double norm_bcp[3];
    crossProduct(line_cb, line_c_int, norm_bcp);
    const double n_bcp_mag = sqrt(norm_bcp[0] * norm_bcp[0]
                                  + norm_bcp[1] * norm_bcp[1]
                                  + norm_bcp[2] * norm_bcp[2]);
    if (n_bcp_mag != 0)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            norm_bcp[dim] /= n_bcp_mag;
        }
    }

    // calculate unit normal vector of triangle cap
    double norm_cap[3];
    crossProduct(line_ac, line_a_int, norm_cap);
    const double n_cap_mag = sqrt(norm_cap[0] * norm_cap[0]
                                  + norm_cap[1] * norm_cap[1]
                                  + norm_cap[2] * norm_cap[2]);
    if (n_cap_mag != 0)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            norm_cap[dim] /= n_cap_mag;
        }
    }

    int tet_flag = -1;

    const double abp = norm_abp[0] * normals_view(side, 0)
                       + norm_abp[1] * normals_view(side, 1)
                       + norm_abp[2] * normals_view(side, 2);
    const double bcp = norm_bcp[0] * normals_view(side, 0)
                       + norm_bcp[1] * normals_view(side, 1)
                       + norm_bcp[2] * normals_view(side, 2);
    const double cap = norm_cap[0] * normals_view(side, 0)
                       + norm_cap[1] * normals_view(side, 1)
                       + norm_cap[2] * normals_view(side, 2);

    if (abp > 0.01 && bcp > 0.01 && cap > 0.01)
    {
        tet_flag = 0;
    }
    else if (abp <= 0.01 && cap > 0.01 && bcp > 0.01)
    {
        tet_flag = 1;
    }
    else if (bcp <= 0.01 && cap > 0.01 && abp > 0.01)
    {
        tet_flag = 2;
    }
    else if (cap <= 0.01 && abp > 0.01 && bcp > 0.01)
    {
        tet_flag = 3;
    }
    else if (abp <= 0.01 && bcp <= 0.01)
    {
        tet_flag = 4;
    }
    else if (bcp <= 0.01 && cap <= 0.01)
    {
        tet_flag = 5;
    }
    else if (cap <= 0.01 && abp <= 0.01)
    {
        tet_flag = 6;
    }

    return tet_flag;
}

/*
  Determines the distance between a point, p, and a plane defined by a point a
  and a normal n
*/
template<class SideViewType, class NormalViewType, class PointViewType>
KOKKOS_INLINE_FUNCTION double planeIntersect(SideViewType& sides_view,
                                             NormalViewType& normals_view,
                                             int side,
                                             PointViewType& ip_view,
                                             int cell,
                                             int point)
{
    // calculate n_hat * a - n_hat * p, which defines the distance to the
    // interaction point
    const double t1 = normals_view(side, 0) * sides_view(side, 0, 0)
                      + normals_view(side, 1) * sides_view(side, 0, 1)
                      + normals_view(side, 2) * sides_view(side, 0, 2);
    const double t2 = normals_view(side, 0) * ip_view(cell, point, 0)
                      + normals_view(side, 1) * ip_view(cell, point, 1)
                      + normals_view(side, 2) * ip_view(cell, point, 2);
    const double param = t1 - t2;
    const double distance = std::abs(param);

    return distance;
}

/*
  Function that returns the distance to a specified edge from a specified point
  p is the test point, p0 is one end of the linear edge given, and p1 is the
  other end of the linear edge given
*/
template<class SideViewType, class PointViewType>
KOKKOS_INLINE_FUNCTION double distanceToLinearEdge(SideViewType& sides_view,
                                                   int side,
                                                   PointViewType& ip_view,
                                                   int cell,
                                                   int point,
                                                   int num_space_dim,
                                                   int index[2])
{
    double line_ap[3];
    double line_ab[3];
    double num = 0.0;
    double denom = 0.0;

    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        // Line between point 0 and the test point
        line_ap[dim] = ip_view(cell, point, dim)
                       - sides_view(side, index[0], dim);
        // Line between point 0 and point 1
        line_ab[dim] = sides_view(side, index[1], dim)
                       - sides_view(side, index[0], dim);
        num += line_ap[dim] * line_ab[dim];
        denom += line_ab[dim] * line_ab[dim];
    }

    double param = -1;
    // don't test case where the line has 0 length
    if (denom != 0)
        param = num / denom;

    double intersection_point[3];

    if (param < 0)
    {
        // the point is closest to end p0
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            intersection_point[dim] = sides_view(side, index[0], dim);
        }
    }
    else if (param > 1)
    {
        // the point is closest to end p1
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            intersection_point[dim] = sides_view(side, index[1], dim);
        }
    }
    else
    {
        // the point "p" is closest to a point "param" along line p0-p1
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            intersection_point[dim] = sides_view(side, index[0], dim)
                                      + param * line_ab[dim];
        }
    }

    double distance = 0.0;
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        distance += pow(ip_view(cell, point, dim) - intersection_point[dim], 2);
    }
    return sqrt(distance);
}

/*
  Function that returns the nearest distance between a triangle and a point
*/
template<class SideViewType, class NormalViewType, class PointViewType>
KOKKOS_INLINE_FUNCTION double
distanceToTriangleFace(SideViewType& sides_view,
                       NormalViewType& normals_view,
                       int side,
                       PointViewType& ip_view,
                       int cell,
                       int point,
                       int index[3])
{
    // Define vectors that store the line between the first point on the
    // triangle and the other two
    double line_ba[3];
    double line_ca[3];
    for (int dim = 0; dim < 3; ++dim)
    {
        line_ba[dim] = sides_view(side, index[1], dim)
                       - sides_view(side, index[0], dim);
        line_ca[dim] = sides_view(side, index[2], dim)
                       - sides_view(side, index[0], dim);
    }

    // Calculate the triangle unit normal vector
    double normal_abc[3];
    crossProduct(line_ba, line_ca, normal_abc);
    normals_view(side, 0) = normal_abc[0];
    normals_view(side, 1) = normal_abc[1];
    normals_view(side, 2) = normal_abc[2];
    const double normal_mag
        = std::sqrt(normals_view(side, 0) * normals_view(side, 0)
                    + normals_view(side, 1) * normals_view(side, 1)
                    + normals_view(side, 2) * normals_view(side, 2));
    normals_view(side, 0) /= normal_mag;
    normals_view(side, 1) /= normal_mag;
    normals_view(side, 2) /= normal_mag;

    // calculate if "p" is closest to the triangle face, or one of the sides
    auto int_flag = tetCheck(
        sides_view, normals_view, side, ip_view, cell, point, index);

    double distance = 1e8;
    double d_ab, d_bc, d_ca;
    int node[2];
    switch (int_flag)
    {
        case 0:
            // point "p" is closest to a point on the triangle interior
            distance = planeIntersect(
                sides_view, normals_view, side, ip_view, cell, point);
            break;
        case 1:
            // point "p" is closest to a point on the edge ab
            node[0] = index[0];
            node[1] = index[1];
            distance = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            break;
        case 2:
            // point "p" is closest to a point on the edge bc
            node[0] = index[1];
            node[1] = index[2];
            distance = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            break;
        case 3:
            // point "p" is closest to a point on the edge ca
            node[0] = index[2];
            node[1] = index[0];
            distance = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            break;
        case 4:
            // point "p" is closest to a point on the edge ab or bc
            node[0] = index[0];
            node[1] = index[1];
            d_ab = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            node[0] = index[1];
            node[1] = index[2];
            d_bc = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            distance = std::fmin(d_ab, d_bc);
            break;
        case 5:
            // point "p" is closest to a point on the edge bc or ca
            node[0] = index[1];
            node[1] = index[2];
            d_bc = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            node[0] = index[2];
            node[1] = index[0];
            d_ca = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            distance = std::fmin(d_ca, d_bc);
            break;
        case 6:
            // point "p" is closest to a point on the edge ab or ca
            node[0] = index[0];
            node[1] = index[1];
            d_ab = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            node[0] = index[2];
            node[1] = index[0];
            d_ca = distanceToLinearEdge(
                sides_view, side, ip_view, cell, point, 3, node);
            distance = std::fmin(d_ab, d_ca);
            break;
    }

    return distance;
}

} // end namespace GeometryPrimitives
} // end namespace VertexCFD
