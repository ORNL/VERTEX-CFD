#ifndef VERTEXCFD_CLOSURE_WALLDISTANCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_WALLDISTANCE_IMPL_HPP

#include <drivers/VertexCFD_MeshManager.hpp>
#include <mesh/VertexCFD_Mesh_GeometryData.hpp>
#include <mesh/VertexCFD_Mesh_GeometryPrimitives.hpp>

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_String_Utilities.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <cmath>

namespace VertexCFD
{
namespace ClosureModel
{

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
WallDistance<EvalType, Traits, NumSpaceDim>::WallDistance(
    const panzer::IntegrationRule& ir,
    Teuchos::RCP<MeshManager> mesh_manager,
    const Teuchos::ParameterList closure_params)
    : _distance("distance", ir.dl_scalar)
    , _ir_degree(ir.cubature_degree)
{
    this->addEvaluatedField(_distance);
    this->setName("distance");

    // Extract the wall names from input to be used to construct the _sides
    // view
    std::vector<std::string> wall_names;
    auto wall_names_list = closure_params.get<std::string>("Wall Names");
    panzer::StringTokenizer(wall_names, wall_names_list, ",", true);

    // create a sidesetGeometry instance and store the side data in the _sides
    // view
    VertexCFD::Mesh::Topology::SidesetGeometry surfaces(mesh_manager->mesh(),
                                                        wall_names);
    _topology = surfaces.topology();
    _key = _topology->getKey();
    _sides = surfaces.sides();

    _normals = Kokkos::View<double**, PHX::mem_space>(
        "normals", _sides.extent(0), num_space_dim);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void WallDistance<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
    _number_worksets = (*sd.worksets_).size();

    // Initialize the size of the _distance_vector view
    _distance_vector
        = Kokkos::View<double***, PHX::mem_space>("distance_vector",
                                                  _number_worksets,
                                                  _distance.extent(0),
                                                  _distance.extent(1));

    // size the iterator vectors for storing intial workset cell data
    _workset_id.resize(_number_worksets);

    // Loop over worksets to calculate wall distance and fill iterator data
    for (std::size_t wks = 0; wks < (*sd.worksets_).size(); ++wks)
    {
        _current_workset = wks;
        panzer::Workset workset = (*sd.worksets_)[wks];
        // If the workset is empty, skip the iterator setup
        if (workset.num_cells > 0)
        {
            _workset_id[wks] = workset.getIdentifier();
        }
        _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
        auto policy = Kokkos::RangePolicy<PHX::exec_space, RegistrationTag>(
            0, workset.num_cells);
        // Call operator for wall distance over cells in workset "wks"
        Kokkos::parallel_for(this->getName(), policy, *this);
    }
}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void WallDistance<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Only fill distance info if the workset is not null
    if (workset.num_cells > 0)
    {
        for (int i = 0; i < _number_worksets; ++i)
        {
            // Check which workset is the current one
            if (_workset_id[i] == workset.getIdentifier())
            {
                _current_workset = i;
                break;
            }
        }

        // Loop over cells in the workset and points in the cell and set the
        // distance based off pre-calculated value
        auto policy = Kokkos::RangePolicy<PHX::exec_space, EvaluateTag>(
            0, workset.num_cells);
        Kokkos::parallel_for("SetDistanceField", policy, *this);
    }
}
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
WallDistance<EvalType, Traits, NumSpaceDim>::operator()(RegistrationTag,
                                                        const int cell) const
{
    const int num_point = _distance.extent(1);
    for (int point = 0; point < num_point; ++point)
    {
        double distance = 1e8;
        double temp = 1e8;

        // loop over sides and calculate the minimum distance to the current ip
        for (long unsigned int n = 0; n < _sides.extent(0); ++n)
        {
            // call the triangle distance function and store the distance if
            // it is a minimum
            if constexpr (num_space_dim == 3)
            {
                if (_key == shards::Tetrahedron<4>::key)
                {
                    int index[3] = {0, 1, 2};
                    temp = GeometryPrimitives::distanceToTriangleFace(
                        _sides, _normals, n, _ip_coords, cell, point, index);
                }
                if (_key == shards::Hexahedron<8>::key)
                {
                    int index[3] = {0, 1, 2};
                    double t2 = GeometryPrimitives::distanceToTriangleFace(
                        _sides, _normals, n, _ip_coords, cell, point, index);
                    index[1] = 2;
                    index[2] = 3;
                    double t1 = GeometryPrimitives::distanceToTriangleFace(
                        _sides, _normals, n, _ip_coords, cell, point, index);
                    temp = std::fmin(t1, t2);
                }
            }
            if constexpr (num_space_dim == 2)
            {
                int index[2] = {0, 1};
                temp = GeometryPrimitives::distanceToLinearEdge(
                    _sides, n, _ip_coords, cell, point, 2, index);
            }
            if (temp < distance)
            {
                distance = temp;
            }
        }

        _distance_vector(_current_workset, cell, point) = distance;
    }
}
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
WallDistance<EvalType, Traits, NumSpaceDim>::operator()(EvaluateTag,
                                                        const int cell) const
{
    const int num_point = _distance.extent(1);
    for (int point = 0; point < num_point; ++point)
    {
        _distance(cell, point)
            = _distance_vector(_current_workset, cell, point);
    }
}
//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_WALLDISTANCE_IMPL_HPP
