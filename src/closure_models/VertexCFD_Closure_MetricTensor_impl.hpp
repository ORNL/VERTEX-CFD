#ifndef VERTEXCFD_CLOSURE_METRICTENSOR_IMPL_HPP
#define VERTEXCFD_CLOSURE_METRICTENSOR_IMPL_HPP

#include <Panzer_Workset_Utilities.hpp>

#include <Shards_BasicTopologies.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
MetricTensor<EvalType, Traits>::MetricTensor(const panzer::IntegrationRule& ir)
    : _metric_tensor("metric_tensor", ir.dl_tensor)
    , _ir_degree(ir.cubature_degree)
    , _num_topo_dim(ir.topology->getDimension())
    , _element_map(Kokkos::ViewAllocateWithoutInitializing("element_map"),
                   _num_topo_dim,
                   _num_topo_dim)
{
    //
    // Define the mapping between the ideal and the reference element.
    //
    auto map = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _element_map);
    switch (ir.topology->getBaseKey())
    {
        case shards::Line<>::key:
            map(0, 0) = 2.0;

            break;
        case shards::Triangle<>::key:
            map(0, 0) = 1.0;
            map(0, 1) = 0.0;

            map(1, 0) = -1.0 / std::sqrt(3.0);
            map(1, 1) = 2.0 / std::sqrt(3.0);

            break;
        case shards::Quadrilateral<>::key:
            map(0, 0) = 2.0;
            map(0, 1) = 0.0;

            map(1, 0) = 0.0;
            map(1, 1) = 2.0;

            break;
        case shards::Tetrahedron<>::key:
            map(0, 0) = 1.0;
            map(0, 1) = 0.0;
            map(0, 2) = 0.0;

            map(1, 0) = -1.0 / std::sqrt(3.0);
            map(1, 1) = 2.0 / std::sqrt(3.0);
            map(1, 2) = 0.0;

            map(2, 0) = -1.0 / std::sqrt(6.0);
            map(2, 1) = -1.0 / std::sqrt(6.0);
            map(2, 2) = std::sqrt(3.0 / 2.0);

            break;
        case shards::Pyramid<>::key:
            map(0, 0) = 2.0;
            map(0, 1) = 0.0;
            map(0, 2) = 0.0;

            map(1, 0) = 0.0;
            map(1, 1) = 2.0;
            map(1, 2) = 0.0;

            map(2, 0) = 0.0;
            map(2, 1) = 0.0;
            map(2, 2) = std::sqrt(2.0);

            break;
        case shards::Wedge<>::key:
            map(0, 0) = 1.0;
            map(0, 1) = 0.0;
            map(0, 2) = 0.0;

            map(1, 0) = -1.0 / std::sqrt(3.0);
            map(1, 1) = 2.0 / std::sqrt(3.0);
            map(1, 2) = 0.0;

            map(2, 0) = 0.0;
            map(2, 1) = 0.0;
            map(2, 2) = 2.0;

            break;
        case shards::Hexahedron<>::key:
            map(0, 0) = 2.0;
            map(0, 1) = 0.0;
            map(0, 2) = 0.0;

            map(1, 0) = 0.0;
            map(1, 1) = 2.0;
            map(1, 2) = 0.0;

            map(2, 0) = 0.0;
            map(2, 1) = 0.0;
            map(2, 2) = 2.0;

            break;
        default:
            using namespace std::string_literals;
            throw std::runtime_error("Invalid base cell topology: "s
                                     + ir.topology->getBaseName());
    }
    Kokkos::deep_copy(_element_map, map);

    this->addEvaluatedField(_metric_tensor);
    this->setName("Metric Tensor");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void MetricTensor<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void MetricTensor<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _jacobian = workset.int_rules[_ir_index]->jac;

    const int num_cell = workset.num_cells;
    const int num_point = _jacobian.extent(1);

    if (_num_topo_dim == 1)
    {
        Kokkos::MDRangePolicy<PHX::Device,
                              Kokkos::Rank<2>,
                              std::integral_constant<int, 1>>
            policy({0, 0}, {num_cell, num_point});

        Kokkos::parallel_for(this->getName(), policy, *this);
    }
    else if (_num_topo_dim == 2)
    {
        Kokkos::MDRangePolicy<PHX::Device,
                              Kokkos::Rank<2>,
                              std::integral_constant<int, 2>>
            policy({0, 0}, {num_cell, num_point});

        Kokkos::parallel_for(this->getName(), policy, *this);
    }
    else if (_num_topo_dim == 3)
    {
        Kokkos::MDRangePolicy<PHX::Device,
                              Kokkos::Rank<2>,
                              std::integral_constant<int, 3>>
            policy({0, 0}, {num_cell, num_point});

        Kokkos::parallel_for(this->getName(), policy, *this);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
template<int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void MetricTensor<EvalType, Traits>::operator()(
    std::integral_constant<int, NumSpaceDim>,
    const int cell,
    const int point) const
{
    constexpr int num_space_dim = NumSpaceDim;

    //
    // Transform Jacobian to ideal element.
    //
    double jac[num_space_dim][num_space_dim]{};

    for (int i = 0; i < num_space_dim; ++i)
        for (int j = 0; j < num_space_dim; ++j)
            for (int k = 0; k < num_space_dim; ++k)
                jac[i][j] += _jacobian(cell, point, i, k) * _element_map(j, k);

    //
    // Compute metric tensor (covariant) from ideal Jacobian.
    //
    for (int i = 0; i < num_space_dim; ++i)
    {
        for (int j = 0; j < num_space_dim; ++j)
        {
            _metric_tensor(cell, point, i, j) = 0.0;
            for (int k = 0; k < num_space_dim; ++k)
                _metric_tensor(cell, point, i, j) += jac[i][k] * jac[j][k];
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_METRICTENSOR_IMPL_HPP
