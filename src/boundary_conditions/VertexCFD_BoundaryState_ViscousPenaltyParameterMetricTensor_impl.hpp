#ifndef VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETERMETRICTENSOR_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETERMETRICTENSOR_IMPL_HPP

#include "VertexCFD_BoundaryState_ViscousPenaltyParameterMetricTensor.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ViscousPenaltyParameterMetricTensor<EvalType, Traits>::
    ViscousPenaltyParameterMetricTensor(const panzer::IntegrationRule& ir)
    : _penalty_param("viscous_penalty_parameter", ir.dl_scalar)
    , _num_space_dim(ir.spatial_dimension)
    , _normals("Side Normal", ir.dl_vector)
    , _metric_tensor("metric_tensor", ir.dl_tensor)
{
    this->addEvaluatedField(_penalty_param);
    this->addDependentField(_normals);
    this->addDependentField(_metric_tensor);
    this->setName("Boundary State Viscous Penalty Parameter in "
                  + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ViscousPenaltyParameterMetricTensor<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ViscousPenaltyParameterMetricTensor<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _normals.extent(1);

    using Kokkos::sqrt;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _penalty_param(cell, point) = 0.0;
            for (int i = 0; i < _num_space_dim; ++i)
            {
                for (int j = 0; j < _num_space_dim; ++j)
                {
                    _penalty_param(cell, point)
                        += _normals(cell, point, i)
                           * _metric_tensor(cell, point, i, j)
                           * _normals(cell, point, j);
                }
            }
            _penalty_param(cell, point) = 1.0
                                          / sqrt(_penalty_param(cell, point));
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETERMETRICTENSOR_IMPL_HPP
