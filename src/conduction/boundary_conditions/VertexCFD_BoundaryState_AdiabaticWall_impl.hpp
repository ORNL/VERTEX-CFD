#ifndef VERTEXCFD_BOUNDARYSTATE_ADIABATICWALL_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_ADIABATICWALL_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <algorithm>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
AdiabaticWall<EvalType, Traits>::AdiabaticWall(const panzer::IntegrationRule& ir)
    : _boundary_temperature("BOUNDARY_temperature", ir.dl_scalar)
    , _boundary_grad_temperature("BOUNDARY_GRAD_temperature", ir.dl_vector)
    , _temperature("temperature", ir.dl_scalar)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
{
    // Evaluated fields
    this->addEvaluatedField(_boundary_temperature);
    this->addEvaluatedField(_boundary_grad_temperature);

    // Dependent fields
    this->addDependentField(_temperature);
    this->addDependentField(_grad_temperature);
    this->addDependentField(_normals);

    // Closure model names
    this->setName("Boundary State Adiabatic Wall");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void AdiabaticWall<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void AdiabaticWall<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_temperature.extent(1);
    const int num_grad_dim = _boundary_grad_temperature.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _boundary_temperature(cell, point) = _temperature(cell, point);

            for (int d = 0; d < num_grad_dim; ++d)
            {
                _boundary_grad_temperature(cell, point, d)
                    = _grad_temperature(cell, point, d);
                for (int dim = 0; dim < num_grad_dim; ++dim)
                {
                    _boundary_grad_temperature(cell, point, d)
                        -= (_grad_temperature(cell, point, dim)
                            * _normals(cell, point, dim))
                           * _normals(cell, point, d);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_ADIABATICWALL_IMPL_HPP
