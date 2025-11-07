#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ConductionTimeDerivative<EvalType, Traits>::ConductionTimeDerivative(
    const panzer::IntegrationRule& ir)
    : _dqdt_energy("DQDT_energy", ir.dl_scalar)
    , _density("solid_density", ir.dl_scalar)
    , _specific_heat("solid_specific_heat_capacity", ir.dl_scalar)
    , _dxdt_temperature("DXDT_temperature", ir.dl_scalar)
{
    // Dependent fields
    this->addDependentField(_dxdt_temperature);
    this->addDependentField(_density);
    this->addDependentField(_specific_heat);

    // Evaluated field
    this->addEvaluatedField(_dqdt_energy);

    this->setName("Conduction Time Derivative");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ConductionTimeDerivative<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ConductionTimeDerivative<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _density.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _dqdt_energy(cell, point) = _density(cell, point)
                                        * _specific_heat(cell, point)
                                        * _dxdt_temperature(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_IMPL_HPP
