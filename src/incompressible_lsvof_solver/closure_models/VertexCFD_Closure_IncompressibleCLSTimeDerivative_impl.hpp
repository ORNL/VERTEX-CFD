#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSTIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSTIMEDERIVATIVE_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleCLSTimeDerivative<EvalType, Traits>::IncompressibleCLSTimeDerivative(
    const panzer::IntegrationRule& ir)
    : _dqdt_sign("DQDT_level_set_equation", ir.dl_scalar)
    , _sign("CLS_sign", ir.dl_scalar)
    , _star_sign("STAR_CLS_sign", ir.dl_scalar)
{
    // Evaluated fields
    this->addEvaluatedField(_dqdt_sign);

    // Dependent fields
    this->addDependentField(_sign);
    this->addDependentField(_star_sign);

    this->setName("Incompressible CLS Time Derivative");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleCLSTimeDerivative<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Set dt = 1.0 / alpha
    // NOTE: this has been verified for the backward Euler and SDIRK family
    // of time steppers in Tempus, but may not hold true for all time steppers.
    _dt = 1.0 / workset.alpha;

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleCLSTimeDerivative<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dqdt_sign.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _dqdt_sign(cell, point)
                = (_sign(cell, point) - _star_sign(cell, point)) / _dt;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSTIMEDERIVATIVE_IMPL_HPP
