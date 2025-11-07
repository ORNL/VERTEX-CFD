#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFNTHPHASEFRACTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFNTHPHASEFRACTION_IMPL_HPP

#include "utils/VertexCFD_Utils_PhaseLayout.hpp"
#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Phalanx_DataLayout_DynamicLayout.hpp>

#include <Sacado_Traits.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleLSVOFNthPhaseFraction<EvalType, Traits>::
    IncompressibleLSVOFNthPhaseFraction(
        const panzer::IntegrationRule& ir,
        const std::vector<std::string>& phase_names,
        const std::string& field_prefix)
    : _alpha_n(field_prefix + phase_names[phase_names.size() - 1], ir.dl_scalar)
    , _phase_layout(
          Utils::buildPhaseLayout(ir.dl_scalar, phase_names.size() - 1))
    , _alphas(field_prefix + "volume_fractions", _phase_layout)
{
    // Evaluated fields
    this->addEvaluatedField(_alpha_n);

    // Dependent fields
    this->addDependentField(_alphas);

    this->setName("Incompressible LSVOF Nth Phase Fraction");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleLSVOFNthPhaseFraction<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFNthPhaseFraction<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _alpha_n.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set alpha_n = 1.0 - alpha_1 - ... - alpha_{n-1}
            _alpha_n(cell, point) = 1.0;

            for (size_t phase = 0; phase < _alphas.extent(2); ++phase)
            {
                _alpha_n(cell, point) -= _alphas(cell, point, phase);
            }

            // Limit alpha_n to [0, 1]
            _alpha_n(cell, point) = SmoothMath::max(
                SmoothMath::min(_alpha_n(cell, point), 1.0, 0.0), 0.0, 0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFNTHPHASEFRACTION_IMPL_HPP
