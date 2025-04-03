#ifndef VERTEXCFD_CLOSURE_MAGNETICCORRECTIONDAMPINGSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_MAGNETICCORRECTIONDAMPINGSOURCE_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
MagneticCorrectionDampingSource<EvalType, Traits>::MagneticCorrectionDampingSource(
    const panzer::IntegrationRule& ir,
    const MHDProperties::FullInductionMHDProperties& mhd_props)
    : _damping_potential_source("DAMPING_SOURCE_magnetic_correction_potential",
                                ir.dl_scalar)
    , _alpha(mhd_props.magneticCorrectionDampingFactor())
    , _scalar_magnetic_potential("scalar_magnetic_potential", ir.dl_scalar)
{
    // Evaluated fields
    this->addEvaluatedField(_damping_potential_source);

    // Dependent fields
    this->addDependentField(_scalar_magnetic_potential);

    this->setName("Magnetic Correction Damping Source");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void MagneticCorrectionDampingSource<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
MagneticCorrectionDampingSource<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _scalar_magnetic_potential.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _damping_potential_source(cell, point)
                = -_alpha * _scalar_magnetic_potential(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_MAGNETICCORRECTIONDAMPINGSOURCE_IMPL_HPP
