#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_PhaseLayout.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFScalarConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFScalarConvectiveFlux(const panzer::IntegrationRule& ir,
                                            const int& phase_index,
                                            const int& num_lsvof_dofs,
                                            const std::string& equation_name,
                                            const std::string& flux_prefix,
                                            const std::string& field_prefix)
    : _scalar_flux(flux_prefix + "CONVECTIVE_FLUX_" + equation_name,
                   ir.dl_vector)
    , _phase_index(phase_index)
    , _phase_layout(Utils::buildPhaseLayout(ir.dl_scalar, num_lsvof_dofs))
    , _volume_fractions(field_prefix + "volume_fractions", _phase_layout)
{
    // Evaluated fields
    this->addEvaluatedField(_scalar_flux);

    // Dependent fields
    this->addDependentField(_volume_fractions);
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");

    this->setName(equation_name + " Incompressible Convective Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFScalarConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFScalarConvectiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _scalar_flux.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _scalar_flux(cell, point, dim)
                    = _volume_fractions(cell, point, _phase_index)
                      * _velocity[dim](cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_IMPL_HPP
