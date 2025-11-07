#ifndef VERTEXCFD_CLOSURE_RADTIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADTIMEDERIVATIVE_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
RADTimeDerivative<EvalType, Traits>::RADTimeDerivative(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop)
    : _num_species(species_prop.numSpecies())
{
    // Evaluated variable
    Utils::addEvaluatedScalarFieldView(*this,
                                       ir.dl_scalar,
                                       _num_species,
                                       _dqdt_rad,
                                       "DQDT_reaction_advection_diffusion_"
                                       "equation_");

    // Dependent variable
    Utils::addDependentScalarFieldView(
        *this, ir.dl_scalar, _num_species, _dxdt_species, "DXDT_species_");

    // Closure model name
    this->setName("RAD Time Derivative");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADTimeDerivative<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void RADTimeDerivative<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dqdt_rad[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int num = 0; num < _num_species; ++num)
                _dqdt_rad[num](cell, point) = _dxdt_species[num](cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADTIMEDERIVATIVE_IMPL_HPP
