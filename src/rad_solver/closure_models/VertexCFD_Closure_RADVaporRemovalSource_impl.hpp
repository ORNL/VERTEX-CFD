#ifndef VERTEXCFD_CLOSURE_RADVAPORREMOVALSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADVAPORREMOVALSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
RADVaporRemovalSource<EvalType, Traits>::RADVaporRemovalSource(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop,
    const std::string& vapor_removal_ratio_name)
    : _num_species(species_prop.numSpecies())
    , _source(vapor_removal_ratio_name, ir.dl_scalar)
    , _multiplier(Kokkos::ViewAllocateWithoutInitializing("vapor_removal_"
                                                          "multiplier"),
                  _num_species)
{
    // Copy vapor removal ratio multiplier vector from host to device
    auto multiplier_host
        = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _multiplier);
    multiplier_host = species_prop.vaporRemovalRatioMultiplier();
    Kokkos::deep_copy(_multiplier, multiplier_host);

    // Evaluated fields
    Utils::addEvaluatedScalarFieldView(*this,
                                       ir.dl_scalar,
                                       _num_species,
                                       _removal_source,
                                       "REMOVAL_SOURCE_reaction_advection_"
                                       "diffusion_equation_");

    // Dependent fields
    this->addDependentField(_source);
    Utils::addDependentScalarFieldView(
        *this, ir.dl_scalar, _num_species, _species, "species_");

    this->setName("Vapor Removal Source Term");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADVaporRemovalSource<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void RADVaporRemovalSource<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _removal_source[0].extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             for (int num = 0; num < _num_species; ++num)
                                 _removal_source[num](cell, point)
                                     = _multiplier[num] * _source(cell, point)
                                       * _species[num](cell, point);
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADVAPORREMOVALSOURCE_IMPL_HPP
