#ifndef VERTEXCFD_CLOSURE_RADREACTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADREACTION_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
RADReaction<EvalType, Traits>::RADReaction(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop,
    const std::string& flux_prefix,
    const std::string& field_prefix)
    : _num_species(species_prop.numSpecies())
    , _bateman_matrix(Kokkos::ViewAllocateWithoutInitializing("bateman_"
                                                              "matrix"),
                      _num_species,
                      _num_species)
{
    // Copy bateman matrix from host to device
    auto bateman_matrix_host
        = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _bateman_matrix);
    bateman_matrix_host = species_prop.batemanMatrix();
    Kokkos::deep_copy(_bateman_matrix, bateman_matrix_host);

    // Evaluated fields
    Utils::addEvaluatedScalarFieldView(
        *this,
        ir.dl_scalar,
        _num_species,
        _reaction_term,
        flux_prefix + "REACTION_TERM_reaction_advection_diffusion_equation_");

    // Dependent fields
    Utils::addDependentScalarFieldView(
        *this, ir.dl_scalar, _num_species, _species, field_prefix + "species_");

    this->setName("RAD Equation Reaction Term");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADReaction<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void RADReaction<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _reaction_term[0].extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             for (int num = 0; num < _num_species; ++num)
                             {
                                 _reaction_term[num](cell, point) = 0.0;
                                 for (int spe = 0; spe < _num_species; ++spe)
                                 {
                                     _reaction_term[num](cell, point)
                                         += _species[spe](cell, point)
                                            * _bateman_matrix(num, spe);
                                 }
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADREACTION_IMPL_HPP
