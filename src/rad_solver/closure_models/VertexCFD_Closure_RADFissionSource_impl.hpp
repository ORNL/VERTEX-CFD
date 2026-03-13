#ifndef VERTEXCFD_CLOSURE_RADFISSIONSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADFISSIONSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
RADFissionSource<EvalType, Traits>::RADFissionSource(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop,
    const std::string& neutron_flux_name)
    : _num_species(species_prop.numSpecies())
    , _neutron_flux(neutron_flux_name, ir.dl_scalar)
    , _gamma(Kokkos::ViewAllocateWithoutInitializing("atoms_per_species"),
             _num_species)
    , _xs(species_prop.fissionCrossSection())
    , _avagadro(6.02214076e23)
{
    // Copy number of atoms vector from host to device
    auto gamma_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _gamma);
    gamma_host = species_prop.atomsPerSpecies();
    Kokkos::deep_copy(_gamma, gamma_host);

    // Evaluated fields
    Utils::addEvaluatedScalarFieldView(*this,
                                       ir.dl_scalar,
                                       _num_species,
                                       _fission_source,
                                       "FISSION_SOURCE_reaction_advection_"
                                       "diffusion_equation_");

    // Dependent fields
    this->addDependentField(_neutron_flux);

    this->setName("RAD Fission Source Term");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADFissionSource<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void RADFissionSource<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _fission_source[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int num = 0; num < _num_species; ++num)
            {
                _fission_source[num](cell, point) = _neutron_flux(cell, point)
                                                    * _gamma[num] * _xs
                                                    / _avagadro;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADFISSIONSOURCE_IMPL_HPP
