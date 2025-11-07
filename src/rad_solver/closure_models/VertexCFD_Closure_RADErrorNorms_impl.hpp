#ifndef VERTEXCFD_CLOSURE_RADERRORNORMS_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADERRORNORMS_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_Array.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
RADErrorNorms<EvalType, Traits>::RADErrorNorms(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop)
    : _num_species(species_prop.numSpecies())
    , _volume("volume", ir.dl_scalar)
{
    // Dependent variables
    Utils::addDependentScalarFieldView(
        *this, ir.dl_scalar, _num_species, _exact_species, "Exact_species_");
    Utils::addDependentScalarFieldView(
        *this, ir.dl_scalar, _num_species, _species, "species_");

    // Evaluated variables
    Utils::addEvaluatedScalarFieldView(*this,
                                       ir.dl_scalar,
                                       _num_species,
                                       _L1_error_species,
                                       "L1_Error_reaction_advection_diffusion_"
                                       "equation_");
    Utils::addEvaluatedScalarFieldView(*this,
                                       ir.dl_scalar,
                                       _num_species,
                                       _L2_error_species,
                                       "L2_Error_reaction_advection_diffusion_"
                                       "equation_");
    this->addEvaluatedField(_volume);

    this->setName("RAD Error Norms");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADErrorNorms<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void RADErrorNorms<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _species[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using Kokkos::abs;
            using Kokkos::pow;
            for (int num = 0; num < _num_species; ++num)
            {
                _L1_error_species[num](cell, point)
                    = abs(_species[num](cell, point)
                          - _exact_species[num](cell, point));

                _L2_error_species[num](cell, point)
                    = pow(_species[num](cell, point)
                              - _exact_species[num](cell, point),
                          2);
            }
            _volume(cell, point) = 1.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_RADERRORNORMS_IMPL_HPP
