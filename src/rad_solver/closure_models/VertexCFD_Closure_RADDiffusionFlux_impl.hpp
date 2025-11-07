#ifndef VERTEXCFD_CLOSURE_RADDIFFUSIONFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADDIFFUSIONFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
RADDiffusionFlux<EvalType, Traits>::RADDiffusionFlux(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop,
    const std::string& flux_prefix,
    const std::string& gradient_prefix)
    : _num_species(species_prop.numSpecies())
    , _num_space_dim(ir.spatial_dimension)
    , _nu(species_prop.constantDiffusionCoefficient())
{
    // Evaluated fields
    Utils::addEvaluatedScalarFieldView(*this, ir.dl_vector, _num_species, _species_flux,
        flux_prefix + "DIFFUSION_FLUX_"
                      "reaction_advection_diffusion_equation_");

    // Dependent fields
    Utils::addDependentScalarFieldView(*this,
                                       ir.dl_vector,
                                       _num_species,
                                       _grad_species,
                                       gradient_prefix + "GRAD_species_");

    this->setName("RAD Equation Diffusion Flux "
                  + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADDiffusionFlux<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void RADDiffusionFlux<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_species[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < _num_space_dim; ++dim)
            {
                for (int num = 0; num < _num_species; ++num)
                {
                    _species_flux[num](cell, point, dim)
                        = _nu * _grad_species[num](cell, point, dim);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADDIFFUSIONFLUX_IMPL_HPP
