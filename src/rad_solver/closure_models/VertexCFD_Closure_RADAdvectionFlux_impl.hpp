#ifndef VERTEXCFD_CLOSURE_RADADVECTIONFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADADVECTIONFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
RADAdvectionFlux<EvalType, Traits, NumSpaceDim>::RADAdvectionFlux(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop,
    const std::string& flux_prefix,
    const std::string& field_prefix)
    : _num_species(species_prop.numSpecies())
{
    // Evaluated fields
    Utils::addEvaluatedScalarFieldView(*this, ir.dl_vector, _num_species, _species_flux,
        flux_prefix + "ADVECTION_FLUX_"
                      "reaction_advection_diffusion_equation_");

    // Dependent fields
    Utils::addDependentScalarFieldView(
        *this, ir.dl_scalar, _num_species, _species, field_prefix + "species_");

    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");

    this->setName("RAD Equation Advection Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void RADAdvectionFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
RADAdvectionFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _species[0].extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             for (int dim = 0; dim < num_space_dim; ++dim)
                             {
                                 for (int sp = 0; sp < _num_species; ++sp)
                                 {
                                     _species_flux[sp](cell, point, dim)
                                         = _species[sp](cell, point)
                                           * _velocity[dim](cell, point);
                                 }
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADADVECTIONFLUX_IMPL_HPP
