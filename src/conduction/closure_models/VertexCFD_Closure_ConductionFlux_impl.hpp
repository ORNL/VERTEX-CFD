#ifndef VERTEXCFD_CLOSURE_CONDUCTIONFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONFLUX_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ConductionFlux<EvalType, Traits>::ConductionFlux(
    const panzer::IntegrationRule& ir,
    const std::string& flux_prefix,
    const std::string& gradient_prefix)
    : _conduction_flux(flux_prefix + "CONDUCTION_FLUX_energy", ir.dl_vector)
    , _grad_temperature(gradient_prefix + "GRAD_temperature", ir.dl_vector)
    , _thermal_conductivity("thermal_conductivity", ir.dl_scalar)
    , _num_space_dim(ir.spatial_dimension)
{
    // Evaluated fields
    this->addEvaluatedField(_conduction_flux);

    // Dependent fields
    this->addDependentField(_grad_temperature);
    this->addDependentField(_thermal_conductivity);

    this->setName("Conduction Flux " + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ConductionFlux<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void ConductionFlux<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _conduction_flux.extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             for (int dim = 0; dim < _num_space_dim; ++dim)
                             {
                                 _conduction_flux(cell, point, dim)
                                     = _thermal_conductivity(cell, point)
                                       * _grad_temperature(cell, point, dim);
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONFLUX_IMPL_HPP
