#ifndef VERTEXCFD_CLOSURE_SOLIDELECTRICPOTENTIALDIFFUSIONFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_SOLIDELECTRICPOTENTIALDIFFUSIONFLUX_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
SolidElectricPotentialDiffusionFlux<EvalType, Traits>::
    SolidElectricPotentialDiffusionFlux(const panzer::IntegrationRule& ir,
                                        const std::string& flux_prefix,
                                        const std::string& gradient_prefix)
    : _electric_potential_flux(
          flux_prefix + "ELECTRIC_POTENTIAL_FLUX_electric_potential_equation",
          ir.dl_vector)
    , _grad_electric_potential(gradient_prefix + "GRAD_electric_potential",
                               ir.dl_vector)
    , _sigma("solid_electric_conductivity", ir.dl_scalar)
    , _num_grad_dim(ir.spatial_dimension)
{
    // Evaluated fields
    this->addEvaluatedField(_electric_potential_flux);

    // Dependent fields
    this->addDependentField(_grad_electric_potential);
    this->addDependentField(_sigma);

    this->setName("Solid Electric Potential Diffusion Flux "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void SolidElectricPotentialDiffusionFlux<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
SolidElectricPotentialDiffusionFlux<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _electric_potential_flux.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < _num_grad_dim; ++dim)
            {
                _electric_potential_flux(cell, point, dim)
                    = -_sigma(cell, point)
                      * _grad_electric_potential(cell, point, dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SOLIDELECTRICPOTENTIALDIFFUSIONFLUX_IMPL_HPP
