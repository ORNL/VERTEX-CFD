#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARVISCOUSFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARVISCOUSFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFScalarViscousFlux<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFScalarViscousFlux(const panzer::IntegrationRule& ir,
                                         const std::string& scalar_name,
                                         const std::string& equation_name,
                                         const std::string& flux_prefix,
                                         const std::string& gradient_prefix)
    : _scalar_flux(flux_prefix + "VISCOUS_FLUX_" + equation_name, ir.dl_vector)
    , _artificial_viscosity("artificial_viscosity", ir.dl_scalar)
    , _grad_scalar(gradient_prefix + "GRAD_" + scalar_name, ir.dl_vector)
{
    // Evaluated fields
    this->addEvaluatedField(_scalar_flux);

    // Dependent fields
    this->addDependentField(_artificial_viscosity);
    this->addDependentField(_grad_scalar);

    this->setName(equation_name + " Incompressible Viscous Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFScalarViscousFlux<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFScalarViscousFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _scalar_flux.extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             for (int dim = 0; dim < num_space_dim; ++dim)
                             {
                                 _scalar_flux(cell, point, dim)
                                     = _artificial_viscosity(cell, point)
                                       * _grad_scalar(cell, point, dim);
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARVISCOUSFLUX_IMPL_HPP
