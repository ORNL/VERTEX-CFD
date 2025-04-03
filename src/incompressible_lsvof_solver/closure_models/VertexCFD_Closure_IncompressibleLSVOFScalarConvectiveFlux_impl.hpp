#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFScalarConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFScalarConvectiveFlux(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& flux_prefix,
        const std::string& field_prefix)
    : _scalar_name(closure_params.get<std::string>("Field Name"))
    , _scalar_equation_name(closure_params.get<std::string>("Equation Name"))
    , _scalar_flux(flux_prefix + "CONVECTIVE_FLUX_" + _scalar_equation_name,
                   ir.dl_vector)
    , _scalar(field_prefix + _scalar_name, ir.dl_scalar)
{
    // Evaluated fields
    this->addEvaluatedField(_scalar_flux);

    // Dependent fields
    this->addDependentField(_scalar);
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");

    this->setName(_scalar_equation_name + " Incompressible Convective Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFScalarConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFScalarConvectiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _scalar_flux.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _scalar_flux(cell, point, dim) = _scalar(cell, point)
                                                 * _velocity[dim](cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_IMPL_HPP
