#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFCONVECTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFCONVECTIVEFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFConvectiveFlux(const panzer::IntegrationRule& ir,
                                      const std::string& flux_prefix,
                                      const std::string& field_prefix)
    : _continuity_flux(flux_prefix + "CONVECTIVE_FLUX_continuity", ir.dl_vector)
    , _rho(field_prefix + "density", ir.dl_scalar)
    , _pressure(field_prefix + "lagrange_pressure", ir.dl_scalar)
{
    // Evaluated fields
    this->addEvaluatedField(_continuity_flux);

    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _momentum_flux,
                                 flux_prefix + "CONVECTIVE_FLUX_"
                                               "momentum_");

    // Dependent fields
    this->addDependentField(_rho);
    this->addDependentField(_pressure);
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");

    this->setName("Incompressible LSVOF Convective Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFConvectiveFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFConvectiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _continuity_flux.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _continuity_flux(cell, point, dim)
                    = _rho(cell, point) * _velocity[dim](cell, point);

                for (int mom_dim = 0; mom_dim < num_space_dim; ++mom_dim)
                {
                    _momentum_flux[mom_dim](cell, point, dim)
                        = _rho(cell, point) * _velocity[dim](cell, point)
                          * _velocity[mom_dim](cell, point);
                    if (mom_dim == dim)
                    {
                        _momentum_flux[mom_dim](cell, point, dim)
                            += _pressure(cell, point);
                    }
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFCONVECTIVEFLUX_IMPL_HPP
