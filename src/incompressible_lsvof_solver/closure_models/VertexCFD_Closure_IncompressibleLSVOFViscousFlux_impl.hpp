#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVISCOUSFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVISCOUSFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFViscousFlux<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFViscousFlux(const panzer::IntegrationRule& ir,
                                   const Teuchos::ParameterList& closure_params,
                                   const Teuchos::ParameterList& user_params,
                                   const std::string& flux_prefix,
                                   const std::string& gradient_prefix,
                                   const std::string& field_prefix)
    : _continuity_flux(flux_prefix + "VISCOUS_FLUX_continuity", ir.dl_vector)
    , _grad_press(gradient_prefix + "GRAD_lagrange_pressure", ir.dl_vector)
    , _rho(field_prefix + "density", ir.dl_scalar)
    , _mu(field_prefix + "dynamic_viscosity", ir.dl_scalar)
    , _betam(closure_params.get<double>("Mixture Artificial Compressibility"))
    , _continuity_model_name(user_params.isType<std::string>("Continuity "
                                                             "Model")
                                 ? user_params.get<std::string>("Continuity "
                                                                "Model")
                                 : "AC")
    , _is_edac(_continuity_model_name == "EDAC" ? true : false)

{
    // Add evaludated fields
    this->addEvaluatedField(_continuity_flux);
    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _momentum_flux,
                                 flux_prefix + "VISCOUS_FLUX_"
                                               "momentum_");

    // Add dependent fields
    this->addDependentField(_rho);
    this->addDependentField(_mu);

    if (_is_edac)
    {
        this->addDependentField(_grad_press);
    }

    Utils::addDependentVectorField(*this,
                                   ir.dl_vector,
                                   _grad_velocity,
                                   gradient_prefix + "GRAD_velocity_");

    this->setName("Incompressible LSVOF Viscous Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFViscousFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFViscousFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _continuity_flux.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Loop over spatial dimension
            for (int i = 0; i < num_space_dim; ++i)
            {
                // Set stress tensor for EDAC continuity model
                if (_is_edac)
                {
                    _continuity_flux(cell, point, i)
                        = _mu(cell, point) * _grad_press(cell, point, i)
                          / _betam;
                }
                // Set stress tensor to zero for AC continuity model
                else
                {
                    _continuity_flux(cell, point, i) = 0.0;
                }

                // Loop over velocity/momentum components
                for (int j = 0; j < num_space_dim; ++j)
                {
                    _momentum_flux[j](cell, point, i)
                        = _mu(cell, point)
                          * (_grad_velocity[j](cell, point, i)
                             + _grad_velocity[i](cell, point, j));
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVISCOUSFLUX_IMPL_HPP
