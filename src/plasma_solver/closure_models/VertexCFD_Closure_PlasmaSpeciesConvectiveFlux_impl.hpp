#ifndef VERTEXCFD_CLOSURE_PLASMASPECIESCONVECTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_PLASMASPECIESCONVECTIVEFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"
#include "utils/VertexCFD_Utils_VelocityLayout.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
PlasmaSpeciesConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    PlasmaSpeciesConvectiveFlux(const panzer::IntegrationRule& ir,
                                const Teuchos::ParameterList& closure_params,
                                const std::string& flux_prefix,
                                const std::string& field_prefix)
    : _species_name(closure_params.get<std::string>("Species Name"))
    , _res_name("_" + closure_params.get<std::string>("Residual Name") + "_")
    , _nd_flux(
          flux_prefix + _species_name + _res_name + "_number_density_equation",
          ir.dl_vector)
    , _energy_flux(flux_prefix + _species_name + _res_name + "_energy_equation",
                   ir.dl_vector)
    , _nd(field_prefix + _species_name + "_number_density", ir.dl_scalar)
    , _velocity(field_prefix + _species_name + "_velocity",
                Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
    , _rho(field_prefix + _species_name + "_momentum_density", ir.dl_scalar)
    , _p(field_prefix + _species_name + "_pressure", ir.dl_scalar)
    , _E(field_prefix + _species_name + "_energy", ir.dl_scalar)

{
    // Evaluated fields
    this->addEvaluatedField(_nd_flux);
    Utils::addEvaluatedVectorField(
        *this,
        ir.dl_vector,
        _momentum_flux,
        flux_prefix + _species_name + _res_name + "momentum_");
    this->addEvaluatedField(_energy_flux);

    // Dependent fields
    this->addDependentField(_nd);
    this->addDependentField(_velocity);
    this->addDependentField(_rho);
    this->addDependentField(_p);
    this->addDependentField(_E);

    this->setName("Species Convective Flux "
                  + std::to_string(ir.spatial_dimension) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void PlasmaSpeciesConvectiveFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
PlasmaSpeciesConvectiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nd_flux.extent(1);
    const int num_grad_dim = _nd_flux.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                // Number density equation
                _nd_flux(cell, point, dim) = _nd(cell, point)
                                             * _velocity(cell, point, dim);

                // Momentum equations
                for (int vel_dim = 0; vel_dim < num_grad_dim; ++vel_dim)
                {
                    _momentum_flux[dim](cell, point, vel_dim)
                        = _rho(cell, point) * _velocity(cell, point, dim)
                          * _velocity(cell, point, vel_dim);
                }
                _momentum_flux[dim](cell, point, dim) += _p(cell, point);

                // Energy equation
                _energy_flux(cell, point, dim)
                    = _velocity(cell, point, dim)
                      * (_E(cell, point) + _p(cell, point));
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_PLASMASPECIESCONVECTIVEFLUX_IMPL_HPP
