#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_IMPL_HPP

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
IncompressibleConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    IncompressibleConvectiveFlux(const panzer::IntegrationRule& ir,
                                 const Teuchos::ParameterList& fluid_params,
                                 const std::string& flux_prefix,
                                 const std::string& field_prefix)
    : _continuity_flux(flux_prefix + "CONVECTIVE_FLUX_continuity", ir.dl_vector)
    , _energy_flux(flux_prefix + "CONVECTIVE_FLUX_energy", ir.dl_vector)
    , _energy_source("NON_CONSERVATIVE_SOURCE_energy", ir.dl_scalar)
    , _lagrange_pressure(field_prefix + "lagrange_pressure", ir.dl_scalar)
    , _rho("density", ir.dl_scalar)
    , _cp("specific_heat_capacity", ir.dl_scalar)
    , _temperature(field_prefix + "temperature", ir.dl_scalar)
    , _grad_temp("GRAD_temperature", ir.dl_vector)
    , _beta(fluid_params.get<double>("Artificial compressibility"))
    , _solve_temp(fluid_params.get<bool>("Build Temperature Equation"))
    , _continuity_model(ConModel::AC)
{
    // Get continuity model type
    if (fluid_params.isType<std::string>("Continuity Model"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<ConModel>(
                Teuchos::tuple<std::string>("AC", "EDAC", "EDACTempNC"), "AC"));
        _continuity_model = type_validator->getIntegralValue(
            fluid_params.get<std::string>("Continuity Model"));
    }

    // Evaluated fields
    this->addEvaluatedField(_continuity_flux);

    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _momentum_flux,
                                 flux_prefix + "CONVECTIVE_FLUX_"
                                               "momentum_");

    if (_solve_temp)
    {
        if (_continuity_model == ConModel::EDACTempNC)
        {
            this->addEvaluatedField(_energy_source);
        }
        else
        {
            this->addEvaluatedField(_energy_flux);
        }
    }

    // Dependent fields
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");
    this->addDependentField(_lagrange_pressure);
    this->addDependentField(_rho);

    if (_solve_temp)
    {
        this->addDependentField(_cp);
        this->addDependentField(_temperature);
        if (_continuity_model == ConModel::EDACTempNC)
        {
            this->addDependentField(_grad_temp);
        }
    }

    this->setName("Incompressible Convective Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleConvectiveFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleConvectiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _continuity_flux.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            if (_solve_temp && _continuity_model == ConModel::EDACTempNC)
                _energy_source(cell, point) = 0.0;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const auto vel_dim = _velocity[dim](cell, point);
                _continuity_flux(cell, point, dim) = _rho(cell, point)
                                                     * vel_dim;

                if (_continuity_model != ConModel::AC)
                {
                    _continuity_flux(cell, point, dim)
                        += _lagrange_pressure(cell, point) * vel_dim / _beta;
                }

                for (int mom_dim = 0; mom_dim < num_space_dim; ++mom_dim)
                {
                    _momentum_flux[mom_dim](cell, point, dim)
                        = _rho(cell, point) * vel_dim
                          * _velocity[mom_dim](cell, point);
                    if (mom_dim == dim)
                    {
                        _momentum_flux[mom_dim](cell, point, dim)
                            += _lagrange_pressure(cell, point);
                    }
                }

                if (_solve_temp)
                {
                    if (_continuity_model == ConModel::EDACTempNC)
                    {
                        _energy_source(cell, point)
                            += _rho(cell, point) * _cp(cell, point)
                               * _grad_temp(cell, point, dim)
                               * _velocity[dim](cell, point);
                    }
                    else
                    {
                        _energy_flux(cell, point, dim)
                            = _rho(cell, point) * _cp(cell, point) * vel_dim
                              * _temperature(cell, point);
                    }
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_IMPL_HPP
