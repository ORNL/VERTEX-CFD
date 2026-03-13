#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleViscousFlux<EvalType, Traits, NumSpaceDim>::IncompressibleViscousFlux(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& fluid_params,
    const Teuchos::ParameterList& turb_params,
    const std::string& flux_prefix,
    const std::string& gradient_prefix)
    : _continuity_flux(flux_prefix + "VISCOUS_FLUX_continuity", ir.dl_vector)
    , _energy_flux(flux_prefix + "VISCOUS_FLUX_energy", ir.dl_vector)
    , _rho("density", ir.dl_scalar)
    , _nu("kinematic_viscosity", ir.dl_scalar)
    , _cp("specific_heat_capacity", ir.dl_scalar)
    , _k("thermal_conductivity", ir.dl_scalar)
    , _grad_press(gradient_prefix + "GRAD_lagrange_pressure", ir.dl_vector)
    , _grad_temp(gradient_prefix + "GRAD_temperature", ir.dl_vector)
    , _nu_t(flux_prefix + "turbulent_eddy_viscosity", ir.dl_scalar)
    , _gamma(std::numeric_limits<double>::quiet_NaN())
    , _beta(fluid_params.get<double>("Artificial compressibility"))
    , _solve_temp(fluid_params.get<bool>("Build Temperature Equation"))
    , _use_turbulence_model(turb_params.get<bool>("Use Turbulence Model"))
    , _continuity_model_name(fluid_params.isType<std::string>("Continuity "
                                                              "Model")
                                 ? fluid_params.get<std::string>("Continuity "
                                                                 "Model")
                                 : "AC")
    , _is_edac(_continuity_model_name == "EDAC" ? true : false)
    , _Pr_t(_solve_temp
                ? (turb_params.isType<double>("Turbulent Prandtl Number")
                       ? turb_params.get<double>("Turbulent Prandtl Number")
                       : 0.85)
                : std::numeric_limits<double>::quiet_NaN())
{
    // Add evaludated fields
    this->addEvaluatedField(_continuity_flux);
    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _momentum_flux,
                                 flux_prefix + "VISCOUS_FLUX_"
                                               "momentum_");
    this->addEvaluatedField(_energy_flux);

    // Add dependent fields
    Utils::addDependentVectorField(*this,
                                   ir.dl_vector,
                                   _grad_velocity,
                                   gradient_prefix + "GRAD_velocity_");

    this->addDependentField(_rho);
    this->addDependentField(_nu);
    if (_is_edac)
    {
        this->addDependentField(_grad_press);
        if (_solve_temp)
            _gamma = fluid_params.get<double>("Heat capacity ratio");
    }

    if (_solve_temp)
    {
        this->addDependentField(_k);
        this->addDependentField(_cp);
        this->addDependentField(_grad_temp);
    }

    if (_use_turbulence_model)
    {
        this->addDependentField(_nu_t);
    }

    this->setName("Incompressible Viscous Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleViscousFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleViscousFlux<EvalType, Traits, NumSpaceDim>::operator()(
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
                    if (_solve_temp)
                    {
                        _continuity_flux(cell, point, i)
                            = _gamma * _k(cell, point)
                              * _grad_press(cell, point, i)
                              / (_beta * _rho(cell, point) * _cp(cell, point));
                    }
                    else
                    {
                        _continuity_flux(cell, point, i)
                            = _rho(cell, point) * _nu(cell, point)
                              * _grad_press(cell, point, i) / _beta;
                    }
                }
                // Set stress tensor to zero for AC continuity model
                else
                {
                    _continuity_flux(cell, point, i) = 0.0;
                }

                // Temperature equation
                if (_solve_temp)
                {
                    _energy_flux(cell, point, i)
                        = _k(cell, point) * _grad_temp(cell, point, i);
                    if (_use_turbulence_model)
                    {
                        _energy_flux(cell, point, i)
                            += _nu_t(cell, point) * _rho(cell, point)
                               * _cp(cell, point) * _grad_temp(cell, point, i)
                               / _Pr_t;
                    }
                }

                // Loop over velocity/momentum components
                for (int j = 0; j < num_space_dim; ++j)
                {
                    _momentum_flux[j](cell, point, i)
                        = _rho(cell, point) * _nu(cell, point)
                          * _grad_velocity[j](cell, point, i);
                    if (_use_turbulence_model)
                    {
                        _momentum_flux[j](cell, point, i)
                            += _rho(cell, point) * _nu_t(cell, point)
                               * _grad_velocity[j](cell, point, i);
                    }
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_IMPL_HPP
