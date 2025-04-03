#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleViscousFlux<EvalType, Traits, NumSpaceDim>::IncompressibleViscousFlux(
    const panzer::IntegrationRule& ir,
    const FluidProperties::ConstantFluidProperties& fluid_prop,
    const Teuchos::ParameterList& user_params,
    const bool use_turbulence_model,
    const std::string& flux_prefix,
    const std::string& gradient_prefix)
    : _continuity_flux(flux_prefix + "VISCOUS_FLUX_continuity", ir.dl_vector)
    , _energy_flux(flux_prefix + "VISCOUS_FLUX_energy", ir.dl_vector)
    , _grad_press(gradient_prefix + "GRAD_lagrange_pressure", ir.dl_vector)
    , _grad_temp(gradient_prefix + "GRAD_temperature", ir.dl_vector)
    , _nu_t(flux_prefix + "turbulent_eddy_viscosity", ir.dl_scalar)
    , _rho(fluid_prop.constantDensity())
    , _nu(fluid_prop.constantKinematicViscosity())
    , _kappa(std::numeric_limits<double>::quiet_NaN())
    , _beta(fluid_prop.artificialCompressibility())
    , _solve_temp(fluid_prop.solveTemperature())
    , _use_turbulence_model(use_turbulence_model)
    , _continuity_model_name(user_params.isType<std::string>("Continuity "
                                                             "Model")
                                 ? user_params.get<std::string>("Continuity "
                                                                "Model")
                                 : "AC")
    , _is_edac(_continuity_model_name == "EDAC" ? true : false)
    , _rhoCp(fluid_prop.constantHeatCapacity())
    , _Pr_t(_solve_temp
                ? (user_params.isType<double>("Turbulent Prandtl Number")
                       ? user_params.get<double>("Turbulent Prandtl Number")
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
    if (_is_edac)
    {
        this->addDependentField(_grad_press);
    }
    if (_solve_temp)
    {
        _kappa = fluid_prop.constantThermalConductivity();
        this->addDependentField(_grad_temp);
    }

    Utils::addDependentVectorField(*this,
                                   ir.dl_vector,
                                   _grad_velocity,
                                   gradient_prefix + "GRAD_velocity_");

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
    const double ratio_kappa_t_nu_t = _rhoCp / _Pr_t;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Loop over spatial dimension
            for (int i = 0; i < num_space_dim; ++i)
            {
                // Set stress tensor for EDAC continuity model
                if (_is_edac)
                {
                    _continuity_flux(cell, point, i)
                        = _rho * _nu * _grad_press(cell, point, i) / _beta;
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
                        = _kappa * _grad_temp(cell, point, i);
                    if (_use_turbulence_model)
                    {
                        _energy_flux(cell, point, i)
                            += _nu_t(cell, point) * ratio_kappa_t_nu_t
                               * _grad_temp(cell, point, i);
                    }
                }

                // Loop over velocity/momentum components
                for (int j = 0; j < num_space_dim; ++j)
                {
                    _momentum_flux[j](cell, point, i)
                        = _rho * _nu * _grad_velocity[j](cell, point, i);
                    if (_use_turbulence_model)
                    {
                        _momentum_flux[j](cell, point, i)
                            += _rho * _nu_t(cell, point)
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
