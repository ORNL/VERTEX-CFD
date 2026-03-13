#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleSUPGFlux<EvalType, Traits, NumSpaceDim>::IncompressibleSUPGFlux(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& fluid_params,
    const Teuchos::ParameterList& closure_params)
    : _continuity_flux("VISCOUS_FLUX_continuity", ir.dl_vector)
    , _energy_flux("VISCOUS_FLUX_energy", ir.dl_vector)
    , _use_source(closure_params.isType<std::string>("Prefix for source "
                                                     "terms"))
    , _source_prefix(_use_source
                         ? closure_params.get<std::string>("Prefix for source "
                                                           "terms")
                         : "")
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _rho("density", ir.dl_scalar)
    , _cp("specific_heat_capacity", ir.dl_scalar)
    , _tau_supg_cont("tau_supg_continuity", ir.dl_scalar)
    , _tau_supg_mom("tau_supg_momentum", ir.dl_scalar)
    , _tau_supg_energy("tau_supg_energy", ir.dl_scalar)
    , _dxdt_temperature("DXDT_temperature", ir.dl_scalar)
    , _temperature("temperature", ir.dl_scalar)
    , _energy_source(_source_prefix + "_SOURCE_energy", ir.dl_scalar)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _solve_temp(fluid_params.get<bool>("Build Temperature Equation"))
{
    // Add contributed fields
    this->addContributedField(_continuity_flux);
    this->addContributedField(_energy_flux);
    Utils::addContributedVectorField(*this,
                                     ir.dl_vector,
                                     _momentum_flux,
                                     "VISCOUS_FLUX_"
                                     "momentum_");

    // Add dependent fields
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _dxdt_velocity, "DXDT_velocity_");
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");
    if (_use_source)
    {
        Utils::addDependentVectorField(*this,
                                   ir.dl_scalar,
                                   _momentum_source,
                                   _source_prefix + "_SOURCE_"
                                   "momentum_");
    }
    this->addDependentField(_rho);
    this->addDependentField(_grad_lagrange_pressure);
    this->addDependentField(_tau_supg_cont);
    this->addDependentField(_tau_supg_mom);
    if (_solve_temp)
    {
        this->addDependentField(_cp);
        this->addDependentField(_dxdt_temperature);
        this->addDependentField(_temperature);
        this->addDependentField(_grad_temperature);
        if (_use_source)
            this->addDependentField(_energy_source);
        this->addDependentField(_tau_supg_energy);
    }

    this->setName("Incompressible SUPG Flux " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleSUPGFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries in parallel for
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg_mom.get_view());
        bytes = scratch_view::shmem_size(
            _tau_supg_mom.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_tau_supg_mom.extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleSUPGFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _tau_supg_mom.extent(1);

    // Initialize scratch memory
    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg_mom.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& vel_res = tmp_field(point, VEL_RES);
            auto&& temp_res = tmp_field(point, TEMP_RES);

            // Compute strong residual of temperature equation
            // (non-conservative version)
            if (_solve_temp)
            {
                temp_res = _rho(cell, point) * _cp(cell, point)
                           * _dxdt_temperature(cell, point);
                if (_use_source)
                    temp_res -= _energy_source(cell, point);

                for (int i = 0; i < num_space_dim; ++i)
                {
                    temp_res += _rho(cell, point) * _cp(cell, point)
                                * _velocity[i](cell, point)
                                * _grad_temperature(cell, point, i);
                }
            }

            // Loop over cartesian components, build momentum residual, and add
            // SUPG contribution to all equations
            for (int j = 0; j < num_space_dim; ++j)
            {
                // Compute strong residual for j-momentum equation
                // (non-conservative version)
                vel_res = _dxdt_velocity[j](cell, point);
                for (int i = 0; i < num_space_dim; ++i)
                {
                    vel_res += _velocity[i](cell, point)
                               * _grad_velocity[j](cell, point, i);
                }
                vel_res *= _rho(cell, point);
                vel_res += _grad_lagrange_pressure(cell, point, j);
                if (_use_source)
                    vel_res -= _momentum_source[j](cell, point);

                // Add SUPG flux to viscous flux for each equation
                _continuity_flux(cell, point, j) += _tau_supg_cont(cell, point)
                                                    * vel_res
                                                    / _rho(cell, point);

                for (int i = 0; i < num_space_dim; ++i)
                {
                    _momentum_flux[j](cell, point, i)
                        += _tau_supg_mom(cell, point) * vel_res
                           * _velocity[i](cell, point);
                }

                if (_solve_temp)
                {
                    _energy_flux(cell, point, j)
                        += _tau_supg_energy(cell, point) * temp_res
                           * _velocity[j](cell, point);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGFLUX_IMPL_HPP
